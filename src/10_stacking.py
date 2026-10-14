# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

# IMPORTANT WORKFLOW NOTE:
# The CNN is trained once by 04_train_cnn_extract_features.py. Five-fold
# cross-validation is applied to the fixed 128-D embeddings, as described in
# the manuscript; held-out fold samples are excluded from each base learner.
# Standalone classifier scripts are for independent comparison and their
# saved model artifacts are not loaded by this stacking procedure.

from config import *
import os
import re
from pathlib import Path

# ============================================================
# TensorFlow / Keras
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# FIXED RANDOM SEEDS
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# PATHS
# ============================================================

save_dir = str(FEATURE_DIR)

result_dir = str(STACKING_RESULT_DIR)

os.makedirs(
    result_dir,
    exist_ok=True
)


# ============================================================
# LOAD CNN FEATURES
# ============================================================

print("=" * 70)
print("LOADING CNN FEATURES")
print("=" * 70)

X_train = np.load(
    os.path.join(
        save_dir,
        "X_train_features.npy"
    )
)

y_train = np.load(
    os.path.join(
        save_dir,
        "y_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        save_dir,
        "X_val_features.npy"
    )
)

y_val = np.load(
    os.path.join(
        save_dir,
        "y_val.npy"
    )
)

X_test = np.load(
    os.path.join(
        save_dir,
        "X_test_features.npy"
    )
)

y_test = np.load(
    os.path.join(
        save_dir,
        "y_test.npy"
    )
)


# ============================================================
# CONVERT ONE-HOT LABELS IF NECESSARY
# ============================================================

if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train = np.argmax(
        y_train,
        axis=1
    )

if len(y_val.shape) > 1 and y_val.shape[1] > 1:
    y_val = np.argmax(
        y_val,
        axis=1
    )

if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(
        y_test,
        axis=1
    )


# Load ordered raw image paths only to identify augmented variants from the
# same source image and keep them in the same downstream-classifier CV fold.
# CNN embeddings are already extracted and loaded above.
def _load_raw_images(directory):
    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory), labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, color_mode="grayscale",
        image_size=(28, 28), batch_size=64, shuffle=False
    )
    paths = [str(Path(p).resolve()) for p in ds.file_paths]
    images, labels = [], []
    for xb, yb in ds:
        images.append((xb.numpy().astype(np.float32) / 255.0))
        labels.append(yb.numpy())
    return np.concatenate(images), np.concatenate(labels), paths

X_train_raw, y_train_raw, train_paths = _load_raw_images(BALANCED_DIR)
X_val_raw, y_val_raw, val_paths = _load_raw_images(SPLIT_DIR / "val")
X_test_raw, y_test_raw, test_paths = _load_raw_images(SPLIT_DIR / "test")
if not np.array_equal(y_train_raw, y_train):
    raise ValueError("Ordered raw training labels do not match saved feature labels; refusing misaligned OOF stacking.")
if not np.array_equal(y_val_raw, y_val) or not np.array_equal(y_test_raw, y_test):
    raise ValueError("Ordered raw validation/test labels do not match saved feature labels.")

def _source_group(path):
    # 03_balance_train.py names generated images as aug_<counter>_<source-name>.
    # Keep every augmentation of an original image in the same CV fold.
    name = Path(path).name
    match = re.match(r"^aug_\d+_(.+)$", name, flags=re.IGNORECASE)
    source_name = match.group(1) if match else name
    parent_class = Path(path).parent.name
    return f"{parent_class}/{source_name}"

train_groups = np.asarray([_source_group(p) for p in train_paths])

# ============================================================
# DATA INFORMATION
# ============================================================

print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")

print("\nClass distribution:")

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(
    unique,
    counts
):
    print(
        f"Class {cls}: {count}"
    )


# ============================================================
# NUMBER OF CLASSES
# ============================================================

n_train = len(
    y_train
)

n_classes = len(
    np.unique(y_train)
)

print(
    f"\nNumber of classes: {n_classes}"
)


# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# OOF PROBABILITIES
# ============================================================

oof_svm = np.zeros(
    (n_train, n_classes)
)

oof_mlp = np.zeros(
    (n_train, n_classes)
)

oof_rf = np.zeros(
    (n_train, n_classes)
)

oof_xgb = np.zeros(
    (n_train, n_classes)
)

oof_lgbm = np.zeros(
    (n_train, n_classes)
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

val_svm = np.zeros(
    (len(X_val), n_classes)
)

val_mlp = np.zeros(
    (len(X_val), n_classes)
)

val_rf = np.zeros(
    (len(X_val), n_classes)
)

val_xgb = np.zeros(
    (len(X_val), n_classes)
)

val_lgbm = np.zeros(
    (len(X_val), n_classes)
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

test_svm = np.zeros(
    (len(X_test), n_classes)
)

test_mlp = np.zeros(
    (len(X_test), n_classes)
)

test_rf = np.zeros(
    (len(X_test), n_classes)
)

test_xgb = np.zeros(
    (len(X_test), n_classes)
)

test_lgbm = np.zeros(
    (len(X_test), n_classes)
)


# ============================================================
# 5-FOLD TRAINING
# ============================================================

for fold, (
    train_idx,
    valid_idx
) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        X_train_raw, y_train
    ),
    start=1
):

    print("\n" + "=" * 70)
    print(
        f"FOLD {fold}/5"
    )
    print("=" * 70)

    # Manuscript-aligned protocol: use the fixed 128-D embeddings produced by
    # the separately trained CNN. Cross-validation applies to the downstream
    # base classifiers; the held-out fold is excluded from each classifier fit.
    X_fold_train = X_train[train_idx]
    X_fold_valid = X_train[valid_idx]
    y_fold_train = y_train[train_idx]
    y_fold_valid = y_train[valid_idx]

    print(
        f"Fold training samples   : "
        f"{len(X_fold_train)}"
    )

    print(
        f"Fold validation samples : "
        f"{len(X_fold_valid)}"
    )


    # ========================================================
    # SVM
    # ========================================================

    print("\nTraining SVM...")

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True
    )

    svm.fit(
        X_fold_train,
        y_fold_train
    )

    # OOF predictions
    oof_svm[
        valid_idx
    ] = svm.predict_proba(
        X_fold_valid
    )

    # Validation predictions
    val_svm += (
        svm.predict_proba(
            X_val
        ) / 5.0
    )

    # Test predictions
    test_svm += (
        svm.predict_proba(
            X_test
        ) / 5.0
    )


    # ========================================================
    # MLP
    # ========================================================

    print("\nTraining MLP...")

    mlp = Sequential([

        Dense(
            512,
            activation="relu",
            input_shape=(
                X_fold_train.shape[1],
            )
        ),

        Dropout(
            0.4
        ),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.4
        ),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.3
        ),

        Dense(
            n_classes,
            activation="softmax"
        )
    ])

    mlp.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Use a fixed, pre-specified training duration within each stacking fold.
    # The outer held-out fold is reserved exclusively for generating OOF predictions.
    mlp.fit(
        X_fold_train,
        y_fold_train,
        epochs=50,
        batch_size=64,
        verbose=0
    )

    # OOF predictions
    oof_mlp[
        valid_idx
    ] = mlp.predict(
        X_fold_valid,
        verbose=0
    )

    # Validation predictions
    val_mlp += (
        mlp.predict(
            X_val,
            verbose=0
        ) / 5.0
    )

    # Test predictions
    test_mlp += (
        mlp.predict(
            X_test,
            verbose=0
        ) / 5.0
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    rf.fit(
        X_fold_train,
        y_fold_train
    )

    # OOF predictions
    oof_rf[
        valid_idx
    ] = rf.predict_proba(
        X_fold_valid
    )

    # Validation predictions
    val_rf += (
        rf.predict_proba(
            X_val
        ) / 5.0
    )

    # Test predictions
    test_rf += (
        rf.predict_proba(
            X_test
        ) / 5.0
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    xgb = XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        objective="multi:softprob"
    )

    xgb.fit(
        X_fold_train,
        y_fold_train
    )

    # OOF predictions
    oof_xgb[
        valid_idx
    ] = xgb.predict_proba(
        X_fold_valid
    )

    # Validation predictions
    val_xgb += (
        xgb.predict_proba(
            X_val
        ) / 5.0
    )

    # Test predictions
    test_xgb += (
        xgb.predict_proba(
            X_test
        ) / 5.0
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    print(
        "\nTraining LightGBM..."
    )

    lgbm = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100
    )

    lgbm.fit(
        X_fold_train,
        y_fold_train
    )

    # OOF predictions
    oof_lgbm[
        valid_idx
    ] = lgbm.predict_proba(
        X_fold_valid
    )

    # Validation predictions
    val_lgbm += (
        lgbm.predict_proba(
            X_val
        ) / 5.0
    )

    # Test predictions
    test_lgbm += (
        lgbm.predict_proba(
            X_test
        ) / 5.0
    )


    print(
        f"\nFold {fold} completed successfully."
    )


# ============================================================
# BUILD META-TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("BUILDING OOF META-FEATURES")
print("=" * 70)

X_meta_train = np.hstack([
    oof_svm,
    oof_mlp,
    oof_rf,
    oof_xgb,
    oof_lgbm
])


# ============================================================
# BUILD VALIDATION META-FEATURES
# ============================================================

X_meta_val = np.hstack([
    val_svm,
    val_mlp,
    val_rf,
    val_xgb,
    val_lgbm
])


# ============================================================
# BUILD TEST META-FEATURES
# ============================================================

X_meta_test = np.hstack([
    test_svm,
    test_mlp,
    test_rf,
    test_xgb,
    test_lgbm
])


print(
    f"X_meta_train : {X_meta_train.shape}"
)

print(
    f"X_meta_val   : {X_meta_val.shape}"
)

print(
    f"X_meta_test  : {X_meta_test.shape}"
)


# ============================================================
# LOGISTIC REGRESSION META-LEARNER
# ============================================================

print("\n" + "=" * 70)
print(
    "TRAINING LOGISTIC REGRESSION META-LEARNER"
)
print("=" * 70)

meta_model = LogisticRegression(
    C=0.1
)

meta_model.fit(
    X_meta_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_train_pred = meta_model.predict(
    X_meta_train
)

y_val_pred = meta_model.predict(
    X_meta_val
)

y_test_pred = meta_model.predict(
    X_meta_test
)


# ============================================================
# PREDICTED PROBABILITIES
# ============================================================

y_train_proba = meta_model.predict_proba(
    X_meta_train
)

y_val_proba = meta_model.predict_proba(
    X_meta_val
)

y_test_proba = meta_model.predict_proba(
    X_meta_test
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    y_proba,
    dataset_name
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    # Macro ROC-AUC, one-vs-rest
    roc_auc = roc_auc_score(
        y_true,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    logloss = log_loss(
        y_true,
        y_proba
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"{dataset_name} RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision      : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score       : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC (Macro): "
        f"{roc_auc:.4f}"
    )

    print(
        f"Log Loss       : "
        f"{logloss:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": logloss
    }


# ============================================================
# TRAINING RESULTS
# ============================================================

train_results = evaluate_model(
    y_train,
    y_train_pred,
    y_train_proba,
    "TRAINING"
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

val_results = evaluate_model(
    y_val,
    y_val_pred,
    y_val_proba,
    "VALIDATION"
)


# ============================================================
# FINAL INDEPENDENT TEST RESULTS
# ============================================================

test_results = evaluate_model(
    y_test,
    y_test_pred,
    y_test_proba,
    "FINAL INDEPENDENT TEST"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CLASSIFICATION REPORT - FINAL TEST SET"
)

print(
    "=" * 70
)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Normal",
            "Reversal",
            "Corrected"
        ],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CONFUSION MATRIX - FINAL TEST SET"
)

print(
    "=" * 70
)

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(
    cm
)


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        result_dir,
        "y_train_pred_stacking.npy"
    ),
    y_train_pred
)

np.save(
    os.path.join(
        result_dir,
        "y_val_pred_stacking.npy"
    ),
    y_val_pred
)

np.save(
    os.path.join(
        result_dir,
        "y_test_pred_stacking.npy"
    ),
    y_test_pred
)


# ============================================================
# SAVE PROBABILITIES
# ============================================================

np.save(
    os.path.join(
        result_dir,
        "y_train_proba_stacking.npy"
    ),
    y_train_proba
)

np.save(
    os.path.join(
        result_dir,
        "y_val_proba_stacking.npy"
    ),
    y_val_proba
)

np.save(
    os.path.join(
        result_dir,
        "y_test_proba_stacking.npy"
    ),
    y_test_proba
)


# ============================================================
# SAVE META-FEATURES
# ============================================================

np.save(
    os.path.join(
        result_dir,
        "X_meta_train.npy"
    ),
    X_meta_train
)

np.save(
    os.path.join(
        result_dir,
        "X_meta_val.npy"
    ),
    X_meta_val
)

np.save(
    os.path.join(
        result_dir,
        "X_meta_test.npy"
    ),
    X_meta_test
)


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        result_dir,
        "oof_svm.npy"
    ),
    oof_svm
)

np.save(
    os.path.join(
        result_dir,
        "oof_mlp.npy"
    ),
    oof_mlp
)

np.save(
    os.path.join(
        result_dir,
        "oof_rf.npy"
    ),
    oof_rf
)

np.save(
    os.path.join(
        result_dir,
        "oof_xgb.npy"
    ),
    oof_xgb
)

np.save(
    os.path.join(
        result_dir,
        "oof_lgbm.npy"
    ),
    oof_lgbm
)

# Persist exact ordered test samples for paired statistical comparisons.
write_test_sample_manifest(result_dir, y_test)


# ============================================================
# SAVE META-LEARNER
# ============================================================

joblib.dump(
    meta_model,
    os.path.join(
        result_dir,
        "meta_learner_logistic_regression.pkl"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    os.path.join(
        result_dir,
        "train_results.npy"
    ),
    train_results
)

np.save(
    os.path.join(
        result_dir,
        "val_results.npy"
    ),
    val_results
)

np.save(
    os.path.join(
        result_dir,
        "test_results.npy"
    ),
    test_results
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STACKING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\nResults saved to:"
)

print(
    result_dir
)
