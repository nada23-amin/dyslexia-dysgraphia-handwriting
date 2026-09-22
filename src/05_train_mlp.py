# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *

# ============================================================
# DISABLE GPU
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(
    "Available devices:",
    tf.config.list_physical_devices()
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# PATHS
# ============================================================

base_path = str(PROJECT_ROOT)

features_path = str(FEATURE_DIR)

results_path = str(MLP_RESULT_DIR)

os.makedirs(
    results_path,
    exist_ok=True
)

# ============================================================
# LOAD CNN FEATURES
# ============================================================

X_train = np.load(
    os.path.join(
        features_path,
        "X_train_features.npy"
    )
)

X_val = np.load(
    os.path.join(
        features_path,
        "X_val_features.npy"
    )
)

X_test = np.load(
    os.path.join(
        features_path,
        "X_test_features.npy"
    )
)

y_train = np.load(
    os.path.join(
        features_path,
        "y_train.npy"
    )
)

y_val = np.load(
    os.path.join(
        features_path,
        "y_val.npy"
    )
)

y_test = np.load(
    os.path.join(
        features_path,
        "y_test.npy"
    )
)

# ============================================================
# DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA SHAPES")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

# ============================================================
# CLASSES
# ============================================================

categories = [
    "Normal",
    "Reversal",
    "Corrected"
]

num_classes = len(
    np.unique(y_train)
)

print(
    "\nNumber of classes:",
    num_classes
)

# ============================================================
# MLP MODEL
# As specified in the paper
# ============================================================

model = Sequential([

    Dense(
        512,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.4),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.4),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 60)
print("MLP MODEL")
print("=" * 60)

model.summary()

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING MLP TRAINING")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=50,
    batch_size=64,
    shuffle=True,
    verbose=1
)

# ============================================================
# TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTING TEST SET")
print("=" * 60)

y_proba = np.zeros(
    (
        X_test.shape[0],
        num_classes
    )
)

for i in tqdm(
    range(
        0,
        X_test.shape[0],
        64
    ),
    desc="Predicting Test",
    ncols=100
):

    x_batch = X_test[
        i:i + 64
    ]

    y_proba[
        i:i + 64
    ] = model.predict(
        x_batch,
        verbose=0
    )

y_pred_mlp = np.argmax(
    y_proba,
    axis=1
)

# ============================================================
# METRICS
# ============================================================

y_test_binarized = label_binarize(
    y_test,
    classes=np.arange(num_classes)
)

mlp_accuracy = accuracy_score(
    y_test,
    y_pred_mlp
)

mlp_precision = precision_score(
    y_test,
    y_pred_mlp,
    average="weighted"
)

mlp_recall = recall_score(
    y_test,
    y_pred_mlp,
    average="weighted"
)

mlp_f1 = f1_score(
    y_test,
    y_pred_mlp,
    average="weighted"
)

mlp_roc_auc = roc_auc_score(
    y_test_binarized,
    y_proba,
    multi_class="ovr",
    average="macro"
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MLP TEST RESULTS")
print("=" * 60)

print(
    f"MLP Accuracy   : {mlp_accuracy * 100:.2f}%"
)

print(
    f"MLP Precision  : {mlp_precision * 100:.2f}%"
)

print(
    f"MLP Recall     : {mlp_recall * 100:.2f}%"
)

print(
    f"MLP F1-Score   : {mlp_f1 * 100:.2f}%"
)

print(
    f"MLP ROC-AUC    : {mlp_roc_auc:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred_mlp,
        target_names=categories
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:\n"
)

print(
    confusion_matrix(
        y_test,
        y_pred_mlp
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

mlp_model_path = os.path.join(
    results_path,
    "mlp_model.keras"
)

model.save(
    mlp_model_path
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        results_path,
        "y_pred_mlp.npy"
    ),
    y_pred_mlp
)

np.save(
    os.path.join(
        results_path,
        "y_proba_mlp.npy"
    ),
    y_proba
)

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.save(
    os.path.join(
        results_path,
        "mlp_training_history.npy"
    ),
    history.history,
    allow_pickle=True
)

print("\n" + "=" * 60)
print("MLP COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"Results saved to:\n{results_path}"
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import joblib
