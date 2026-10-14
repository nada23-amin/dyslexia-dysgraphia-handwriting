# ==============================================================================
# DYSGRAPHIA - PROPOSED HYBRID FRAMEWORK
# ==============================================================================
# Original Dysgraphia dataset:
#   Low Potential Dysgraphia
#   Potential Dysgraphia
#
# Pipeline:
#   Original data
#       -> SHA-256 duplicate removal
#       -> 70/15/15 image-level split
#       -> training-set resampling/augmentation ONLY on training set
#       -> preprocessing
#       -> 28x28 CNN
#       -> 128-D CNN feature extraction
#       -> SVM / MLP / RF / XGBoost / LightGBM
#       -> 5-fold OOF stacking
#       -> Logistic Regression meta-learner
#
# IMPORTANT:
# Validation and independent test sets are never resampled or augmented.
# ==============================================================================

import os
import random
import hashlib
import warnings
import json
import joblib
from pathlib import Path

import numpy as np
import cv2
# The manuscript experiments were conducted CPU-only. Set before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.contingency_tables import mcnemar

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. REPRODUCIBILITY
# ==============================================================================

SEED = 42
ABLATION_MODE = os.environ.get("ABLATION_MODE", "standard").strip().lower()
OUTPUT_ROOT = (Path(__file__).resolve().parents[1] / "outputs"
               if ABLATION_MODE == "standard"
               else Path(__file__).resolve().parents[1] / "outputs" / "ablation" / ABLATION_MODE)

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("DYSGRAPHIA - PROPOSED HYBRID FRAMEWORK")
print("=" * 80)
print("TensorFlow:", tf.__version__)
print("Seed:", SEED)

# ==============================================================================
# 2. GOOGLE DRIVE
# ==============================================================================


# ==============================================================================
# 3. ORIGINAL DATASET
# ==============================================================================

DATASET_DIR = os.environ.get(
    "DYSGRAPHIA_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dysgraphia_raw")
)

CLASS_NAMES = [
    "Low Potential Dysgraphia",
    "Potential Dysgraphia"
]

print("\nOriginal dataset:")
print(DATASET_DIR)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory was not found:\n{DATASET_DIR}"
    )

for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_dir):
        raise FileNotFoundError(
            f"Class directory was not found:\n{class_dir}"
        )

    print(f"  {class_name}: {class_dir}")

# ==============================================================================
# 4. COLLECT IMAGES
# ==============================================================================

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def collect_images(class_dir):
    paths = []

    for root, _, files in os.walk(class_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, file))

    return sorted(paths)


all_paths = []
all_labels = []

for label, class_name in enumerate(CLASS_NAMES):
    paths = collect_images(
        os.path.join(DATASET_DIR, class_name)
    )

    print(f"\n{class_name}: {len(paths)} images")

    all_paths.extend(paths)
    all_labels.extend([label] * len(paths))

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

print("\nTotal original images:", len(all_paths))

# ==============================================================================
# 5. GLOBAL SHA-256 DUPLICATE REMOVAL
# ==============================================================================

def compute_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


print("\nRemoving duplicate images using SHA-256...")

seen_hashes = {}
duplicate_audit_rows = []

unique_paths = []
unique_labels = []

duplicates = 0

for path, label in zip(all_paths, all_labels):
    image_hash = compute_image_hash(path)

    if image_hash in seen_hashes:
        duplicates += 1
        canonical_path, canonical_label = seen_hashes[image_hash]
        duplicate_audit_rows.append({
            "duplicate_path": str(path),
            "canonical_path": str(canonical_path),
            "duplicate_class": CLASS_NAMES[int(label)],
            "canonical_class": CLASS_NAMES[int(canonical_label)],
            "sha256": image_hash,
            "cross_class_duplicate": bool(label != canonical_label),
        })
        continue

    seen_hashes[image_hash] = (path, label)
    unique_paths.append(path)
    unique_labels.append(label)

unique_paths = np.array(unique_paths)
unique_labels = np.array(unique_labels)

print("Duplicates removed:", duplicates)
print("Unique images:", len(unique_paths))

# Persist the exact duplicate audit rather than silently discarding duplicate paths.
import csv
_audit_dir = OUTPUT_ROOT / "audit" / "dysgraphia"
_audit_dir.mkdir(parents=True, exist_ok=True)
with open(_audit_dir / "duplicate_audit.csv", "w", newline="", encoding="utf-8") as _f:
    _fields = ["duplicate_path", "canonical_path", "duplicate_class", "canonical_class", "sha256", "cross_class_duplicate"]
    _writer = csv.DictWriter(_f, fieldnames=_fields)
    _writer.writeheader()
    _writer.writerows(duplicate_audit_rows)
if any(row["cross_class_duplicate"] for row in duplicate_audit_rows):
    raise ValueError("Exact duplicate files were found across different class labels. Resolve label conflicts before training; see outputs/audit/dysgraphia/duplicate_audit.csv.")

for label, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name}: "
        f"{np.sum(unique_labels == label)}"
    )

# ==============================================================================
# 6. 70/15/15 IMAGE-LEVEL SPLIT
# ==============================================================================

print("\nCreating 70/15/15 train/validation/test split...")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    unique_paths,
    unique_labels,
    test_size=0.30,
    random_state=SEED,
    shuffle=True,
    stratify=unique_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
    stratify=temp_labels
)

# IMPORTANT: labels used by CNN/evaluation
y_train = train_labels.astype(np.int32)
y_val = val_labels.astype(np.int32)
y_test = test_labels.astype(np.int32)

print("\nSplit sizes:")
print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))

for label, class_name in enumerate(CLASS_NAMES):
    print(f"\n{class_name}:")
    print("  Train:", np.sum(train_labels == label))
    print("  Validation:", np.sum(val_labels == label))
    print("  Test:", np.sum(test_labels == label))

# ==============================================================================
# 7. PREPROCESSING
# ==============================================================================

def deskew_image(image):

    try:

        edges = cv2.Canny(
            image,
            50,
            200
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is not None:

            angles = [
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                ) * 180 / np.pi

                for line in lines

                for x1, y1, x2, y2 in [line[0]]
            ]

            median_angle = np.median(
                angles
            )

            if abs(median_angle) > 0.1:

                h, w = image.shape[:2]

                center = (
                    w // 2,
                    h // 2
                )

                matrix = cv2.getRotationMatrix2D(
                    center,
                    median_angle,
                    1.0
                )

                return cv2.warpAffine(
                    image,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

    except Exception as e:

        print(
            "Deskew warning:",
            e
        )

    return image


def preprocess_array(image):

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Ablation variant: omit enhancement, denoising, thresholding, and deskewing.
    # Grayscale conversion and resizing remain common to all variants.
    if ABLATION_MODE == "no_preprocessing":
        return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    image = cv2.fastNlMeansDenoising(
        image,
        None,
        h=30
    )

    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        2
    )

    image = deskew_image(image)

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    return image


def read_and_preprocess(path):
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return preprocess_array(image)

# ==============================================================================
# 8. TRAINING-SET RESAMPLING / AUGMENTATION
# ==============================================================================
# Target: 2000 training samples per class.
# Validation and test remain completely untouched.
#
# Augmentation:
#   - rotation: -20 to +20 degrees
#   - Gaussian noise: sigma = 10
#
# The augmentation is applied only to the training set.

TARGET_TRAIN_PER_CLASS = 2000


def augment_training_image(image):

    # Same augmentation configuration used in the Dyslexia pipeline:
    # rotation -> Gaussian noise -> CLAHE -> adaptive threshold.
    # Horizontal flipping is NOT used.

    angle = random.uniform(-20, 20)

    h, w = image.shape[:2]

    center = (
        w // 2,
        h // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    noise = np.random.normal(
        loc=0,
        scale=10,
        size=rotated.shape
    ).astype(np.float32)

    noisy = (
        rotated.astype(np.float32)
        + noise
    )

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    if ABLATION_MODE == "no_preprocessing":
        return np.clip(noisy, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        noisy
    )

    augmented = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        2
    )

    return augmented


print("\nPreprocessing original training images...")

train_images_original = []
for i, path in enumerate(train_paths):
    train_images_original.append(
        read_and_preprocess(path)
    )

    if (i + 1) % 25 == 0 or i + 1 == len(train_paths):
        print(
            f"\rProcessed {i + 1}/{len(train_paths)}",
            end=""
        )

print()

train_images_original = np.array(
    train_images_original,
    dtype=np.uint8
)

# Validation and test are preprocessed only; no augmentation/resampling.
print("\nPreprocessing validation images...")

val_images = np.array(
    [read_and_preprocess(p) for p in val_paths],
    dtype=np.uint8
)

print("\nPreprocessing test images...")

test_images = np.array(
    [read_and_preprocess(p) for p in test_paths],
    dtype=np.uint8
)

# Build training set. In the no_resampling ablation, use only the original
# training images without class balancing or augmentation. Otherwise preserve the
# manuscript's target of 2,000 training images per class.
balanced_train_images = []
balanced_train_labels = []
balanced_train_source_ids = []
use_resampling = ABLATION_MODE != "no_resampling"

print("\\nBuilding training set...")
for label, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(train_labels == label)[0]
    class_images = train_images_original[class_indices]
    print(f"\\n{class_name}: original train samples = {len(class_images)}")

    if not use_resampling:
        selected = list(class_images)
        balanced_train_images.extend(selected)
        balanced_train_labels.extend([label] * len(selected))
        balanced_train_source_ids.extend(class_indices.tolist())
        print(f"{class_name}: no resampling; retained {len(selected)} original training images")
        continue

    selected = list(class_images)
    selected_source_ids = class_indices.tolist()
    while len(selected) < TARGET_TRAIN_PER_CLASS:
        source_idx = random.randrange(len(class_images))
        source = class_images[source_idx]
        selected.append(augment_training_image(source))
        selected_source_ids.append(int(class_indices[source_idx]))
    selected = selected[:TARGET_TRAIN_PER_CLASS]
    selected_source_ids = selected_source_ids[:TARGET_TRAIN_PER_CLASS]
    balanced_train_images.extend(selected)
    balanced_train_labels.extend([label] * len(selected))
    balanced_train_source_ids.extend(selected_source_ids)
    print(f"{class_name}: resampled train samples = {len(selected)}")

X_train_img = np.array(balanced_train_images, dtype=np.uint8)
y_train = np.array(balanced_train_labels, dtype=np.int32)

train_source_ids = np.asarray(balanced_train_source_ids, dtype=np.int64)
shuffle_indices = np.random.permutation(len(X_train_img))
X_train_img = X_train_img[shuffle_indices]
y_train = y_train[shuffle_indices]
train_source_ids = train_source_ids[shuffle_indices]

print("\\nFinal training shape:", X_train_img.shape)
print("Final training distribution:", {
    int(label): int(np.sum(y_train == label)) for label in np.unique(y_train)
})

# ==============================================================================
# 9. CNN INPUT: 28 x 28
# ==============================================================================

def prepare_cnn_input(images):

    resized = np.array([
        cv2.resize(
            image,
            (28, 28),
            interpolation=cv2.INTER_AREA
        )
        for image in images
    ])

    resized = (
        resized.astype("float32") / 255.0
    )

    resized = np.expand_dims(
        resized,
        axis=-1
    )

    return resized


X_train_cnn = prepare_cnn_input(
    X_train_img
)

X_val_cnn = prepare_cnn_input(
    val_images
)

X_test_cnn = prepare_cnn_input(
    test_images
)

# Persist exact split tensors/labels for independent baseline training and post-hoc visualizations.
_SPLIT_ARTIFACT_DIR = OUTPUT_ROOT / "splits" / "dysgraphia"
_SPLIT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
for _name, _array in {
    "X_train_cnn.npy": X_train_cnn, "y_train.npy": y_train,
    "train_source_group_ids.npy": train_source_ids,
    "X_val_cnn.npy": X_val_cnn, "y_val.npy": y_val,
    "X_test_cnn.npy": X_test_cnn, "y_test.npy": y_test,
    "X_test_preprocessed_128.npy": test_images,
}.items():
    np.save(_SPLIT_ARTIFACT_DIR / _name, _array)
# Keep the exact held-out image order alongside y_test for auditing paired predictions.
with open(_SPLIT_ARTIFACT_DIR / "test_sample_paths.json", "w", encoding="utf-8") as _f:
    json.dump([str(Path(p).resolve()) for p in test_paths], _f, indent=2)
_split_manifest = {
    "task": "dysgraphia", "split_protocol": "image-level 70/15/15", "seed": SEED,
    "class_names": CLASS_NAMES, "n_unique_original_images": int(len(unique_paths)),
    "n_original_train_before_augmentation": int(len(train_paths)),
    "n_train_after_augmentation": int(len(y_train)),
    "n_validation": int(len(y_val)), "n_test": int(len(y_test)),
    "original_class_counts": {CLASS_NAMES[i]: int(np.sum(unique_labels == i)) for i in range(len(CLASS_NAMES))},
    "manuscript_expected_counts": {"original_total": 249, "original_train": 174, "validation": 37, "test": 38, "train_after_augmentation": 4000},
    "n_test_by_class": {CLASS_NAMES[i]: int(np.sum(y_test == i)) for i in range(len(CLASS_NAMES))},
    "participant_grouping": "not possible; participant IDs unavailable",
    "augmentation_applied_to": "training subset only",
}
_split_manifest["matches_manuscript_expected_counts"] = all([
    _split_manifest["n_unique_original_images"] == 249,
    _split_manifest["n_original_train_before_augmentation"] == 174,
    _split_manifest["n_validation"] == 37,
    _split_manifest["n_test"] == 38,
    _split_manifest["n_train_after_augmentation"] == 4000,
    _split_manifest["original_class_counts"] == {"Low Potential Dysgraphia": 135, "Potential Dysgraphia": 114},
])
with open(_SPLIT_ARTIFACT_DIR / "split_manifest.json", "w", encoding="utf-8") as _f:
    json.dump(_split_manifest, _f, indent=2)
if not _split_manifest["matches_manuscript_expected_counts"]:
    print("WARNING: dysgraphia source/split counts differ from manuscript-reported counts. Verify dataset version and duplicate audit before submission.")

print("\nCNN data shapes:")
print("X_train:", X_train_cnn.shape)
print("X_val  :", X_val_cnn.shape)
print("X_test :", X_test_cnn.shape)

# ==============================================================================
# 10. CNN FEATURE EXTRACTOR
# ==============================================================================

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.keras.backend.clear_session()

cnn = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu",
        name="feature_dense"
    ),

    Dropout(0.3),

    Dense(
        2,
        activation="softmax"
    )
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model:")
cnn.summary()

# ==============================================================================
# 11. CNN TRAINING
# ==============================================================================

# The manuscript specifies 20 epochs with validation monitoring. The learning-rate
# scheduler monitors validation loss but does not stop training early.
print("\nTraining CNN for 20 epochs with validation monitoring...")
history = cnn.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)],
    verbose=1
)


# Save the CNN whose 128-D penultimate-layer features are used downstream.
_MODEL_DIR = OUTPUT_ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
cnn.save(_MODEL_DIR / "dysgraphia_feature_extractor_cnn.keras")

# ==============================================================================
# 12. CNN EVALUATION
# ==============================================================================

def evaluate_binary(
    y_true,
    y_pred,
    probabilities
):

    result = {}

    result["Accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    result["Weighted Precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    result["Weighted Recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    result["Weighted F1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # The manuscript's stacking table reports macro-averaged precision, recall, and F1.
    # Keep weighted metrics too for the base-classifier comparison table.
    result["Macro Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    result["Macro Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    result["Macro F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    result["ROC-AUC"] = roc_auc_score(
        y_true,
        probabilities[:, 1]
    )

    result["LogLoss"] = log_loss(
        y_true,
        probabilities
    )

    return result


val_cnn_prob = cnn.predict(
    X_val_cnn,
    verbose=0
)

test_cnn_prob = cnn.predict(
    X_test_cnn,
    verbose=0
)

val_cnn_pred = np.argmax(
    val_cnn_prob,
    axis=1
)

test_cnn_pred = np.argmax(
    test_cnn_prob,
    axis=1
)

val_cnn_metrics = evaluate_binary(
    y_val,
    val_cnn_pred,
    val_cnn_prob
)

test_cnn_metrics = evaluate_binary(
    y_test,
    test_cnn_pred,
    test_cnn_prob
)

print("\n" + "=" * 80)
print("END-TO-END CNN RESULTS")
print("=" * 80)

print("\nValidation:")
for key, value in val_cnn_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\nIndependent Test:")
for key, value in test_cnn_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\nTest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_cnn_pred
    )
)

# ==============================================================================
# 13. 128-D CNN FEATURE EXTRACTION
# ==============================================================================

print("\nExtracting 128-dimensional CNN features...")

feature_extractor = tf.keras.Model(
    inputs=cnn.inputs,
    outputs=cnn.get_layer(
        "feature_dense"
    ).output
)

X_train_features = feature_extractor.predict(
    X_train_cnn,
    verbose=0
)

X_val_features = feature_extractor.predict(
    X_val_cnn,
    verbose=0
)

X_test_features = feature_extractor.predict(
    X_test_cnn,
    verbose=0
)

print("\nCNN feature shapes:")
print("X_train_features:", X_train_features.shape)
print("X_val_features  :", X_val_features.shape)
print("X_test_features :", X_test_features.shape)

# ==============================================================================
# 14. BASE CLASSIFIERS
# ==============================================================================

def build_mlp(input_dim):

    model = Sequential([

        Dense(
            512,
            activation="relu",
            input_shape=(input_dim,)
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
            2,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def make_model(name):

    if name == "SVM":

        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True
        )

    if name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )

    if name == "XGBoost":

        return XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            objective="multi:softprob",
            num_class=2
        )

    if name == "LightGBM":

        return LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            objective="multiclass",
            num_class=2,
            verbose=-1
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = [
    "MLP",
    "SVM",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]

# ==============================================================================
# 15. STANDALONE KERAS MLP
# ==============================================================================

print("\n" + "=" * 80)
print("STANDALONE MLP")
print("=" * 80)

standalone_mlp = build_mlp(
    X_train_features.shape[1]
)

standalone_mlp_history = standalone_mlp.fit(
    X_train_features,
    y_train,
    validation_data=(
        X_val_features,
        y_val
    ),
    epochs=50,
    batch_size=64,
    shuffle=True,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1
)

standalone_mlp_test_prob = standalone_mlp.predict(
    X_test_features,
    verbose=0
)

standalone_mlp_test_pred = np.argmax(
    standalone_mlp_test_prob,
    axis=1
)

standalone_mlp_metrics = evaluate_binary(
    y_test,
    standalone_mlp_test_pred,
    standalone_mlp_test_prob
)

print("\nStandalone MLP test confusion matrix:")
print(
    confusion_matrix(
        y_test,
        standalone_mlp_test_pred
    )
)

# ==============================================================================\n# 16. STANDALONE CLASSICAL CLASSIFIERS\n# ==============================================================================\n\nprint("\\n" + "=" * 80)\nprint("STANDALONE CLASSICAL CLASSIFIERS")\nprint("=" * 80)\n\nstandalone_base_results = {}\n\n# ------------------------------------------------------------------------------\n# SVM\n# ------------------------------------------------------------------------------\n\nprint("\\nTraining standalone SVM...")\n\nstandalone_svm = SVC(\n    kernel="rbf",\n    C=1.0,\n    gamma="scale",\n    probability=True\n)\n\nstandalone_svm.fit(\n    X_train_features,\n    y_train\n)\n\nsvm_prob = standalone_svm.predict_proba(\n    X_test_features\n)\nsvm_pred = np.argmax(\n    svm_prob,\n    axis=1\n)\n\nstandalone_base_results["SVM"] = evaluate_binary(\n    y_test,\n    svm_pred,\n    svm_prob\n)\n\n# ------------------------------------------------------------------------------\n# RANDOM FOREST\n# ------------------------------------------------------------------------------\n\nprint("\\nTraining standalone Random Forest...")\n\nstandalone_rf = RandomForestClassifier(\n    n_estimators=100,\n    max_depth=None,\n    random_state=42\n)\n\nstandalone_rf.fit(\n    X_train_features,\n    y_train\n)\n\nrf_prob = standalone_rf.predict_proba(\n    X_test_features\n)\nrf_pred = np.argmax(\n    rf_prob,\n    axis=1\n)\n\nstandalone_base_results["Random Forest"] = evaluate_binary(\n    y_test,\n    rf_pred,\n    rf_prob\n)\n\n# ------------------------------------------------------------------------------\n# XGBOOST\n# ------------------------------------------------------------------------------\n\nprint("\\nTraining standalone XGBoost...")\n\nstandalone_xgb = make_model(\n    "XGBoost"\n)\n\nstandalone_xgb.fit(\n    X_train_features,\n    y_train\n)\n\nxgb_prob = standalone_xgb.predict_proba(\n    X_test_features\n)\nxgb_pred = np.argmax(\n    xgb_prob,\n    axis=1\n)\n\nstandalone_base_results["XGBoost"] = evaluate_binary(\n    y_test,\n    xgb_pred,\n    xgb_prob\n)\n\n# ------------------------------------------------------------------------------\n# LIGHTGBM\n# ------------------------------------------------------------------------------\n\nprint("\\nTraining standalone LightGBM...")\n\nstandalone_lgbm = make_model(\n    "LightGBM"\n)\n\nstandalone_lgbm.fit(\n    X_train_features,\n    y_train\n)\n\nlgbm_prob = standalone_lgbm.predict_proba(\n    X_test_features\n)\nlgbm_pred = np.argmax(\n    lgbm_prob,\n    axis=1\n)\n\nstandalone_base_results["LightGBM"] = evaluate_binary(\n    y_test,\n    lgbm_pred,\n    lgbm_prob\n)\n\nprint("\\nStandalone classifier summary:")\nfor name, result in standalone_base_results.items():\n    print(\n        f"{name:18s} "\n        f"Accuracy={result['Accuracy'] * 100:.2f}% | "\n        f"Weighted F1={result['Weighted F1'] * 100:.2f}% | "\n        f"ROC-AUC={result['ROC-AUC']:.4f}"\n    )\n\n# ==============================================================================\n# 17. 5-FOLD OOF STACKING\n

# ==============================================================================
# OOF predictions train the meta-learner.
# For validation/test, predictions from the five fitted folds are averaged.
# This prevents the validation/test features from depending on a single fold.

print("\n" + "=" * 80)
print("5-FOLD STACKING")
print("=" * 80)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# Manuscript protocol: retain both class probabilities from each binary base
# classifier (5 classifiers x 2 class probabilities = 10 meta-features).
N_CLASSES = 2
N_META_FEATURES = len(MODEL_NAMES) * N_CLASSES

oof_meta = np.zeros(
    (len(X_train_features), N_META_FEATURES),
    dtype=np.float32
)

val_fold_predictions = {
    name: []
    for name in MODEL_NAMES
}

test_fold_predictions = {
    name: []
    for name in MODEL_NAMES
}

for fold, (train_idx, valid_idx) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X_train_cnn, y_train
    ),
    start=1
):

    print(f"\nFold {fold}/5")

    # Manuscript-aligned protocol: CNN embeddings are extracted once by the
    # feature-extraction stage and treated as fixed features for downstream
    # classifiers. The five-fold split applies to base-classifier fitting.
    X_fold_train = X_train_features[train_idx]
    X_fold_valid = X_train_features[valid_idx]
    X_val_fold = X_val_features
    X_test_fold = X_test_features

    for model_idx, name in enumerate(
        MODEL_NAMES
    ):

        print(f"  Training {name}...")

        if name == "MLP":

            model = build_mlp(
                X_fold_train.shape[1]
            )

            # Match the manuscript's stacking protocol: fixed 50 epochs,
            # batch size 64, and no EarlyStopping within the OOF fold.
            # The held-out fold is used only to generate OOF predictions.
            model.fit(
                X_fold_train,
                y_train[train_idx],
                epochs=50,
                batch_size=64,
                verbose=0
            )

            oof_prob = model.predict(
                X_fold_valid, verbose=0
            )
            val_prob = model.predict(X_val_fold, verbose=0)
            test_prob = model.predict(X_test_fold, verbose=0)

        else:
            model = make_model(name)
            model.fit(X_fold_train, y_train[train_idx])
            oof_prob = model.predict_proba(X_fold_valid)
            val_prob = model.predict_proba(X_val_fold)
            test_prob = model.predict_proba(X_test_fold)

        # Store the complete two-class probability vector for this model.
        start = model_idx * N_CLASSES
        stop = start + N_CLASSES
        oof_meta[valid_idx, start:stop] = oof_prob
        val_fold_predictions[name].append(val_prob)
        test_fold_predictions[name].append(test_prob)

# Average each class probability over the five fold-specific models.
val_meta = np.column_stack([
    np.mean(val_fold_predictions[name], axis=0)
    for name in MODEL_NAMES
])
test_meta = np.column_stack([
    np.mean(test_fold_predictions[name], axis=0)
    for name in MODEL_NAMES
])

print("\nOOF meta-feature shape:")
print(oof_meta.shape)

print("Validation meta-feature shape:")
print(val_meta.shape)

print("Test meta-feature shape:")
print(test_meta.shape)

# ==============================================================================
# 16. LOGISTIC REGRESSION META-LEARNER
# ==============================================================================

meta_learner = LogisticRegression(
    C=0.1,
    random_state=SEED,
    max_iter=1000
)

meta_learner.fit(
    oof_meta,
    y_train
)

# ==============================================================================
# 17. FINAL STACKING RESULTS
# ==============================================================================

val_stack_prob = meta_learner.predict_proba(
    val_meta
)[:, 1]

test_stack_prob = meta_learner.predict_proba(
    test_meta
)[:, 1]

val_stack_pred = (
    val_stack_prob >= 0.5
).astype(int)

test_stack_pred = (
    test_stack_prob >= 0.5
).astype(int)

stack_val_metrics = evaluate_binary(
    y_val,
    val_stack_pred,
    np.column_stack([
        1 - val_stack_prob,
        val_stack_prob
    ])
)

stack_test_metrics = evaluate_binary(
    y_test,
    test_stack_pred,
    np.column_stack([
        1 - test_stack_prob,
        test_stack_prob
    ])
)

print("\n" + "=" * 80)
print("FINAL STACKING RESULTS")
print("=" * 80)

print("\nValidation:")
for key, value in stack_val_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\nIndependent Test:")
for key, value in stack_test_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\nTest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_stack_pred
    )
)

# ==============================================================================
# 18. BASE CLASSIFIER TEST RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("BASE CLASSIFIER RESULTS")
print("=" * 80)

base_test_results = {}

for model_idx, name in enumerate(MODEL_NAMES):

    start = model_idx * N_CLASSES
    probabilities = test_meta[:, start:start + N_CLASSES]
    predictions = np.argmax(probabilities, axis=1)

    result = evaluate_binary(y_test, predictions, probabilities)

    base_test_results[name] = result

    print(
        f"\n{name}:"
    )

    for key, value in result.items():
        print(
            f"{key:20s}: {value:.6f}"
        )

# ==============================================================================
# 19. SAVE TEST PREDICTIONS, PROBABILITIES, METRICS, AND PAIRED TESTS
# ==============================================================================

RESULT_DIR = OUTPUT_ROOT / "results" / "dysgraphia"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Save labels and meta-features so every reported metric can be recomputed later.
np.save(RESULT_DIR / "y_test_true.npy", y_test)
np.save(RESULT_DIR / "test_meta_features.npy", test_meta)
np.save(RESULT_DIR / "oof_meta_features.npy", oof_meta)
np.save(RESULT_DIR / "y_test_pred_stacking.npy", test_stack_pred)
np.save(RESULT_DIR / "y_test_proba_stacking.npy", np.column_stack([1-test_stack_prob, test_stack_prob]))
joblib.dump(meta_learner, RESULT_DIR / "stacking_meta_learner.joblib")

# Record the exact ordered held-out source images and labels used by every
# prediction written below. All models in this script use the same test arrays.
_ordered_test_paths = [str(Path(p).resolve()) for p in test_paths]
_ordered_paths_bytes = json.dumps(
    _ordered_test_paths, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")
_test_labels_bytes = np.asarray(y_test, dtype=np.int64).tobytes()
_dysgraphia_test_manifest = {
    "task": "dysgraphia",
    "n_test": int(len(y_test)),
    "ordered_test_paths": _ordered_test_paths,
    "ordered_paths_sha256": hashlib.sha256(_ordered_paths_bytes).hexdigest(),
    "test_labels_sha256": hashlib.sha256(_test_labels_bytes).hexdigest(),
    "label_source": "y_test from current dysgraphia_pipeline.py run",
    "prediction_source": "all model predictions generated in this pipeline run",
}
(RESULT_DIR / "test_sample_manifest.json").write_text(
    json.dumps(_dysgraphia_test_manifest, indent=2), encoding="utf-8"
)

all_test_predictions = {"Stacking": test_stack_pred}
all_metrics = {"Stacking": stack_test_metrics}
for model_idx, name in enumerate(MODEL_NAMES):
    start = model_idx * N_CLASSES
    prob = test_meta[:, start:start + N_CLASSES]
    pred = np.argmax(prob, axis=1)
    all_test_predictions[name] = pred
    np.save(RESULT_DIR / f"y_test_pred_{name.lower().replace(' ', '_')}.npy", pred)
    np.save(RESULT_DIR / f"y_test_proba_{name.lower().replace(' ', '_')}.npy", prob)
    all_metrics[name] = base_test_results[name]

cm = confusion_matrix(y_test, test_stack_pred, labels=[0, 1])
np.savetxt(RESULT_DIR / "confusion_matrix_stacking.csv", cm, delimiter=",", fmt="%d")
with open(RESULT_DIR / "classification_report_stacking.json", "w", encoding="utf-8") as f:
    json.dump(classification_report(y_test, test_stack_pred, labels=[0, 1],
                                   target_names=CLASS_NAMES, output_dict=True,
                                   zero_division=0), f, indent=2)
with open(RESULT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({k: {m: float(v) for m, v in val.items()} for k, val in all_metrics.items()}, f, indent=2)

# Exact two-sided McNemar tests on paired predictions from the identical test set.
# Fail closed if the current-run manifest or saved predictions are inconsistent.
_manifest_path = RESULT_DIR / "test_sample_manifest.json"
if not _manifest_path.exists():
    raise FileNotFoundError("Missing current-run test_sample_manifest.json; cannot verify paired McNemar inputs.")
_manifest_check = json.loads(_manifest_path.read_text(encoding="utf-8"))
_expected_label_hash = hashlib.sha256(np.asarray(y_test, dtype=np.int64).tobytes()).hexdigest()
if (_manifest_check.get("task") != "dysgraphia"
        or _manifest_check.get("n_test") != len(y_test)
        or _manifest_check.get("test_labels_sha256") != _expected_label_hash
        or len(_manifest_check.get("ordered_test_paths", [])) != len(y_test)):
    raise ValueError("Dysgraphia test manifest does not match this run; refusing McNemar.")
for _model_name, _model_pred in all_test_predictions.items():
    if len(_model_pred) != len(y_test):
        raise ValueError(f"McNemar refused: {_model_name} predictions do not match test labels length.")
    if np.any((_model_pred < 0) | (_model_pred > 1)):
        raise ValueError(f"McNemar refused: invalid class encoding in {_model_name} predictions.")
mcnemar_results = {}
for name, pred in all_test_predictions.items():
    if name == "Stacking":
        continue
    stack_correct = (test_stack_pred == y_test)
    model_correct = (pred == y_test)
    table = np.array([
        [np.sum(stack_correct & model_correct), np.sum(stack_correct & ~model_correct)],
        [np.sum(~stack_correct & model_correct), np.sum(~stack_correct & ~model_correct)]
    ], dtype=int)
    test_result = mcnemar(table, exact=True)
    mcnemar_results[f"Stacking_vs_{name}"] = {
        "contingency_table": table.tolist(),
        "statistic": float(test_result.statistic),
        "p_value": float(test_result.pvalue),
        "n_test": int(len(y_test))
    }
with open(RESULT_DIR / "mcnemar_exact_paired.json", "w", encoding="utf-8") as f:
    json.dump(mcnemar_results, f, indent=2)

print(f"\nSaved final Dysgraphia test predictions and analyses to: {RESULT_DIR}")

# ==============================================================================
# 20. FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("FINAL PROPOSED FRAMEWORK SUMMARY")
print("=" * 80)

print("\nEnd-to-End CNN:")
for key, value in test_cnn_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\nBase Classifiers:")
for name, result in base_test_results.items():
    print(
        f"{name:20s}: "
        f"Accuracy={result['Accuracy']:.6f} | "
        f"Macro-F1={result['Macro F1']:.6f} | "
        f"ROC-AUC={result['ROC-AUC']:.6f}"
    )

print("\nStacking:")
for key, value in stack_test_metrics.items():
    print(f"{key:20s}: {value:.6f}")

print("\n" + "=" * 80)
print("PROPOSED FRAMEWORK COMPLETED")
print("=" * 80)
