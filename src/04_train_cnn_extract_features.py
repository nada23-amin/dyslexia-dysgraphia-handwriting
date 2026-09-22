# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *

# ================================================================
# DYSLEXIA CNN TRAINING + 128-D FEATURE EXTRACTION
# MATCHED TO THE MANUSCRIPT METHODOLOGY
# ================================================================

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# 2. PATHS
# ================================================================

BASE_DIR = PROJECT_ROOT
TRAIN_DIR = str(BALANCED_DIR)
VAL_DIR = str(SPLIT_DIR / "val")
TEST_DIR = str(SPLIT_DIR / "test")
RESULTS_DIR = str(CNN_RESULT_DIR)
FEATURES_DIR = str(FEATURE_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
MODEL_PATH = os.path.join(RESULTS_DIR, "dyslexia_cnn.keras")


# ================================================================
# 3. PARAMETERS
# ================================================================

IMG_SIZE = (28, 28)

BATCH_SIZE = 64

EPOCHS = 20

CLASS_NAMES = [
    "Normal",
    "Reversal",
    "Corrected"
]

NUM_CLASSES = 3


# ================================================================
# 4. LOAD DATA
# ================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# 5. NORMALIZATION
# ================================================================

normalization_layer = layers.Rescaling(
    1.0 / 255.0
)

train_ds = train_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ================================================================
# 6. PREFETCH
# ================================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ================================================================
# 7. BUILD LIGHTWEIGHT CNN
# ================================================================

print("\n" + "=" * 70)
print("BUILDING CNN MODEL")
print("=" * 70)

model = models.Sequential(
    name="Dyslexia_CNN"
)

# Input
model.add(
    layers.Input(
        shape=(28, 28, 1)
    )
)

# Convolutional Block 1
model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Convolutional Block 2
model.add(
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Flatten
model.add(
    layers.Flatten()
)

# 128-D feature embedding
model.add(
    layers.Dense(
        128,
        activation="relu",
        name="feature_dense"
    )
)

# Dropout
model.add(
    layers.Dropout(
        0.3
    )
)

# CNN training output layer
model.add(
    layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="output_layer"
    )
)


# ================================================================
# 8. MODEL SUMMARY
# ================================================================

print()

model.summary()


# ================================================================
# 9. COMPILE CNN
# ================================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ================================================================
# 10. EARLY STOPPING
# ================================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ================================================================
# 11. LEARNING RATE REDUCTION
# ================================================================

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# ================================================================
# 12. TRAIN CNN
# ================================================================

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ================================================================
# 13. SAVE TRAINED CNN
# ================================================================

print("\n" + "=" * 70)
print("SAVING CNN MODEL")
print("=" * 70)

model.save(
    MODEL_PATH
)

print("✓ Model saved successfully:")
print(MODEL_PATH)


# ================================================================
# 14. FINAL VALIDATION EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION EVALUATION")
print("=" * 70)

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print(
    f"\nValidation Loss: {val_loss:.6f}"
)

print(
    f"Validation Accuracy: {val_accuracy:.6f}"
)

print(
    f"Validation Accuracy: {val_accuracy * 100:.2f}%"
)


# ================================================================
# 15. FINAL TEST EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Loss: {test_loss:.6f}"
)

print(
    f"Test Accuracy: {test_accuracy:.6f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ================================================================
# 16. CREATE 128-D FEATURE EXTRACTOR
# ================================================================

print("\n" + "=" * 70)
print("CREATING 128-D FEATURE EXTRACTOR")
print("=" * 70)

feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer(
        "feature_dense"
    ).output
)

print("✓ Feature extractor created")
print("Feature dimension: 128")


# ================================================================
# 17. FEATURE EXTRACTION FUNCTION
# ================================================================

def extract_features(dataset):

    features = []
    labels = []

    for images, batch_labels in dataset:

        batch_features = feature_extractor(
            images,
            training=False
        )

        features.append(
            batch_features.numpy()
        )

        labels.append(
            batch_labels.numpy()
        )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return features, labels


# ================================================================
# 18. EXTRACT TRAIN FEATURES
# ================================================================

print("\n" + "=" * 70)
print("EXTRACTING TRAIN FEATURES")
print("=" * 70)

X_train_features, y_train = extract_features(
    train_ds
)

print(
    "X_train_features:",
    X_train_features.shape
)


# ================================================================
# 19. EXTRACT VALIDATION FEATURES
# ================================================================

print("\n" + "=" * 70)
print("EXTRACTING VALIDATION FEATURES")
print("=" * 70)

X_val_features, y_val = extract_features(
    val_ds
)

print(
    "X_val_features:",
    X_val_features.shape
)


# ================================================================
# 20. EXTRACT TEST FEATURES
# ================================================================

print("\n" + "=" * 70)
print("EXTRACTING TEST FEATURES")
print("=" * 70)

X_test_features, y_test = extract_features(
    test_ds
)

print(
    "X_test_features:",
    X_test_features.shape
)


# ================================================================
# 21. CONVERT ONE-HOT LABELS TO CLASS INDICES
# ================================================================

y_train = np.argmax(
    y_train,
    axis=1
)

y_val = np.argmax(
    y_val,
    axis=1
)

y_test = np.argmax(
    y_test,
    axis=1
)


# ================================================================
# 22. PRINT FINAL FEATURE SHAPES
# ================================================================

print("\n" + "=" * 70)
print("FINAL FEATURE SHAPES")
print("=" * 70)

print(
    "X_train_features :",
    X_train_features.shape
)

print(
    "y_train           :",
    y_train.shape
)

print(
    "X_val_features   :",
    X_val_features.shape
)

print(
    "y_val             :",
    y_val.shape
)

print(
    "X_test_features  :",
    X_test_features.shape
)

print(
    "y_test            :",
    y_test.shape
)


# ================================================================
# 23. SAVE 128-D FEATURES
# ================================================================

print("\n" + "=" * 70)
print("SAVING FEATURES")
print("=" * 70)

np.save(
    os.path.join(
        FEATURES_DIR,
        "X_train_features.npy"
    ),
    X_train_features
)

np.save(
    os.path.join(
        FEATURES_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        FEATURES_DIR,
        "X_val_features.npy"
    ),
    X_val_features
)

np.save(
    os.path.join(
        FEATURES_DIR,
        "y_val.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        FEATURES_DIR,
        "X_test_features.npy"
    ),
    X_test_features
)

np.save(
    os.path.join(
        FEATURES_DIR,
        "y_test.npy"
    ),
    y_test
)

print("✓ All 128-D feature files saved.")


# ================================================================
# 24. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("CNN FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print(
    "\nFinal Validation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)

print(
    "Final Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\nFeature Shapes:")

print(
    "Train:",
    X_train_features.shape
)

print(
    "Validation:",
    X_val_features.shape
)

print(
    "Test:",
    X_test_features.shape
)

print("\nModel:")
print(MODEL_PATH)

print("\nResults:")
print(RESULTS_DIR)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
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
from tqdm import tqdm
import os
import random
