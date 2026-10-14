# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

# Standalone evaluation script: trains this classifier independently on the
# saved CNN features for individual-model comparison. The stacking workflow
# (src/10_stacking.py) intentionally refits its own fold-specific base models
# to generate out-of-fold meta-features; it does not load this model artifact.

from config import *

# =======================
# المسار المحلي
# =======================
base_path = str(PROJECT_ROOT)

features_path = str(FEATURE_DIR)

results_path = str(SVM_RESULT_DIR)

os.makedirs(
    results_path,
    exist_ok=True
)


# =======================
# تحميل المميزات والتسميات
# =======================
X_train_features = np.load(
    os.path.join(
        features_path,
        "X_train_features.npy"
    )
)

X_val_features = np.load(
    os.path.join(
        features_path,
        "X_val_features.npy"
    )
)

X_test_features = np.load(
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


# =======================
# Data information
# =======================
print("\n=======================")
print("DATA SHAPES")
print("=======================")

print(
    f"Training   : {X_train_features.shape}"
)

print(
    f"Validation : {X_val_features.shape}"
)

print(
    f"Test       : {X_test_features.shape}"
)


# =======================
# Classes
# =======================
categories = [
    "Normal",
    "Reversal",
    "Corrected"
]


# =======================
# Training SVM
# =======================
print("\n=======================")
print("TRAINING SVM")
print("=======================")

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True
)

svm_model.fit(
    X_train_features,
    y_train
)

print(
    "SVM model trained successfully."
)


# =======================
# Validation
# =======================
print("\n=======================")
print("VALIDATION")
print("=======================")

y_val_pred = svm_model.predict(
    X_val_features
)

svm_val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

print(
    f"SVM Validation Accuracy: "
    f"{svm_val_accuracy * 100:.2f}%"
)


# =======================
# Test prediction
# =======================
print("\n=======================")
print("TEST PREDICTION")
print("=======================")

y_pred_svm = svm_model.predict(
    X_test_features
)

y_proba_svm = svm_model.predict_proba(
    X_test_features
)


# =======================
# Metrics
# =======================
num_classes = len(
    np.unique(y_train)
)

y_test_binarized = label_binarize(
    y_test,
    classes=np.arange(num_classes)
)


svm_accuracy = accuracy_score(
    y_test,
    y_pred_svm
)

svm_precision = precision_score(
    y_test,
    y_pred_svm,
    average="weighted"
)

svm_recall = recall_score(
    y_test,
    y_pred_svm,
    average="weighted"
)

svm_f1 = f1_score(
    y_test,
    y_pred_svm,
    average="weighted"
)

svm_roc_auc = roc_auc_score(
    y_test_binarized,
    y_proba_svm,
    multi_class="ovr",
    average="macro"
)


# =======================
# Final results
# =======================
print("\n===================================")
print("FINAL SVM TEST RESULTS")
print("===================================")

print(
    f"SVM Accuracy   : "
    f"{svm_accuracy * 100:.2f}%"
)

print(
    f"SVM Precision  : "
    f"{svm_precision * 100:.2f}%"
)

print(
    f"SVM Recall     : "
    f"{svm_recall * 100:.2f}%"
)

print(
    f"SVM F1-Score   : "
    f"{svm_f1 * 100:.2f}%"
)

print(
    f"SVM ROC-AUC    : "
    f"{svm_roc_auc:.4f}"
)


# =======================
# Classification Report
# =======================
print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred_svm,
        target_names=categories
    )
)


# =======================
# Confusion Matrix
# =======================
print(
    "\nConfusion Matrix:\n"
)

print(
    confusion_matrix(
        y_test,
        y_pred_svm
    )
)


# =======================
# Save model
# =======================
svm_model_path = os.path.join(
    results_path,
    "svm_model_revised.pkl"
)

joblib.dump(
    svm_model,
    svm_model_path
)


# =======================
# Save predictions
# =======================
np.save(
    os.path.join(
        results_path,
        "y_pred_svm_revised.npy"
    ),
    y_pred_svm
)

np.save(
    os.path.join(
        results_path,
        "y_proba_svm_revised.npy"
    ),
    y_proba_svm
)


# =======================
# DONE
# =======================
print("\n===================================")
print("SVM COMPLETED SUCCESSFULLY")
print("===================================")

print(
    f"Results saved to:\n"
    f"{results_path}"
)

# Record exact ordered held-out samples used for these saved predictions.
write_test_sample_manifest(results_path, y_test)
