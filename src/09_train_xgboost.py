# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

# Standalone evaluation script: trains this classifier independently on the
# saved CNN features for individual-model comparison. The stacking workflow
# (src/10_stacking.py) intentionally refits its own fold-specific base models
# to generate out-of-fold meta-features; it does not load this model artifact.

from config import *

# =======================
# المسارات
# =======================
base_path = str(PROJECT_ROOT)

features_path = str(FEATURE_DIR)

results_path = str(XGB_RESULT_DIR)

os.makedirs(
    results_path,
    exist_ok=True
)


# =======================
# تحميل البيانات
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
# Data Shapes
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

num_classes = len(
    np.unique(y_train)
)


# =======================
# تدريب XGBoost
# Settings as specified in the paper
# =======================
print("\n=======================")
print("TRAINING XGBOOST")
print("=======================")

xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    objective="multi:softprob"
)

xgb_model.fit(
    X_train_features,
    y_train
)

print(
    "XGBoost model "
    "trained successfully."
)


# =======================
# Validation
# =======================
print("\n=======================")
print("VALIDATION")
print("=======================")

y_val_pred = xgb_model.predict(
    X_val_features
)

y_val_proba = xgb_model.predict_proba(
    X_val_features
)

xgb_val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

print(
    f"XGBoost Validation Accuracy: "
    f"{xgb_val_accuracy * 100:.2f}%"
)


# =======================
# Test prediction
# =======================
print("\n=======================")
print("TEST PREDICTION")
print("=======================")

y_pred_xgb = xgb_model.predict(
    X_test_features
)

y_pred_xgb_proba = (
    xgb_model.predict_proba(
        X_test_features
    )
)


# =======================
# Metrics
# =======================
y_test_binarized = label_binarize(
    y_test,
    classes=np.arange(num_classes)
)


xgb_accuracy = accuracy_score(
    y_test,
    y_pred_xgb
)

xgb_precision = precision_score(
    y_test,
    y_pred_xgb,
    average="weighted"
)

xgb_recall = recall_score(
    y_test,
    y_pred_xgb,
    average="weighted"
)

xgb_f1 = f1_score(
    y_test,
    y_pred_xgb,
    average="weighted"
)

# Macro ROC-AUC, one-vs-rest
xgb_roc_auc = roc_auc_score(
    y_test_binarized,
    y_pred_xgb_proba,
    multi_class="ovr",
    average="macro"
)

xgb_log_loss = log_loss(
    y_test_binarized,
    y_pred_xgb_proba
)


# =======================
# Final Results
# =======================
print("\n===================================")
print("FINAL XGBOOST TEST RESULTS")
print("===================================")

print(
    f"Accuracy        : "
    f"{xgb_accuracy * 100:.2f}%"
)

print(
    f"Precision (W)   : "
    f"{xgb_precision:.4f}"
)

print(
    f"Recall (W)      : "
    f"{xgb_recall:.4f}"
)

print(
    f"F1-Score (W)    : "
    f"{xgb_f1:.4f}"
)

print(
    f"ROC-AUC (Macro) : "
    f"{xgb_roc_auc:.4f}"
)

print(
    f"Log Loss        : "
    f"{xgb_log_loss:.4f}"
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
        y_pred_xgb,
        target_names=categories
    )
)


# =======================
# Confusion Matrix
# =======================
xgb_conf_matrix = confusion_matrix(
    y_test,
    y_pred_xgb
)

print(
    "\nConfusion Matrix:\n"
)

print(
    xgb_conf_matrix
)


# =======================
# Plot Confusion Matrix
# =======================
plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    xgb_conf_matrix,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=categories,
    yticklabels=categories
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "Confusion Matrix - XGBoost Model"
)

plt.tight_layout()

plt.show()


# =======================
# Save model
# =======================
xgb_model_path = os.path.join(
    results_path,
    "xgb_model_revised.pkl"
)

joblib.dump(
    xgb_model,
    xgb_model_path
)


# =======================
# Save test predictions
# =======================
np.save(
    os.path.join(
        results_path,
        "y_pred_xgb_revised.npy"
    ),
    y_pred_xgb
)

np.save(
    os.path.join(
        results_path,
        "y_proba_xgb_revised.npy"
    ),
    y_pred_xgb_proba
)


# =======================
# Save validation predictions
# =======================
np.save(
    os.path.join(
        results_path,
        "y_val_pred_xgb_revised.npy"
    ),
    y_val_pred
)

np.save(
    os.path.join(
        results_path,
        "y_val_proba_xgb_revised.npy"
    ),
    y_val_proba
)


# =======================
# DONE
# =======================
print("\n===================================")
print("XGBOOST COMPLETED")
print("===================================")

print(
    f"Results saved to:\n"
    f"{results_path}"
)
import numpy as np
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# Record exact ordered held-out samples used for these saved predictions.
write_test_sample_manifest(results_path, y_test)
