# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *

# =======================
# المسارات
# =======================
base_path = str(PROJECT_ROOT)

features_path = str(FEATURE_DIR)

results_path = str(LGBM_RESULT_DIR)

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
# تدريب LightGBM
# Settings as specified in the paper
# =======================
print("\n=======================")
print("TRAINING LIGHTGBM")
print("=======================")

lgbm_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100
)

lgbm_model.fit(
    X_train_features,
    y_train
)

print(
    "LightGBM model "
    "trained successfully."
)


# =======================
# Validation
# =======================
print("\n=======================")
print("VALIDATION")
print("=======================")

y_val_pred = lgbm_model.predict(
    X_val_features
)

y_val_proba = lgbm_model.predict_proba(
    X_val_features
)

lgbm_val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

print(
    f"LightGBM Validation Accuracy: "
    f"{lgbm_val_accuracy * 100:.2f}%"
)


# =======================
# Test prediction
# =======================
print("\n=======================")
print("TEST PREDICTION")
print("=======================")

y_pred_lgbm = lgbm_model.predict(
    X_test_features
)

y_pred_lgbm_proba = (
    lgbm_model.predict_proba(
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


lgbm_accuracy = accuracy_score(
    y_test,
    y_pred_lgbm
)

lgbm_precision = precision_score(
    y_test,
    y_pred_lgbm,
    average="weighted"
)

lgbm_recall = recall_score(
    y_test,
    y_pred_lgbm,
    average="weighted"
)

lgbm_f1 = f1_score(
    y_test,
    y_pred_lgbm,
    average="weighted"
)

# Macro ROC-AUC, one-vs-rest
lgbm_roc_auc = roc_auc_score(
    y_test_binarized,
    y_pred_lgbm_proba,
    multi_class="ovr",
    average="macro"
)

lgbm_log_loss = log_loss(
    y_test_binarized,
    y_pred_lgbm_proba
)


# =======================
# Final Results
# =======================
print("\n===================================")
print("FINAL LIGHTGBM TEST RESULTS")
print("===================================")

print(
    f"Accuracy        : "
    f"{lgbm_accuracy * 100:.2f}%"
)

print(
    f"Precision (W)   : "
    f"{lgbm_precision:.4f}"
)

print(
    f"Recall (W)      : "
    f"{lgbm_recall:.4f}"
)

print(
    f"F1-Score (W)    : "
    f"{lgbm_f1:.4f}"
)

print(
    f"ROC-AUC (Macro) : "
    f"{lgbm_roc_auc:.4f}"
)

print(
    f"Log Loss        : "
    f"{lgbm_log_loss:.4f}"
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
        y_pred_lgbm,
        target_names=categories
    )
)


# =======================
# Confusion Matrix
# =======================
lgbm_conf_matrix = confusion_matrix(
    y_test,
    y_pred_lgbm
)

print(
    "\nConfusion Matrix:\n"
)

print(
    lgbm_conf_matrix
)


# =======================
# Plot Confusion Matrix
# =======================
plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    lgbm_conf_matrix,
    annot=True,
    fmt="d",
    cmap="Purples",
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
    "Confusion Matrix - LightGBM Model"
)

plt.tight_layout()

plt.show()


# =======================
# Save model
# =======================
lgbm_model_path = os.path.join(
    results_path,
    "lgbm_model_revised.pkl"
)

joblib.dump(
    lgbm_model,
    lgbm_model_path
)


# =======================
# Save test predictions
# =======================
np.save(
    os.path.join(
        results_path,
        "y_pred_lgbm_revised.npy"
    ),
    y_pred_lgbm
)

np.save(
    os.path.join(
        results_path,
        "y_proba_lgbm_revised.npy"
    ),
    y_pred_lgbm_proba
)


# =======================
# Save validation predictions
# =======================
np.save(
    os.path.join(
        results_path,
        "y_val_pred_lgbm_revised.npy"
    ),
    y_val_pred
)

np.save(
    os.path.join(
        results_path,
        "y_val_proba_lgbm_revised.npy"
    ),
    y_val_proba
)


# =======================
# DONE
# =======================
print("\n===================================")
print("LIGHTGBM COMPLETED")
print("===================================")

print(
    f"Results saved to:\n"
    f"{results_path}"
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

import os
import xgboost as xgb
from sklearn.preprocessing import label_binarize
import joblib
