# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *

# ================================================================
# RANDOM FOREST CLASSIFIER
# ================================================================

import numpy as np
import joblib

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

# ================================================================
# 1. LOAD CNN FEATURES
# ================================================================

print("\n" + "=" * 70)
print("LOADING CNN FEATURES")
print("=" * 70)

X_train = np.load(FEATURE_DIR / "X_train_features.npy")
y_train = np.load(FEATURE_DIR / "y_train.npy")

X_val = np.load(FEATURE_DIR / "X_val_features.npy")
y_val = np.load(FEATURE_DIR / "y_val.npy")

X_test = np.load(FEATURE_DIR / "X_test_features.npy")
y_test = np.load(FEATURE_DIR / "y_test.npy")

print(f"X_train: {X_train.shape}")
print(f"X_val  : {X_val.shape}")
print(f"X_test : {X_test.shape}")


# ================================================================
# 2. BUILD RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("BUILDING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)


# ================================================================
# 3. TRAIN
# ================================================================

print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)


# ================================================================
# 4. PREDICTIONS
# ================================================================

y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

y_train_prob = rf_model.predict_proba(X_train)
y_val_prob = rf_model.predict_proba(X_val)
y_test_prob = rf_model.predict_proba(X_test)


# ================================================================
# 5. EVALUATION FUNCTION
# ================================================================

def evaluate_random_forest(
    y_true,
    y_pred,
    y_prob,
    dataset_name
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    loss = log_loss(
        y_true,
        y_prob
    )

    print("\n" + "-" * 70)
    print(dataset_name)
    print("-" * 70)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall   : {recall:.4f}")
    print(f"Weighted F1-score : {f1:.4f}")
    print(f"Macro ROC-AUC     : {roc_auc:.4f}")
    print(f"Log Loss          : {loss:.4f}")

    return {
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1-score": f1,
        "Macro ROC-AUC": roc_auc,
        "Log Loss": loss
    }


# ================================================================
# 6. EVALUATE
# ================================================================

train_results = evaluate_random_forest(
    y_train,
    y_train_pred,
    y_train_prob,
    "TRAINING SET"
)

val_results = evaluate_random_forest(
    y_val,
    y_val_pred,
    y_val_prob,
    "VALIDATION SET"
)

test_results = evaluate_random_forest(
    y_test,
    y_test_pred,
    y_test_prob,
    "TEST SET"
)


# ================================================================
# 7. CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ================================================================
# 8. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(cm)


# ================================================================
# 9. SAVE MODEL
# ================================================================

joblib.dump(rf_model, RF_RESULT_DIR / "random_forest_model.pkl")

print("\nRandom Forest model saved successfully.")


# ================================================================
# 10. SAVE PREDICTIONS AND PROBABILITIES
# ================================================================

np.save(RF_RESULT_DIR / "rf_test_predictions.npy", y_test_pred)

np.save(RF_RESULT_DIR / "rf_test_probabilities.npy", y_test_prob)

print("Predictions and probabilities saved successfully.")


# ================================================================
# 11. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy          : "
    f"{test_results['Accuracy']:.4f}"
)

print(
    f"Weighted Precision: "
    f"{test_results['Weighted Precision']:.4f}"
)

print(
    f"Weighted Recall   : "
    f"{test_results['Weighted Recall']:.4f}"
)

print(
    f"Weighted F1-score : "
    f"{test_results['Weighted F1-score']:.4f}"
)

print(
    f"Macro ROC-AUC     : "
    f"{test_results['Macro ROC-AUC']:.4f}"
)

print(
    f"Log Loss          : "
    f"{test_results['Log Loss']:.4f}"
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
import lightgbm as lgb
from sklearn.preprocessing import label_binarize
import joblib
