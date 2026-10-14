"""Save Dyslexia test predictions and exact paired McNemar tests.

The Stacking predictions are produced from saved test meta-features and the fitted
Logistic Regression meta-learner. Comparator predictions are loaded from the
standalone classifier scripts' saved test predictions. Thus each McNemar test
compares Stacking with an independently trained classifier on the same test labels,
not with fold-specific base-model predictions from the stacking procedure.
"""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "outputs" / "features"
RESULT_ROOT = ROOT / "outputs" / "results"
RESULT_DIR = RESULT_ROOT / "stacking"
MODEL_PATH = RESULT_DIR / "meta_learner_logistic_regression.pkl"

MODEL_NAMES = ["SVM", "MLP", "Random Forest", "XGBoost", "LightGBM"]
N_CLASSES = 3
STANDALONE_PREDICTIONS = {
    "SVM": RESULT_ROOT / "svm" / "y_pred_svm_revised.npy",
    "MLP": RESULT_ROOT / "mlp" / "y_pred_mlp.npy",
    "Random Forest": RESULT_ROOT / "random_forest" / "rf_test_predictions.npy",
    "XGBoost": RESULT_ROOT / "xgboost" / "y_pred_xgb_revised.npy",
    "LightGBM": RESULT_ROOT / "lightgbm" / "y_pred_lgbm_revised.npy",
}
STANDALONE_PROBABILITIES = {
    "SVM": RESULT_ROOT / "svm" / "y_proba_svm_revised.npy",
    "MLP": RESULT_ROOT / "mlp" / "y_proba_mlp.npy",
    "Random Forest": RESULT_ROOT / "random_forest" / "rf_test_probabilities.npy",
    "XGBoost": RESULT_ROOT / "xgboost" / "y_proba_xgb_revised.npy",
    "LightGBM": RESULT_ROOT / "lightgbm" / "y_proba_lgbm_revised.npy",
}
N_CLASSES = 3


def load_labels(path: Path) -> np.ndarray:
    y = np.load(path)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return np.asarray(y).reshape(-1).astype(int)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    y_path = FEATURE_DIR / "y_test.npy"
    meta_path = RESULT_DIR / "X_meta_test.npy"
    missing = [str(path) for path in (y_path, meta_path, MODEL_PATH) if not path.exists()]
    missing.extend(str(path) for path in STANDALONE_PREDICTIONS.values() if not path.exists())
    missing.extend(str(path) for path in STANDALONE_PROBABILITIES.values() if not path.exists())
    if missing:
        raise FileNotFoundError(
            "Required stacking and standalone-classifier outputs are missing. Run the "
            "feature extraction, standalone classifier, and stacking stages first. Missing: "
            + ", ".join(missing)
        )

    y_true = load_labels(y_path)
    # Shape equality is not enough for paired testing: verify every saved model
    # artifact was produced against the same ordered held-out image manifest.
    manifest_paths = [RESULT_DIR / "test_sample_manifest.json"]
    manifest_paths.extend(
        path.parent / "test_sample_manifest.json" for path in STANDALONE_PREDICTIONS.values()
    )
    missing_manifests = [str(path) for path in manifest_paths if not path.exists()]
    if missing_manifests:
        raise FileNotFoundError(
            "McNemar requires test_sample_manifest.json from the current run for Stacking "
            "and every standalone classifier. Rerun feature extraction, all standalone "
            "classifiers, and stacking before this analysis. Missing: " + ", ".join(missing_manifests)
        )
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    reference_manifest = manifests[0]
    for path, manifest in zip(manifest_paths[1:], manifests[1:]):
        if (manifest.get("ordered_paths_sha256") != reference_manifest.get("ordered_paths_sha256")
                or manifest.get("test_labels_sha256") != reference_manifest.get("test_labels_sha256")
                or manifest.get("n_test") != reference_manifest.get("n_test")):
            raise ValueError(
                f"Test-image identity/order mismatch in {path}. Rerun all models from one clean split "
                "before interpreting paired McNemar results."
            )
    if int(reference_manifest.get("n_test", -1)) != len(y_true):
        raise ValueError("Test manifest sample count does not match the labels loaded for McNemar.")
    if np.any((y_true < 0) | (y_true >= N_CLASSES)):
        raise ValueError(f"Test labels must be encoded as integers 0..{N_CLASSES - 1}.")
    X_meta_test = np.load(meta_path)
    meta_model = joblib.load(MODEL_PATH)
    expected = len(MODEL_NAMES) * N_CLASSES
    if X_meta_test.ndim != 2 or X_meta_test.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} Dyslexia meta-features (5 models x 3 classes), "
            f"found shape {X_meta_test.shape}. Check the stacking pipeline."
        )
    if len(y_true) != len(X_meta_test):
        raise ValueError("Test labels and stacking meta-features have different sample counts.")

    stack_prob = meta_model.predict_proba(X_meta_test)
    stack_pred = np.argmax(stack_prob, axis=1)
    predictions = {"Stacking": stack_pred}
    probabilities = {"Stacking": stack_prob}

    for name in MODEL_NAMES:
        pred = load_labels(STANDALONE_PREDICTIONS[name])
        prob = np.load(STANDALONE_PROBABILITIES[name])
        if len(pred) != len(y_true) or len(prob) != len(y_true):
            raise ValueError(
                f"The saved independent {name} predictions do not match the test-set size. "
                "Rerun the standalone classifier on the same feature split."
            )
        if prob.ndim != 2 or prob.shape[1] != N_CLASSES:
            raise ValueError(f"Expected {N_CLASSES} class probabilities for {name}; got {prob.shape}.")
        if np.any((pred < 0) | (pred >= N_CLASSES)):
            raise ValueError(f"Invalid class labels found in standalone predictions for {name}.")
        if not np.all(np.isfinite(prob)) or np.any(prob < -1e-8) or np.any(prob > 1 + 1e-8):
            raise ValueError(f"Standalone probabilities for {name} contain invalid values.")
        if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError(f"Standalone probabilities for {name} do not sum to one; check class order/output.")
        if not np.array_equal(pred, np.argmax(prob, axis=1)):
            raise ValueError(
                f"Saved predictions and probability argmax disagree for {name}. "
                "Regenerate both outputs from the same model run before McNemar testing."
            )
        predictions[name] = pred
        probabilities[name] = prob

    np.save(RESULT_DIR / "y_test_true.npy", y_true)
    metrics = {}
    for name, pred in predictions.items():
        safe = name.lower().replace(" ", "_")
        np.save(RESULT_DIR / f"y_test_pred_{safe}.npy", pred)
        np.save(RESULT_DIR / f"y_test_proba_{safe}.npy", probabilities[name])
        cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
        np.savetxt(RESULT_DIR / f"confusion_matrix_{safe}.csv", cm, delimiter=",", fmt="%d")
        metrics[name] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "classification_report": classification_report(
                y_true, pred, labels=[0, 1, 2], output_dict=True, zero_division=0
            ),
        }

    paired = {}
    stack_correct = predictions["Stacking"] == y_true
    for name in MODEL_NAMES:
        model_correct = predictions[name] == y_true
        table = np.array([
            [np.sum(stack_correct & model_correct), np.sum(stack_correct & ~model_correct)],
            [np.sum(~stack_correct & model_correct), np.sum(~stack_correct & ~model_correct)],
        ], dtype=int)
        result = mcnemar(table, exact=True)
        paired[f"Stacking_vs_{name}"] = {
            "comparison": "stacking_vs_independently_trained_classifier",
            "stacking_prediction_source": str(RESULT_DIR / "X_meta_test.npy") + " + " + str(MODEL_PATH),
            "comparator_prediction_source": str(STANDALONE_PREDICTIONS[name]),
            "test_label_source": str(y_path),
            "test_split": "same held-out dyslexia test image split",
            "contingency_table": table.tolist(),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n_test": int(len(y_true)),
        }

    with open(RESULT_DIR / "test_metrics_from_saved_meta_features.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULT_DIR / "mcnemar_exact_paired.json", "w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2)
    print(f"Saved Dyslexia test predictions, confusion matrices, metrics and McNemar results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
