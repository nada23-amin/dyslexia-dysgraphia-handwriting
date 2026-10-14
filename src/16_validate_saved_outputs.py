"""Validate saved test predictions and metric consistency after running pipelines.

This utility does not train models and does not alter any reported values. It
checks prediction/label lengths, class-probability shapes, confusion-matrix totals,
and the identity between accuracy and weighted recall in single-label tasks.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from config import PROJECT_ROOT, FEATURE_DIR

TASKS = {
    "dyslexia": {
        "root": PROJECT_ROOT / "outputs" / "results",
        "y_candidates": [FEATURE_DIR / "y_test.npy", PROJECT_ROOT / "outputs/results/stacking/y_test_true.npy"],
        "models": {
            "stacking": ("stacking/y_test_pred_stacking.npy", "stacking/y_test_proba_stacking.npy"),
            "mlp": ("mlp/y_pred_mlp.npy", "mlp/y_proba_mlp.npy"),
            "svm": ("svm/y_pred_svm_revised.npy", "svm/y_proba_svm_revised.npy"),
            "random_forest": ("random_forest/rf_test_predictions.npy", "random_forest/rf_test_probabilities.npy"),
            "lightgbm": ("lightgbm/y_pred_lgbm_revised.npy", "lightgbm/y_proba_lgbm_revised.npy"),
            "xgboost": ("xgboost/y_pred_xgb_revised.npy", "xgboost/y_proba_xgb_revised.npy"),
        },
        "n_classes": 3,
    },
    "dysgraphia": {
        "root": PROJECT_ROOT / "outputs" / "results",
        "y_candidates": [PROJECT_ROOT / "outputs/results/dysgraphia/y_test_true.npy"],
        "models": {
            "stacking": ("dysgraphia/y_test_pred_stacking.npy", "dysgraphia/y_test_proba_stacking.npy"),
            "mlp": ("dysgraphia/y_test_pred_mlp.npy", "dysgraphia/y_test_proba_mlp.npy"),
            "svm": ("dysgraphia/y_test_pred_svm.npy", "dysgraphia/y_test_proba_svm.npy"),
            "random_forest": ("dysgraphia/y_test_pred_random_forest.npy", "dysgraphia/y_test_proba_random_forest.npy"),
            "xgboost": ("dysgraphia/y_test_pred_xgboost.npy", "dysgraphia/y_test_proba_xgboost.npy"),
            "lightgbm": ("dysgraphia/y_test_pred_lightgbm.npy", "dysgraphia/y_test_proba_lightgbm.npy"),
        },
        "n_classes": 2,
    },
}

def load_labels(paths):
    for path in paths:
        if path.exists():
            y = np.load(path)
            if y.ndim == 2 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            return np.asarray(y).reshape(-1).astype(int), path
    raise FileNotFoundError("No saved y_test file found: " + ", ".join(map(str, paths)))

def main():
    report = {"validation_only": True, "tasks": {}}
    for task, cfg in TASKS.items():
        try:
            y, y_path = load_labels(cfg["y_candidates"])
        except FileNotFoundError as exc:
            report["tasks"][task] = {"status": "not_run", "reason": str(exc)}
            continue
        task_report = {"status": "checked", "y_test_source": str(y_path), "n_test": int(len(y)), "models": {}}
        for name, (pred_rel, proba_rel) in cfg["models"].items():
            pred_path, proba_path = cfg["root"] / pred_rel, cfg["root"] / proba_rel
            if not pred_path.exists() or not proba_path.exists():
                task_report["models"][name] = {"status": "missing_output", "missing": [str(p) for p in (pred_path, proba_path) if not p.exists()]}
                continue
            pred = np.asarray(np.load(pred_path)).reshape(-1).astype(int)
            proba = np.asarray(np.load(proba_path))
            issues = []
            if len(pred) != len(y): issues.append(f"prediction length {len(pred)} != labels {len(y)}")
            if proba.ndim != 2 or proba.shape[1] != cfg["n_classes"]: issues.append(f"probability shape {proba.shape} expected (*,{cfg['n_classes']})")
            if len(proba) != len(y): issues.append(f"probability rows {len(proba)} != labels {len(y)}")
            if not issues:
                acc = float(accuracy_score(y, pred))
                weighted_recall = float(recall_score(y, pred, average="weighted", zero_division=0))
                cm = confusion_matrix(y, pred, labels=list(range(cfg["n_classes"])))
                if not np.isclose(acc, weighted_recall, atol=1e-12):
                    issues.append("accuracy and weighted recall differ despite using the same single-label predictions")
                if int(cm.sum()) != len(y): issues.append("confusion matrix total does not equal test-set size")
                if not np.all(np.isfinite(proba)): issues.append("probabilities contain non-finite values")
                if np.any(proba < -1e-8) or np.any(proba > 1 + 1e-8): issues.append("probabilities fall outside [0,1]")
                if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-3): issues.append("probability rows do not sum to approximately 1")
                model_result = {"n_test": int(len(y)), "accuracy": acc, "weighted_recall": weighted_recall, "confusion_matrix": cm.tolist(), "issues": issues}
            else:
                model_result = {"issues": issues}
            model_result["status"] = "pass" if not issues else "needs_review"
            task_report["models"][name] = model_result
        report["tasks"][task] = task_report
    out = PROJECT_ROOT / "outputs" / "results" / "saved_output_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved validation report: {out}")
    print("This validates saved-output consistency only; it does not prove that historical manuscript values were generated by this run.")

if __name__ == "__main__":
    main()
