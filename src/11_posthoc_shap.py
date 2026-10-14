"""Grouped post-hoc SHAP analysis for the trained stacking meta-learners.

Run after the main pipelines. The analysis uses saved meta-features and fitted
logistic-regression meta-learners; it does not retrain the base CNN/classifiers.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from config import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
rng = np.random.default_rng(SEED)

TASKS = {
    "dyslexia": {
        "result_dir": OUTPUT_ROOT / "results" / "stacking",
        "model_file": "meta_learner_logistic_regression.pkl",
        "background_file": "X_meta_train.npy",
        "test_file": "X_meta_test.npy",
        "n_classes_per_model": 3,
        "model_names": ["SVM", "MLP", "Random Forest", "XGBoost", "LightGBM"],
        "target_class_index": 1,  # Reversal, according to manuscript label order
        "test_limit": 80,
    },
    "dysgraphia": {
        "result_dir": OUTPUT_ROOT / "results" / "dysgraphia",
        "model_file": "stacking_meta_learner.joblib",
        "background_file": "oof_meta_features.npy",
        "test_file": "test_meta_features.npy",
        "n_classes_per_model": 2,
        "model_names": ["MLP", "SVM", "Random Forest", "XGBoost", "LightGBM"],
        "target_class_index": 1,  # Potential Dysgraphia
        "test_limit": None,
    },
}


def main():
    for task, cfg in TASKS.items():
        out_dir = cfg["result_dir"]
        model_path = out_dir / cfg["model_file"]
        bg_path = out_dir / cfg["background_file"]
        test_path = out_dir / cfg["test_file"]
        missing = [str(p) for p in (model_path, bg_path, test_path) if not p.exists()]
        if missing:
            warnings.warn(f"Skipping {task}; run its pipeline first. Missing: {missing}")
            continue

        model = joblib.load(model_path)
        background_all = np.load(bg_path)
        test_all = np.load(test_path)
        if background_all.ndim != 2 or test_all.ndim != 2:
            raise ValueError(f"Expected 2D meta-features for {task}.")
        expected = len(cfg["model_names"]) * cfg["n_classes_per_model"]
        if background_all.shape[1] != expected or test_all.shape[1] != expected:
            raise ValueError(
                f"{task}: expected {expected} meta-features per manuscript protocol, "
                f"found train={background_all.shape[1]}, test={test_all.shape[1]}."
            )

        bg_n = min(50, len(background_all))
        bg_idx = rng.choice(len(background_all), size=bg_n, replace=False)
        background = background_all[bg_idx]
        test_n = min(cfg["test_limit"], len(test_all)) if cfg["test_limit"] else len(test_all)
        test_idx = np.arange(test_n)
        test = test_all[test_idx]

        target = cfg["target_class_index"]
        def model_output(x):
            return model.predict_proba(np.asarray(x))[:, target]

        masker = shap.maskers.Independent(background)
        explainer = shap.PermutationExplainer(model_output, masker, seed=SEED)
        # SHAP max_evals is a model-evaluation budget. Set it to the number
        # of evaluations corresponding to 100 forward/reverse permutation cycles.
        permutation_cycles = 100
        max_evals = 2 * test.shape[1] * permutation_cycles + 1
        values = explainer(test, max_evals=max_evals).values
        if values.ndim == 3:
            values = values[..., 0]

        # Group each classifier's class-probability columns as described in Section 3.7.
        grouped = np.zeros((len(test), len(cfg["model_names"])), dtype=float)
        target_probability_features = np.zeros_like(grouped)
        for i in range(len(cfg["model_names"])):
            start = i * cfg["n_classes_per_model"]
            end = start + cfg["n_classes_per_model"]
            grouped[:, i] = values[:, start:end].sum(axis=1)
            # Feature color in the manuscript plot represents the target-class
            # probability emitted by this base classifier.
            target_probability_features[:, i] = test[:, start + target]

        names = cfg["model_names"]
        pd.DataFrame(grouped, columns=names).to_csv(out_dir / "shap_grouped_values.csv", index=False)
        pd.DataFrame(target_probability_features, columns=names).to_csv(
            out_dir / "shap_grouped_feature_values.csv", index=False
        )
        summary = pd.DataFrame({
            "base_classifier": names,
            "mean_absolute_grouped_shap": np.mean(np.abs(grouped), axis=0),
            "mean_grouped_shap": np.mean(grouped, axis=0),
            "target_class_index": target,
            "background_n": bg_n,
            "explained_test_n": len(test),
            "seed": SEED,
            "permutation_cycles": permutation_cycles,
            "max_evals_model_evaluations": max_evals,
        }).sort_values("mean_absolute_grouped_shap", ascending=False)
        summary.to_csv(out_dir / "shap_grouped_summary.csv", index=False)

        # Grouped beeswarm summary: SHAP value is the horizontal coordinate;
        # target-class probability is used for feature coloring, matching the
        # manuscript's stated visualization semantics.
        shap.summary_plot(
            grouped,
            features=target_probability_features,
            feature_names=names,
            show=False,
            plot_size=(8, 4.8),
            max_display=len(names),
        )
        plt.title(f"Grouped SHAP — {task.title()}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_grouped_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved grouped SHAP results for {task} to {out_dir}")

if __name__ == "__main__":
    main()
