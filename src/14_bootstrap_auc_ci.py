"""Compute sample-level bootstrap 95% CIs for ROC-AUC from saved test outputs.

Reads saved held-out test labels and probability arrays from the task-specific output
paths below. Uses 2,000 bootstrap replicates and seed 42. Results are computed from
these artifacts and are not hard-coded. The source manifest is saved with the results.
Replicates containing only one class are skipped for AUC calculation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from config import PROJECT_ROOT, OUTPUT_ROOT, FEATURE_DIR, CLASS_NAMES, SEED

N_BOOT = 2000
TASKS = {
    "dyslexia": {
        "out": OUTPUT_ROOT / "results" / "bootstrap_dyslexia",
        "y_candidates": [FEATURE_DIR / "y_test.npy", OUTPUT_ROOT / "results/stacking/y_test_true.npy"],
        "models": {
            "Stacking": OUTPUT_ROOT / "results/stacking/y_test_proba_stacking.npy",
            "MLP": OUTPUT_ROOT / "results/mlp/y_proba_mlp.npy",
            "SVM": OUTPUT_ROOT / "results/svm/y_proba_svm_revised.npy",
            "Random Forest": OUTPUT_ROOT / "results/random_forest/rf_test_probabilities.npy",
            "LightGBM": OUTPUT_ROOT / "results/lightgbm/y_proba_lgbm_revised.npy",
            "XGBoost": OUTPUT_ROOT / "results/xgboost/y_proba_xgb_revised.npy",
        },
        "multiclass": True,
        "manifest_paths": [
            OUTPUT_ROOT / "results/stacking/test_sample_manifest.json",
            OUTPUT_ROOT / "results/mlp/test_sample_manifest.json",
            OUTPUT_ROOT / "results/svm/test_sample_manifest.json",
            OUTPUT_ROOT / "results/random_forest/test_sample_manifest.json",
            OUTPUT_ROOT / "results/xgboost/test_sample_manifest.json",
            OUTPUT_ROOT / "results/lightgbm/test_sample_manifest.json",
        ],
    },
    "dysgraphia": {
        "out": OUTPUT_ROOT / "results" / "bootstrap_dysgraphia",
        "y_candidates": [OUTPUT_ROOT / "results/dysgraphia/y_test_true.npy"],
        "models": {
            "Stacking": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_stacking.npy",
            "MLP": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_mlp.npy",
            "SVM": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_svm.npy",
            "Random Forest": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_random_forest.npy",
            "XGBoost": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_xgboost.npy",
            "LightGBM": OUTPUT_ROOT / "results/dysgraphia/y_test_proba_lightgbm.npy",
        },
        "multiclass": False,
        "manifest_paths": [OUTPUT_ROOT / "results/dysgraphia/test_sample_manifest.json"],
    },
}

def load_y(candidates):
    """Load test labels and fail if multiple available sources disagree."""
    available = [path for path in candidates if path.exists()]
    if not available:
        raise FileNotFoundError("No saved test labels found. Expected one of: " + ", ".join(map(str, candidates)))
    loaded = []
    for path in available:
        y = np.load(path)
        if y.ndim == 2 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        loaded.append((path, np.asarray(y).reshape(-1).astype(int)))
    reference_path, reference_y = loaded[0]
    for path, y in loaded[1:]:
        if y.shape != reference_y.shape or not np.array_equal(y, reference_y):
            raise ValueError(
                "Available test-label sources disagree in shape or order: "
                f"{reference_path} vs {path}. Refusing to calculate bootstrap CIs "
                "until the test split and label ordering are reconciled."
            )
    return reference_y

def validate_test_manifests(cfg, y, task):
    """Require a current-run manifest tying saved labels/probabilities to ordered test images."""
    paths = cfg.get("manifest_paths", [])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{task} bootstrap requires current-run test_sample_manifest.json files. "
            "Rerun the relevant model stages before computing CIs. Missing: " + ", ".join(missing)
        )
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = manifests[0]
    label_hash = hashlib.sha256(np.asarray(y, dtype=np.int64).tobytes()).hexdigest()
    if reference.get("n_test") != len(y) or reference.get("test_labels_sha256") != label_hash:
        raise ValueError(f"{task} bootstrap labels do not match the current test manifest.")
    for path, manifest in zip(paths[1:], manifests[1:]):
        if (manifest.get("ordered_paths_sha256") != reference.get("ordered_paths_sha256")
                or manifest.get("test_labels_sha256") != reference.get("test_labels_sha256")
                or manifest.get("n_test") != reference.get("n_test")):
            raise ValueError(
                f"{task} bootstrap refused: test-image identity/order mismatch in {path}. "
                "Regenerate predictions and manifests from the same test split."
            )
    return manifests


def auc(y, p, multiclass):
    if multiclass:
        return float(roc_auc_score(y, p, labels=np.arange(p.shape[1]), multi_class="ovr", average="macro"))
    p = np.asarray(p)
    positive = p[:, 1] if p.ndim == 2 else p.reshape(-1)
    return float(roc_auc_score(y, positive))

def main():
    all_results = {"seed": SEED, "bootstrap_replicates_requested": N_BOOT, "interval": "percentile 95% CI", "unit": "image/sample-level; not participant-level"}
    for task, cfg in TASKS.items():
        try: y = load_y(cfg["y_candidates"])
        except FileNotFoundError as e:
            print(f"SKIP {task}: {e}"); continue
        try:
            manifests = validate_test_manifests(cfg, y, task)
        except (FileNotFoundError, ValueError) as e:
            print(f"STOP {task}: {e}")
            continue
        expected_label_values = set(range(len(CLASS_NAMES) if cfg["multiclass"] else 2))
        if not set(np.unique(y)).issubset(expected_label_values):
            raise ValueError(
                f"{task} test labels contain values outside the expected class encoding "
                f"{sorted(expected_label_values)}: {sorted(np.unique(y).tolist())}"
            )
        task_results = {}
        for name, path in cfg["models"].items():
            # Stable stream keyed by task/model identity. Inserting or reordering
            # entries in TASKS does not change an existing model's bootstrap draws.
            stream_key = f"{task}::{name}".encode("utf-8")
            stream_offset = int.from_bytes(hashlib.sha256(stream_key).digest()[:4], "big")
            rng = np.random.default_rng(SEED + stream_offset)
            if not path.exists():
                print(f"SKIP {task}/{name}: missing {path}"); continue
            p = np.load(path)
            if len(p) != len(y):
                print(f"SKIP {task}/{name}: {len(p)} predictions vs {len(y)} labels"); continue
            expected_cols = len(CLASS_NAMES) if cfg["multiclass"] else 2
            if p.ndim != 2 or p.shape[1] != expected_cols:
                print(f"SKIP {task}/{name}: expected probability shape (n, {expected_cols}), got {p.shape}"); continue
            if not np.all(np.isfinite(p)):
                print(f"SKIP {task}/{name}: probability array contains non-finite values"); continue
            if np.any(p < -1e-8) or np.any(p > 1 + 1e-8):
                print(f"SKIP {task}/{name}: probabilities fall outside [0, 1]"); continue
            if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
                print(f"SKIP {task}/{name}: probability rows do not sum to one; check output/class order"); continue
            try: point = auc(y,p,cfg["multiclass"])
            except Exception as e:
                print(f"SKIP {task}/{name}: point AUC failed: {e}"); continue
            boot = []
            for _ in range(N_BOOT):
                idx = rng.integers(0,len(y),size=len(y))
                yb = y[idx]
                if len(np.unique(yb)) < 2: continue
                try: boot.append(auc(yb,p[idx],cfg["multiclass"]))
                except (ValueError, IndexError): continue
            if not boot:
                print(f"SKIP {task}/{name}: no valid bootstrap replicates"); continue
            task_results[name] = {
                "n_test": int(len(y)), "point_auc": point,
                "ci95_percentile": [float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))],
                "valid_bootstrap_replicates": int(len(boot)),
                "requested_bootstrap_replicates": N_BOOT,
            }
        cfg["out"].mkdir(parents=True,exist_ok=True)
        with open(cfg["out"] / "bootstrap_auc_ci.json","w",encoding="utf-8") as f: json.dump(task_results,f,indent=2)
        source_manifest = {
            "task": task,
            "test_label_candidates": [str(p) for p in cfg["y_candidates"]],
            "probability_sources": {name: str(path) for name, path in cfg["models"].items()},
            "test_manifest_paths": [str(path) for path in cfg.get("manifest_paths", [])],
            "ordered_paths_sha256": manifests[0].get("ordered_paths_sha256"),
            "test_labels_sha256": manifests[0].get("test_labels_sha256"),
            "bootstrap_replicates": N_BOOT,
            "seed": SEED,
            "resampling_unit": "image/sample",
            "interval": "percentile 95% CI",
        }
        with open(cfg["out"] / "bootstrap_source_manifest.json", "w", encoding="utf-8") as f:
            json.dump(source_manifest, f, indent=2)
        all_results[task] = task_results
    out = OUTPUT_ROOT / "results" / "bootstrap_auc_ci_all.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w",encoding="utf-8") as f: json.dump(all_results,f,indent=2)
    print(f"Saved bootstrap summaries to {out}. Compare all values with manuscript tables before reporting.")

if __name__ == "__main__": main()
