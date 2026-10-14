"""Central configuration for the public reproducibility package."""
from pathlib import Path
import os
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODE = os.environ.get("ABLATION_MODE", "standard").strip().lower()
# Each ablation run receives isolated generated outputs so variants cannot overwrite
# the full-framework outputs or one another.
OUTPUT_ROOT = (PROJECT_ROOT / "outputs" if ABLATION_MODE == "standard"
               else PROJECT_ROOT / "outputs" / "ablation" / ABLATION_MODE)

# Users can override the dataset location without editing source code.
# Example (PowerShell):
#   $env:DYSLEXIA_DATA_DIR = "C:\\path\\to\\Handwriting"
# If unset, the repository expects data/raw/{Corrected,Normal,Reversal}.
_DATA_ENV = os.environ.get("DYSLEXIA_DATA_DIR")
RAW_DIR = Path(_DATA_ENV).expanduser().resolve() if _DATA_ENV else PROJECT_ROOT / "data" / "raw"
DATA_DIR = RAW_DIR

PROCESSED_DIR = OUTPUT_ROOT / "processed_data"
SPLIT_DIR = OUTPUT_ROOT / "splits"
BALANCED_DIR = OUTPUT_ROOT / "balanced_train"
MODEL_DIR = OUTPUT_ROOT / "models"
FEATURE_DIR = OUTPUT_ROOT / "features"
RESULT_DIR = OUTPUT_ROOT / "results"
FIGURE_DIR = OUTPUT_ROOT / "figures"
CNN_RESULT_DIR = MODEL_DIR / "cnn"
MLP_RESULT_DIR = RESULT_DIR / "mlp"
SVM_RESULT_DIR = RESULT_DIR / "svm"
RF_RESULT_DIR = RESULT_DIR / "random_forest"
LGBM_RESULT_DIR = RESULT_DIR / "lightgbm"
XGB_RESULT_DIR = RESULT_DIR / "xgboost"
STACKING_RESULT_DIR = RESULT_DIR / "stacking"

CLASSES = ["Corrected", "Normal", "Reversal"]
CLASS_NAMES = ["Normal", "Reversal", "Corrected"]
NUM_CLASSES = 3
SEED = 42
IMG_SIZE = (28, 28)
BATCH_SIZE = 64
EPOCHS = 20
FEATURE_DIM = 128
TARGET_TRAIN_SIZE = None  # computed dynamically from the largest training class

_OUTPUT_DIRS = [
    PROCESSED_DIR, SPLIT_DIR, BALANCED_DIR, MODEL_DIR, FEATURE_DIR,
    RESULT_DIR, FIGURE_DIR, CNN_RESULT_DIR, MLP_RESULT_DIR,
    SVM_RESULT_DIR, RF_RESULT_DIR, LGBM_RESULT_DIR, XGB_RESULT_DIR,
    STACKING_RESULT_DIR,
]
for _path in _OUTPUT_DIRS:
    _path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def write_test_sample_manifest(output_dir, expected_labels):
    """Write an auditable link between a model's saved test outputs and the exact ordered test split."""
    import hashlib
    import json
    output_dir = Path(output_dir)
    path_file = FEATURE_DIR / "test_sample_paths.json"
    labels_file = FEATURE_DIR / "y_test.npy"
    if not path_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            "Cannot write test manifest: rerun CNN feature extraction to create "
            "features/test_sample_paths.json and features/y_test.npy first."
        )
    paths = json.loads(path_file.read_text(encoding="utf-8"))
    labels = np.asarray(np.load(labels_file)).reshape(-1)
    expected = np.asarray(expected_labels).reshape(-1)
    if len(paths) != len(labels) or len(expected) != len(labels) or not np.array_equal(expected.astype(int), labels.astype(int)):
        raise ValueError("Test manifest paths/labels do not match the labels used by this model run.")
    canonical_paths = json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    canonical_labels = np.asarray(labels, dtype=np.int64).tobytes()
    payload = {
        "task": "dyslexia",
        "n_test": int(len(labels)),
        "ordered_test_paths": paths,
        "ordered_paths_sha256": hashlib.sha256(canonical_paths).hexdigest(),
        "test_labels_sha256": hashlib.sha256(canonical_labels).hexdigest(),
        "label_source": str(labels_file),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_sample_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
