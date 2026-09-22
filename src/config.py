"""Central configuration for the public reproducibility package."""
from pathlib import Path
import os
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Users can override the dataset location without editing source code.
# Example (PowerShell):
#   $env:DYSLEXIA_DATA_DIR = "C:\\path\\to\\Handwriting"
# If unset, the repository expects data/raw/{Corrected,Normal,Reversal}.
_DATA_ENV = os.environ.get("DYSLEXIA_DATA_DIR")
RAW_DIR = Path(_DATA_ENV).expanduser().resolve() if _DATA_ENV else PROJECT_ROOT / "data" / "raw"
DATA_DIR = RAW_DIR

PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"
SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"
BALANCED_DIR = PROJECT_ROOT / "outputs" / "balanced_train"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
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
