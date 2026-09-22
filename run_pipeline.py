from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DYSLEXIA_STAGES = [
    ("Dyslexia - preprocessing", ROOT / "src" / "01_preprocess.py"),
    ("Dyslexia - split", ROOT / "src" / "02_split_dataset.py"),
    ("Dyslexia - training-only balancing", ROOT / "src" / "03_balance_train.py"),
    ("Dyslexia - CNN and 128-D features", ROOT / "src" / "04_train_cnn_extract_features.py"),
    ("Dyslexia - MLP", ROOT / "src" / "05_train_mlp.py"),
    ("Dyslexia - SVM", ROOT / "src" / "06_train_svm.py"),
    ("Dyslexia - Random Forest", ROOT / "src" / "07_train_random_forest.py"),
    ("Dyslexia - LightGBM", ROOT / "src" / "08_train_lightgbm.py"),
    ("Dyslexia - XGBoost", ROOT / "src" / "09_train_xgboost.py"),
    ("Dyslexia - stacking", ROOT / "src" / "10_stacking.py"),
]

ALL_STAGES = DYSLEXIA_STAGES + [
    ("Dysgraphia - proposed hybrid pipeline", ROOT / "src" / "dysgraphia_pipeline.py"),
]


def run(stages):
    for name, script in stages:
        print("\n" + "=" * 80)
        print(f"RUNNING: {name}")
        print("=" * 80)
        result = subprocess.run([PYTHON, str(script)], cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(f"Stage failed: {name} (exit code {result.returncode})")

    print("\nMain research pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main handwriting screening pipelines.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dyslexia", action="store_true", help="Run the Dyslexia pipeline only.")
    group.add_argument("--dysgraphia", action="store_true", help="Run the Dysgraphia pipeline only.")
    args = parser.parse_args()

    if args.dyslexia:
        run(DYSLEXIA_STAGES)
    elif args.dysgraphia:
        run([ALL_STAGES[-1]])
    else:
        run(ALL_STAGES)
