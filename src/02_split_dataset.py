# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *
import os
import shutil
from sklearn.model_selection import train_test_split

# PATHS
from pathlib import Path

BASE_DIR = PROJECT_ROOT

input_folder = PROCESSED_DIR
output_folder = SPLIT_DIR

categories = ["Corrected", "Normal", "Reversal"]


categories = ["Corrected", "Normal", "Reversal"]

RANDOM_STATE = 42


# ============================================================
# REMOVE OLD SPLIT
# ============================================================

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# ============================================================
# CREATE CLEAN OUTPUT FOLDERS
# ============================================================

for split in ["train", "val", "test"]:
    for category in categories:
        os.makedirs(
            os.path.join(output_folder, split, category),
            exist_ok=True
        )


# ============================================================
# SPLIT DATA: 70% TRAIN / 15% VAL / 15% TEST
# ============================================================

print("=" * 70)
print("STARTING DATA SPLIT")
print("70% TRAIN / 15% VALIDATION / 15% TEST")
print("=" * 70)

total_original = 0
total_train = 0
total_val = 0
total_test = 0


for category in categories:

    category_path = os.path.join(
        input_folder,
        category
    )

    image_files = [
        f for f in os.listdir(category_path)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".bmp")
        )
    ]

    print(f"\n{category}")
    print("-" * 50)
    print(f"Total images: {len(image_files):,}")

    total_original += len(image_files)

    # 70% Train / 30% Temporary
    train_files, temp_files = train_test_split(
        image_files,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # 15% Validation / 15% Test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"Train      : {len(train_files):,}")
    print(f"Validation : {len(val_files):,}")
    print(f"Test       : {len(test_files):,}")

    split_data = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in split_data.items():

        destination = os.path.join(
            output_folder,
            split_name,
            category
        )

        for filename in files:

            source = os.path.join(
                category_path,
                filename
            )

            target = os.path.join(
                destination,
                filename
            )

            shutil.copy2(
                source,
                target
            )

    total_train += len(train_files)
    total_val += len(val_files)
    total_test += len(test_files)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT SUMMARY")
print("=" * 70)

print(f"Total          : {total_original:,}")
print(f"Train          : {total_train:,}")
print(f"Validation     : {total_val:,}")
print(f"Test           : {total_test:,}")

print("-" * 70)

print(f"Train %        : {total_train / total_original * 100:.2f}%")
print(f"Validation %   : {total_val / total_original * 100:.2f}%")
print(f"Test %         : {total_test / total_original * 100:.2f}%")

print("-" * 70)

print(f"Output folder:\n{output_folder}")
