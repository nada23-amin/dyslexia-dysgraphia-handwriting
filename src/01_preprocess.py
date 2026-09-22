# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *

from pathlib import Path


DATA_DIR = RAW_DIR

classes = ["Corrected", "Normal", "Reversal"]

print("=" * 60)
print("DYSLEXIA DATASET AUDIT")
print("=" * 60)

total_images = 0

for class_name in classes:
    class_dir = DATA_DIR / class_name

    images = list(class_dir.glob("*.png"))

    count = len(images)
    total_images += count

    print(f"{class_name:12s}: {count:,} images")

print("-" * 60)
print(f"{'TOTAL':12s}: {total_images:,} images")
print("=" * 60)
import os
import cv2
import numpy as np
import logging
import concurrent.futures
import hashlib
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# PATHS
# ============================================================
from pathlib import Path

BASE_DIR = PROJECT_ROOT

input_folder = RAW_DIR
output_folder = PROCESSED_DIR

os.makedirs(output_folder, exist_ok=True)

classes = ["Corrected", "Normal", "Reversal"]


# ============================================================
# COMPUTE IMAGE HASH
# ============================================================

def compute_image_hash(image_path):
    try:
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logging.warning(f"Error computing hash for {image_path}: {e}")
        return None


# ============================================================
# GLOBAL DUPLICATE REMOVAL
# ============================================================

def remove_global_duplicates():

    seen_hashes = {}
    duplicates = []

    print("=" * 60)
    print("GLOBAL DUPLICATE AUDIT")
    print("=" * 60)

    for category in classes:

        category_path = os.path.join(
            input_folder,
            category
        )

        image_files = [
            f for f in os.listdir(category_path)
            if f.lower().endswith(
                (".jpg", ".png", ".jpeg", ".tif", ".bmp")
            )
        ]

        for filename in tqdm(
            image_files,
            desc=f"Checking {category}"
        ):

            image_path = os.path.join(
                category_path,
                filename
            )

            image_hash = compute_image_hash(image_path)

            if image_hash is None:
                continue

            if image_hash in seen_hashes:

                duplicates.append(
                    (
                        image_path,
                        seen_hashes[image_hash]
                    )
                )

            else:

                seen_hashes[image_hash] = image_path

    # Delete duplicate copies
    for duplicate_path, original_path in duplicates:

        try:
            os.remove(duplicate_path)

        except Exception as e:

            logging.error(
                f"Error deleting {duplicate_path}: {e}"
            )

    return len(duplicates)


# ============================================================
# DESKEW
# ============================================================

def deskew(image):

    try:

        edges = cv2.Canny(
            image,
            50,
            200
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is not None:

            angles = [
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                ) * 180 / np.pi

                for line in lines

                for x1, y1, x2, y2 in [line[0]]
            ]

            median_angle = np.median(angles)

            if abs(median_angle) > 0.1:

                h, w = image.shape[:2]

                center = (
                    w // 2,
                    h // 2
                )

                M = cv2.getRotationMatrix2D(
                    center,
                    median_angle,
                    1.0
                )

                return cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

    except Exception as e:

        logging.warning(
            f"Error in deskew: {e}"
        )

    return image


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    img_path,
    output_path
):

    try:

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or image.size == 0:

            logging.warning(
                f"Skipping empty image: {img_path}"
            )

            return

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(image)

        # Denoising
        denoised = cv2.fastNlMeansDenoising(
            enhanced,
            h=30
        )

        # Adaptive Gaussian Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            2
        )

        # Deskew
        deskewed = deskew(
            adaptive_thresh
        )

        # Resize
        resized = cv2.resize(
            deskewed,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )

        # Save
        cv2.imwrite(
            output_path,
            resized
        )

    except Exception as e:

        logging.error(
            f"Error processing {img_path}: {e}"
        )


# ============================================================
# PROCESS CATEGORY
# ============================================================

def process_category(
    category
):

    category_path = os.path.join(
        input_folder,
        category
    )

    output_category_path = os.path.join(
        output_folder,
        category
    )

    os.makedirs(
        output_category_path,
        exist_ok=True
    )

    image_files = [
        f for f in os.listdir(category_path)
        if f.lower().endswith(
            (".jpg", ".png", ".jpeg", ".tif", ".bmp")
        )
    ]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = [
            executor.submit(
                process_image,
                os.path.join(
                    category_path,
                    filename
                ),
                os.path.join(
                    output_category_path,
                    filename
                )
            )
            for filename in image_files
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Processing {category}"
        ):
            future.result()


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("DYSLEXIA DATASET CLEANING AND PREPROCESSING")
print("=" * 60)

# 1. Global duplicate removal BEFORE splitting
remove_global_duplicates()

# 2. Preprocess all classes
for category in classes:

    process_category(category)

print("=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(
    f"Output saved in:\n{output_folder}"
)
