# This file is a public GitHub-ready refactor of the corresponding stage
# from the original research pipeline. Scientific model settings are kept
# unchanged unless explicitly documented in README.md.

from config import *
import os
import shutil
import cv2

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# PATHS
# ============================================================

split_path = str(SPLIT_DIR)

balanced_train_path = str(BALANCED_DIR)

# ============================================================
# CLASSES
# ============================================================

categories = CLASSES

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = SEED

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# TARGET SIZE
# Largest training class in the TRAIN split
# ============================================================

# The target is deliberately derived from the current training split.
# This avoids hard-coding a dataset-specific count while preserving
# the manuscript methodology: balance every training class to the
# size of the largest training class.
train_counts = {}
for category in categories:
    category_dir = os.path.join(split_path, "train", category)
    train_counts[category] = sum(
        1
        for filename in os.listdir(category_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    )

if not train_counts:
    raise RuntimeError("No training classes were found in the split directory.")

target_size = max(train_counts.values())

print("Training counts before balancing:")
for category, count in train_counts.items():
    print(f"  {category}: {count:,}")
print(f"Dynamic target size (largest training class): {target_size:,}")

# ============================================================
# CLEAN OLD BALANCED DATASET
# ============================================================

if os.path.exists(balanced_train_path):
    shutil.rmtree(balanced_train_path)

os.makedirs(
    balanced_train_path,
    exist_ok=True
)

# ============================================================
# START
# ============================================================

print("=" * 70)
print("STARTING TRAINING-ONLY DATA BALANCING")
print("=" * 70)

print(f"Target images per class: {target_size}")
print("Augmentation is applied ONLY to the training set.")
print("Validation and Test sets remain completely untouched.")
print("=" * 70)


# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image(image):

    try:

        if image is None:
            return None

        h, w = image.shape[:2]

        # ----------------------------------------------------
        # 1. Random rotation: -20° to +20°
        # ----------------------------------------------------

        angle = random.uniform(-20, 20)

        center = (
            w // 2,
            h // 2
        )

        M_rotation = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            M_rotation,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # ----------------------------------------------------
        # 2. Gaussian noise
        # μ = 0, σ = 10
        # ----------------------------------------------------

        noise = np.random.normal(
            loc=0,
            scale=10,
            size=rotated.shape
        ).astype(np.float32)

        noisy = (
            rotated.astype(np.float32)
            + noise
        )

        noisy = np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)

        # ----------------------------------------------------
        # 3. CLAHE
        # ----------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            noisy
        )

        # ----------------------------------------------------
        # 4. Adaptive Gaussian Thresholding
        # ----------------------------------------------------

        augmented = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            2
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Horizontal flipping is NOT used.
        # ----------------------------------------------------

        return augmented

    except Exception as e:

        logging.warning(
            f"Augmentation error: {e}"
        )

        return None


# ============================================================
# COPY ORIGINAL TRAINING IMAGES
# ============================================================

def copy_original_image(args):

    src, dst = args

    try:

        shutil.copy2(
            src,
            dst
        )

        return True

    except Exception as e:

        logging.warning(
            f"Copy error: {src} -> {e}"
        )

        return False


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for category in categories:

    print("\n" + "=" * 70)
    print(f"PROCESSING CLASS: {category}")
    print("=" * 70)

    # IMPORTANT:
    # Use TRAIN only.
    source_category_path = os.path.join(
        split_path,
        "train",
        category
    )

    output_category_path = os.path.join(
        balanced_train_path,
        category
    )

    os.makedirs(
        output_category_path,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get training images
    # --------------------------------------------------------

    images = [
        f for f in os.listdir(source_category_path)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            )
        )
    ]

    random.shuffle(images)

    original_count = len(images)

    print(
        f"Original training images : {original_count:,}"
    )

    print(
        f"Target training images   : {target_size:,}"
    )

    # ========================================================
    # CASE 1:
    # CLASS ALREADY EQUAL TO TARGET
    # ========================================================

    if original_count == target_size:

        print("No augmentation required.")

        copy_tasks = []

        for filename in images:

            src = os.path.join(
                source_category_path,
                filename
            )

            dst = os.path.join(
                output_category_path,
                filename
            )

            copy_tasks.append(
                (src, dst)
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            list(
                tqdm(
                    executor.map(
                        copy_original_image,
                        copy_tasks
                    ),
                    total=len(copy_tasks),
                    desc=f"Copying {category}"
                )
            )

    # ========================================================
    # CASE 2:
    # CLASS IS SMALLER THAN TARGET
    # ========================================================

    elif original_count < target_size:

        required_augmented = (
            target_size - original_count
        )

        print(
            f"Augmenting {required_augmented:,} "
            f"additional images..."
        )

        # ----------------------------------------------------
        # First copy ALL original training images
        # ----------------------------------------------------

        copy_tasks = []

        for filename in images:

            src = os.path.join(
                source_category_path,
                filename
            )

            dst = os.path.join(
                output_category_path,
                filename
            )

            copy_tasks.append(
                (src, dst)
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            list(
                tqdm(
                    executor.map(
                        copy_original_image,
                        copy_tasks
                    ),
                    total=len(copy_tasks),
                    desc=f"Copying originals - {category}"
                )
            )

        # ----------------------------------------------------
        # Generate augmented images
        # ----------------------------------------------------

        generated = 0

        progress = tqdm(
            total=required_augmented,
            desc=f"Augmenting {category}"
        )

        while generated < required_augmented:

            # Select ONLY an ORIGINAL training image
            filename = random.choice(images)

            img_path = os.path.join(
                source_category_path,
                filename
            )

            image = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            augmented = augment_image(
                image
            )

            if augmented is None:
                continue

            new_filename = (
                f"aug_{generated + 1:05d}_{filename}"
            )

            output_path = os.path.join(
                output_category_path,
                new_filename
            )

            success = cv2.imwrite(
                output_path,
                augmented
            )

            if success:

                generated += 1
                progress.update(1)

        progress.close()

    # ========================================================
    # CASE 3:
    # CLASS IS LARGER THAN TARGET
    # ========================================================

    else:

        print(
            f"Class contains {original_count:,} images, "
            f"which is greater than target."
        )

        print(
            f"Randomly selecting {target_size:,} images."
        )

        selected_images = random.sample(
            images,
            target_size
        )

        copy_tasks = []

        for filename in selected_images:

            src = os.path.join(
                source_category_path,
                filename
            )

            dst = os.path.join(
                output_category_path,
                filename
            )

            copy_tasks.append(
                (src, dst)
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            list(
                tqdm(
                    executor.map(
                        copy_original_image,
                        copy_tasks
                    ),
                    total=len(copy_tasks),
                    desc=f"Selecting {category}"
                )
            )

    # ========================================================
    # FINAL COUNT FOR CLASS
    # ========================================================

    final_images = [
        f for f in os.listdir(output_category_path)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            )
        )
    ]

    final_count = len(final_images)

    print(
        f"Final training images: {final_count:,}"
    )

    if final_count == target_size:

        print(
            "✓ Class successfully balanced."
        )

    else:

        print(
            f"⚠ WARNING: Expected {target_size:,}, "
            f"but found {final_count:,}"
        )


# ============================================================
# FINAL DATASET AUDIT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL BALANCED TRAINING DATASET")
print("=" * 70)

total_final = 0

for category in categories:

    category_path = os.path.join(
        balanced_train_path,
        category
    )

    count = len([
        f for f in os.listdir(category_path)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            )
        )
    ])

    total_final += count

    print(
        f"{category:<15}: {count:,}"
    )

print("-" * 70)

print(
    f"{'TOTAL':<15}: {total_final:,}"
)

expected_total = (
    target_size * len(categories)
)

print(
    f"{'EXPECTED':<15}: {expected_total:,}"
)

print("-" * 70)

if total_final == expected_total:

    print(
        "✓ BALANCING COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "⚠ CHECK DATASET COUNTS"
    )

print("=" * 70)

print(
    "\nBalanced training dataset saved at:"
)

print(
    balanced_train_path
)

print("\nIMPORTANT:")
print("✓ Training images were balanced using augmentation.")
print("✓ Validation images were NOT augmented.")
print("✓ Test images were NOT augmented.")
print("✓ Horizontal flipping was NOT used.")
print("✓ Random seed = 42")
