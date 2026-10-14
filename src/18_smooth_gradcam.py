"""Generate Smooth Grad-CAM overlays for representative test images.

The script uses the saved CNN feature-extractor/classifier models and held-out
samples. It averages Grad-CAM heatmaps over small Gaussian input perturbations.
These visualizations are qualitative and are not clinical evidence.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf
import matplotlib.pyplot as plt
from config import OUTPUT_ROOT, SPLIT_DIR, CLASS_NAMES, SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = OUTPUT_ROOT / "figures" / "smooth_gradcam"
OUT.mkdir(parents=True, exist_ok=True)
N_SMOOTH = 20
NOISE_STD = 0.03


def last_conv_layer(model):
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not convs:
        raise ValueError(f"No Conv2D layer found in model: {model.name}")
    return convs[-1]


def gradcam_heatmap(model, batch, class_index, conv_layer):
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(batch, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are disconnected from the selected convolutional layer.")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    max_value = tf.reduce_max(cam)
    cam = tf.where(max_value > 0, cam / (max_value + 1e-8), tf.zeros_like(cam))
    return cam.numpy()


def smooth_heatmap(model, image, class_index, conv_layer, seed):
    rng = np.random.default_rng(seed)
    maps = []
    for _ in range(N_SMOOTH):
        noisy = np.clip(image + rng.normal(0, NOISE_STD, size=image.shape), 0.0, 1.0).astype(np.float32)
        maps.append(gradcam_heatmap(model, noisy[None, ...], int(class_index), conv_layer))
    result = np.mean(maps, axis=0)
    result = cv2.resize(result, (128, 128), interpolation=cv2.INTER_CUBIC)
    result = np.maximum(result, 0)
    if result.max() > 0:
        result /= result.max()
    return result


def save_panel(task, class_name, image_128, heatmap, true_label, predicted_label, probability):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].imshow(image_128, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Input\nTrue: {class_name}")
    axes[1].imshow(image_128, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    axes[1].set_title(f"Smooth Grad-CAM\nPredicted: {predicted_label} ({probability:.2f})")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{task.title()} — {class_name}", fontsize=10)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in class_name).strip("_").lower()
    path = OUT / f"{task}_smooth_gradcam_{safe}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def dyslexia_samples():
    model_path = OUTPUT_ROOT / "models" / "cnn" / "dyslexia_cnn.keras"
    if not model_path.exists():
        print(f"SKIP dyslexia Grad-CAM: run the main Dyslexia pipeline first ({model_path} missing).")
        return
    model = tf.keras.models.load_model(model_path)
    conv = last_conv_layer(model)
    test_root = SPLIT_DIR / "test"
    records = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = test_root / class_name
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not files: raise FileNotFoundError(f"No test images found for class {class_name}: {class_dir}")
        selected = files[0]
        raw = cv2.imread(str(selected), cv2.IMREAD_GRAYSCALE)
        if raw is None: raise ValueError(f"Could not read {selected}")
        img128 = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA)
        img28 = cv2.resize(img128, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        batch = img28[None, ..., None]
        proba = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(proba))
        heat = smooth_heatmap(model, batch[0], class_idx, conv, SEED + class_idx)
        out = save_panel("dyslexia", class_name, img128.astype(np.float32)/255.0, heat, class_idx, CLASS_NAMES[pred], float(proba[pred]))
        records.append({"task":"dyslexia", "image":str(selected), "true_class":class_name, "predicted_class":CLASS_NAMES[pred], "predicted_probability":float(proba[pred]), "target_class_for_heatmap":class_name, "output":str(out), "smoothing_samples":N_SMOOTH, "noise_std":NOISE_STD})
    return records


def dysgraphia_samples():
    model_path = OUTPUT_ROOT / "models" / "dysgraphia_feature_extractor_cnn.keras"
    split_dir = OUTPUT_ROOT / "splits" / "dysgraphia"
    required = [model_path, split_dir / "X_test_cnn.npy", split_dir / "y_test.npy", split_dir / "X_test_preprocessed_128.npy"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("SKIP dysgraphia Grad-CAM: run the main Dysgraphia pipeline first. Missing: " + ", ".join(missing))
        return
    model = tf.keras.models.load_model(model_path)
    conv = last_conv_layer(model)
    X = np.load(split_dir / "X_test_cnn.npy")
    y = np.load(split_dir / "y_test.npy").astype(int)
    originals = np.load(split_dir / "X_test_preprocessed_128.npy")
    names = ["Low Potential Dysgraphia", "Potential Dysgraphia"]
    records = []
    for class_idx, class_name in enumerate(names):
        indices = np.flatnonzero(y == class_idx)
        if len(indices) == 0:
            print(f"SKIP dysgraphia class {class_name}: no test example in this split.")
            continue
        i = int(indices[0])
        batch = X[i:i+1]
        proba = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(proba))
        heat = smooth_heatmap(model, batch[0], class_idx, conv, SEED + 10 + class_idx)
        out = save_panel("dysgraphia", class_name, originals[i].astype(np.float32)/255.0, heat, class_idx, names[pred], float(proba[pred]))
        records.append({"task":"dysgraphia", "test_index":i, "true_class":class_name, "predicted_class":names[pred], "predicted_probability":float(proba[pred]), "target_class_for_heatmap":class_name, "output":str(out), "smoothing_samples":N_SMOOTH, "noise_std":NOISE_STD})
    return records


def main():
    tf.keras.utils.set_random_seed(SEED)
    all_records = []
    for fn in (dyslexia_samples, dysgraphia_samples):
        records = fn()
        if records: all_records.extend(records)
    manifest = {"method":"Smooth Grad-CAM: average of Grad-CAM maps over Gaussian input perturbations", "n_smoothing_samples":N_SMOOTH, "noise_std_normalized_input":NOISE_STD, "seed":SEED, "interpretation":"qualitative visualization only; not clinical evidence", "outputs":all_records}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
