"""Dyslexia end-to-end CNN baseline from manuscript Section 4.8 / Table 19.

Uses the already-created image-level split: balanced training images under
outputs/balanced_train and untouched validation/test images under outputs/splits.
It does not overwrite the hybrid pipeline's models or saved features.
"""
from pathlib import Path
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, log_loss)
from config import PROJECT_ROOT, OUTPUT_ROOT, BALANCED_DIR, SPLIT_DIR, CLASS_NAMES, SEED, IMG_SIZE, BATCH_SIZE, EPOCHS

# Respect ABLATION_MODE so baseline outputs cannot overwrite the standard run or
# another ablation variant.
OUT = OUTPUT_ROOT / "results" / "dyslexia_end_to_end_cnn"
MODEL_DIR = OUTPUT_ROOT / "models" / "dyslexia_end_to_end_cnn"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )

def main():
    tf.keras.utils.set_random_seed(SEED)
    train_dir = BALANCED_DIR
    val_dir = SPLIT_DIR / "val"
    test_dir = SPLIT_DIR / "test"
    for path in (train_dir, val_dir, test_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required split directory missing: {path}. Run the preprocessing/split/balance stages first.")
    train_ds = dataset(str(train_dir), True).map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset(str(val_dir), False).map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset(str(test_dir), False).map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    y_true = np.concatenate([y.numpy() for _,y in test_ds])
    proba = model.predict(test_ds, verbose=0)
    pred = np.argmax(proba, axis=1)
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true,pred)),
        "macro_precision": float(precision_score(y_true,pred,average="macro",zero_division=0)),
        "macro_recall": float(recall_score(y_true,pred,average="macro",zero_division=0)),
        "macro_f1": float(f1_score(y_true,pred,average="macro",zero_division=0)),
        "macro_ovr_roc_auc": float(roc_auc_score(y_true,proba,multi_class="ovr",average="macro")),
        "log_loss": float(log_loss(y_true,proba,labels=list(range(len(CLASS_NAMES))))),
        "class_names": CLASS_NAMES,
        "epochs_completed": len(hist.history.get("loss",[])),
        "seed": SEED,
        "training_directory": str(train_dir),
        "validation_directory": str(val_dir),
        "test_directory": str(test_dir),
        "note": "This is a new execution; compare its metrics with the manuscript and do not assume historical values are reproduced.",
    }
    model.save(MODEL_DIR / "dyslexia_end_to_end_cnn.keras")
    np.save(OUT / "y_test_true.npy", y_true)
    np.save(OUT / "y_test_pred.npy", pred)
    np.save(OUT / "y_test_proba.npy", proba)
    np.savetxt(OUT / "confusion_matrix.csv", confusion_matrix(y_true,pred,labels=list(range(len(CLASS_NAMES)))), delimiter=",", fmt="%d")
    with open(OUT / "metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    with open(OUT / "classification_report.json","w",encoding="utf-8") as f: json.dump(classification_report(y_true,pred,target_names=CLASS_NAMES,output_dict=True,zero_division=0),f,indent=2)
    print(json.dumps(metrics,indent=2))
    print(f"Saved baseline outputs to {OUT}")

if __name__ == "__main__":
    main()
