"""
=============================================================
 DRONE ASSISTED ABNORMAL DETECTION OF WOMEN IN PUBLIC PLACES
 Training Script -- MobileNetV2 + LSTM
=============================================================
"""

import os
import sys
import io
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Force UTF-8 output to avoid cp1252 UnicodeEncodeError from Keras progress bars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ------------------------------------------------------------------ CONFIG
DATASET_ROOT  = os.path.join(os.path.dirname(__file__), "Dataset")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "model")
NUM_FRAMES    = 16
IMG_HEIGHT    = 112
IMG_WIDTH     = 112
BATCH_SIZE    = 4
EPOCHS        = 30
LEARNING_RATE = 1e-4
RANDOM_SEED   = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------ HELPERS
def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract num_frames evenly-spaced frames from a video."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((num_frames, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32) / 255.0   # (T, H, W, 3)


def build_dataset():
    """Walk the Dataset folder and return (paths, labels)."""
    paths, labels = [], []
    categories = {"violent": 1, "non-violent": 0}
    for cat_name, label in categories.items():
        cat_dir = os.path.join(DATASET_ROOT, cat_name)
        for cam in ["cam1", "cam2"]:
            cam_dir = os.path.join(cat_dir, cam)
            if not os.path.isdir(cam_dir):
                continue
            for fname in sorted(os.listdir(cam_dir)):
                if fname.lower().endswith(".mp4"):
                    paths.append(os.path.join(cam_dir, fname))
                    labels.append(label)
    return paths, labels


# ------------------------------------------------------------------ TF DATASET
def load_video_tf(path, label):
    frames = tf.numpy_function(
        lambda p: extract_frames(p.decode("utf-8")),
        [path], tf.float32
    )
    frames.set_shape([NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3])
    return frames, label


def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=RANDOM_SEED)
    ds = ds.map(load_video_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------------ MODEL
def build_model():
    """MobileNetV2 (frozen) feature extractor + 2-layer LSTM classifier."""
    base_cnn = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base_cnn.trainable = False

    frame_input = layers.Input(
        shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3), name="video_input"
    )
    x = layers.TimeDistributed(base_cnn, name="cnn_features")(frame_input)
    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(64, name="lstm_2")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = Model(inputs=frame_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------------------------ PLOTTING
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Drone Abnormal Detection -- Training History", fontsize=14)
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[OK] Training history saved -> {save_path}")


# ------------------------------------------------------------------ MAIN
def main():
    print("=" * 60)
    print(" DRONE ASSISTED ABNORMAL DETECTION -- TRAINING")
    print("=" * 60)

    # 1. Dataset
    print("\n[1] Scanning dataset ...")
    paths, labels = build_dataset()
    n_violent     = sum(labels)
    n_normal      = len(labels) - n_violent
    print(f"    Total videos  : {len(paths)}")
    print(f"    Violent       : {n_violent}")
    print(f"    Non-violent   : {n_normal}")

    # 2. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=0.20,
        stratify=labels, random_state=RANDOM_SEED
    )
    print(f"\n[2] Split -- Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 3. Class weights (handle imbalance)
    total   = len(y_train)
    w0      = total / (2.0 * (total - sum(y_train)))   # non-violent
    w1      = total / (2.0 * sum(y_train))              # violent
    class_weights = {0: w0, 1: w1}
    print(f"\n[3] Class weights -- non-violent: {w0:.2f}, violent: {w1:.2f}")

    # 4. Build tf.data pipelines
    print("\n[4] Building data pipelines ...")
    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   shuffle=False)

    # 5. Build model (MobileNetV2 weights download happens here)
    print("\n[5] Building model (downloading ImageNet weights if needed) ...")
    model = build_model()
    model.summary()

    # 6. Callbacks
    ckpt_path = os.path.join(MODEL_DIR, "best_model.keras")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, verbose=1),
    ]

    # 7. Train
    print("\n[6] Training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # 8. Save & plot
    final_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_path)
    print(f"[OK] Final model saved -> {final_path}")
    plot_history(history)

    # 9. Evaluate
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n{'='*60}")
    print(f"  Validation Accuracy : {val_acc*100:.2f}%")
    print(f"  Validation Loss     : {val_loss:.4f}")
    print(f"  Best model saved to : {ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
