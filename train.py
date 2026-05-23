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
EPOCHS        = 30  # Increased for better convergence on drone videos
LEARNING_RATE = 1e-4
RANDOM_SEED   = 42

# Drone video names (real drone footage -- treated as high-priority violent samples)
DRONE_VIDEO_NAMES = {"drone1", "drone2", "drone3", "drone4", "drone5"}
DRONE_SAMPLE_WEIGHT = 10.0  # Give drone videos 10x importance during training

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
    """Walk the Dataset folder and return (paths, labels, sample_weights)."""
    paths, labels, weights = [], [], []
    categories = {"violent": 1, "non-violent": 0}
    drone_found = []
    for cat_name, label in categories.items():
        cat_dir = os.path.join(DATASET_ROOT, cat_name)
        for cam in ["cam1", "cam2"]:
            cam_dir = os.path.join(cat_dir, cam)
            if not os.path.isdir(cam_dir):
                continue
            for fname in sorted(os.listdir(cam_dir)):
                # Accept BOTH .mp4 and .MP4 (drone videos use uppercase extension)
                if fname.lower().endswith(".mp4"):
                    fpath = os.path.join(cam_dir, fname)
                    stem  = os.path.splitext(fname)[0].lower()
                    is_drone = stem in DRONE_VIDEO_NAMES
                    # Drone videos are violent beating footage -- label = 1
                    effective_label = 1 if is_drone else label
                    paths.append(fpath)
                    labels.append(effective_label)
                    if is_drone:
                        weights.append(DRONE_SAMPLE_WEIGHT)
                        drone_found.append(fname)
                    else:
                        weights.append(1.0)
    print(f"\n[DRONE] Found {len(drone_found)} drone video(s): {drone_found}")
    return paths, labels, weights


# ------------------------------------------------------------------ TF DATASET
def load_video_tf(path, label, weight=1.0):
    frames = tf.numpy_function(
        lambda p: extract_frames(p.decode("utf-8")),
        [path], tf.float32
    )
    frames.set_shape([NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3])
    return frames, label, weight


def make_dataset(paths, labels, shuffle=True, sample_weights=None):
    if sample_weights is None:
        sample_weights = [1.0] * len(paths)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=RANDOM_SEED)
        
    ds = ds.map(load_video_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------------ MODEL
def build_model():
    """MobileNetV2 (fine-tuned) + Data Augmentation + LSTM classifier."""
    
    # Preprocessing and Augmentation Layers (Applied per-frame) 
    # Distorsion kept minimal to avoid confusing the small dataset
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ], name="data_augmentation")

    base_cnn = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # Freeze all layers first, then unfreeze the top 30 layers for fine-tuning
    # This lets the CNN adapt its high-level features to drone footage
    base_cnn.trainable = False
    for layer in base_cnn.layers[-30:]:
        layer.trainable = True
    print(f"[MODEL] MobileNetV2 fine-tuning: top 30 layers unfrozen out of {len(base_cnn.layers)} total")

    frame_input = layers.Input(
        shape=(NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3), name="video_input"
    )
    
    # Apply data augmentation to all frames in the sequence independently
    def augment_frames(frames):
        # frames shape: (batch, time, h, w, c)
        # Reshape to (batch * time, h, w, c) to apply augmentation consistently
        batch_size = tf.shape(frames)[0]
        time_steps = tf.shape(frames)[1]
        reshaped = tf.reshape(frames, [-1, IMG_HEIGHT, IMG_WIDTH, 3])
        augmented = data_augmentation(reshaped)
        return tf.reshape(augmented, [batch_size, time_steps, IMG_HEIGHT, IMG_WIDTH, 3])
        
    # Apply data augmentation to all frames in the sequence independently using a robust custom layer
    @tf.keras.utils.register_keras_serializable(package="Custom")
    class DataAugmentationLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.data_augmentation = data_augmentation

        def call(self, inputs, training=None):
            if training:
                batch_size = tf.shape(inputs)[0]
                time_steps = tf.shape(inputs)[1]
                reshaped = tf.reshape(inputs, [-1, IMG_HEIGHT, IMG_WIDTH, 3])
                augmented = self.data_augmentation(reshaped, training=True)
                return tf.reshape(augmented, [batch_size, time_steps, IMG_HEIGHT, IMG_WIDTH, 3])
            return inputs
            
        def compute_output_shape(self, input_shape):
            return input_shape

    x = DataAugmentationLayer(name="data_augmentation_layer")(frame_input)

    x = layers.TimeDistributed(base_cnn, name="cnn_features")(x)
    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(x) # Simplified for better generalization
    x = layers.Dropout(0.3)(x) 
    x = layers.LSTM(64, name="lstm_2")(x) # Simplified
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = Model(inputs=frame_input, outputs=output)
    
    # Initial learning rate set in config (2e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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
    paths, labels, sample_weights = build_dataset()

    n_violent     = sum(labels)
    n_normal      = len(labels) - n_violent
    n_drone       = sum(1 for p in paths if os.path.splitext(os.path.basename(p))[0].lower() in DRONE_VIDEO_NAMES)
    print(f"    Total videos  : {len(paths)}")
    print(f"    Violent       : {n_violent}  (includes {n_drone} real drone videos)")
    print(f"    Non-violent   : {n_normal}")

    # 2. Train/val split -- keep drone videos in training set always
    from sklearn.model_selection import train_test_split

    drone_indices     = [i for i, p in enumerate(paths)
                         if os.path.splitext(os.path.basename(p))[0].lower() in DRONE_VIDEO_NAMES]
    other_indices     = [i for i in range(len(paths)) if i not in set(drone_indices)]

    other_labels      = [labels[i] for i in other_indices]
    other_train_idx, other_val_idx = train_test_split(
        other_indices, test_size=0.20,
        stratify=other_labels, random_state=RANDOM_SEED
    )

    # Always keep drone videos in training -- never in validation
    train_idx = other_train_idx + drone_indices
    val_idx   = other_val_idx

    X_train = [paths[i]          for i in train_idx]
    y_train = [labels[i]         for i in train_idx]
    w_train = [sample_weights[i] for i in train_idx]
    X_val   = [paths[i]          for i in val_idx]
    y_val   = [labels[i]         for i in val_idx]

    print(f"\n[2] Split -- Train: {len(X_train)}  |  Val: {len(X_val)}")
    print(f"    Drone videos guaranteed in training set: {sum(1 for p in X_train if os.path.splitext(os.path.basename(p))[0].lower() in DRONE_VIDEO_NAMES)}")

    # 3. Class weights (handle imbalance)
    total   = len(y_train)
    w0      = total / (2.0 * max(1, total - sum(y_train)))   # non-violent
    w1      = total / (2.0 * max(1, sum(y_train)))           # violent
    class_weights = {0: w0, 1: w1}
    print(f"\n[3] Class weights -- non-violent: {w0:.2f}, violent: {w1:.2f}")

    # 4. Build tf.data pipelines (with per-sample weights for drone videos)
    print("\n[4] Building data pipelines ...")
    train_ds = make_dataset(X_train, y_train, shuffle=True, sample_weights=w_train)
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
        EarlyStopping(monitor="val_loss", patience=10, # Longer patience for tiny dataset
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, verbose=1),
    ]

    # 7. Train
    print("\n[6] Training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
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
