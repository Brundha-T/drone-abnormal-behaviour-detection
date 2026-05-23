import os
import numpy as np
import cv2
import tensorflow as tf

# ------------------------------------------------------------------ CONFIG
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model.keras")
TEST_VIDEO = r"Dataset\violent\cam1\104.mp4"

# ------------------------------------------------------------------ CUSTOM LAYER
@tf.keras.utils.register_keras_serializable(package="Custom")
class DataAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, training=None):
        return inputs
    def compute_output_shape(self, input_shape):
         return input_shape

def extract_frames(video_path, num_frames=16, img_height=112, img_width=112):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_width, img_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32) / 255.0

print(f"Loading Model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"DataAugmentationLayer": DataAugmentationLayer})

print(f"Testing Video: {TEST_VIDEO}")
frames = extract_frames(TEST_VIDEO)
if frames is not None:
    batch = np.expand_dims(frames, axis=0)
    prediction = model.predict(batch, verbose=0)
    print("="*60)
    print(f"RAW PREDICTION: {prediction[0][0]}")
    conf = prediction[0][0]
    if conf >= 0.5:
        print(f"RESULT: ABNORMAL ({conf*100:.2f}%)")
    else:
        print(f"RESULT: NORMAL ({(1-conf)*100:.2f}%)")
    print("="*60)
else:
    print("Failed to extract frames.")
