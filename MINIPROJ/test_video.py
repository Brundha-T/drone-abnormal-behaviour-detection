"""
=============================================================
 DRONE ASSISTED ABNORMAL DETECTION OF WOMEN IN PUBLIC PLACES
 Inference Script — Annotates a test video with predictions
=============================================================
Usage:
    python test_video.py --video "Dataset/violent/cam1/1.mp4"
    python test_video.py --video path/to/any/video.mp4
"""

import os
import sys
import io
import cv2
import argparse
import numpy as np
import tensorflow as tf

# Force UTF-8 output to avoid cp1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model", "best_model.keras")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
NUM_FRAMES  = 16
IMG_HEIGHT  = 112
IMG_WIDTH   = 112

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────── HELPERS ──────────────────────────────
def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract evenly-spaced frames for model input."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

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


def draw_overlay(frame, label, confidence, is_abnormal):
    """Draw a styled overlay banner on a video frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    banner_h = max(60, int(h * 0.12))
    if is_abnormal:
        color = (0, 0, 200)   # Red (BGR) for ABNORMAL
    else:
        color = (0, 150, 0)   # Green (BGR) for NORMAL

    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Main label text
    label_text  = f"  {'⚠ ABNORMAL BEHAVIOR DETECTED' if is_abnormal else '✓ NORMAL'}"
    conf_text   = f"  Confidence: {confidence*100:.1f}%"
    font        = cv2.FONT_HERSHEY_DUPLEX
    font_scale  = max(0.5, w / 1000)
    thickness   = max(1, int(w / 600))

    text_y_main = int(banner_h * 0.55)
    text_y_conf = int(banner_h * 0.90)

    cv2.putText(frame, label_text,
                (10, text_y_main), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, conf_text,
                (10, text_y_conf), font, font_scale * 0.75, (220, 220, 220), thickness, cv2.LINE_AA)

    # Confidence bar at bottom
    bar_h = max(8, int(h * 0.015))
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (50, 50, 50), -1)
    bar_width = int(w * confidence)
    cv2.rectangle(frame, (0, h - bar_h), (bar_width, h), color, -1)

    return frame


def annotate_video(video_path, model, output_path):
    """Run frame-by-frame prediction and write annotated video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"  Resolution : {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total}")

    # ── Get model prediction for the whole clip ──
    frames_input = extract_frames(video_path)
    if frames_input is None:
        print("[!] Could not extract frames — skipping prediction.")
        raw_conf     = 0.5
        is_abnormal  = False
    else:
        batch    = np.expand_dims(frames_input, axis=0)   # (1, T, H, W, 3)
        raw_conf = float(model.predict(batch, verbose=0)[0][0])
        # Adjustable threshold to reduce false positives
        THRESHOLD = 0.60
        
        # confidence toward the predicted class
        if raw_conf >= THRESHOLD:
            is_abnormal = True
            confidence  = raw_conf
            label       = "ABNORMAL"
        else:
            is_abnormal = False
            confidence  = 1.0 - raw_conf
            label       = "NORMAL"

    print(f"  Prediction : {label}  ({confidence*100:.1f}% confidence)")

    # ── Annotate every frame ──
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = draw_overlay(frame.copy(), label, confidence, is_abnormal)
        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  Frames written : {frame_idx}")


# ─────────────────────── MAIN ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Drone Abnormal Detection — Inference on a single video"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input .mp4 video file"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to trained model .h5 file (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    video_path = args.video
    model_path = args.model

    if not os.path.isfile(video_path):
        print(f"[✗] Video not found: {video_path}")
        return
    if not os.path.isfile(model_path):
        print(f"[✗] Model not found: {model_path}")
        print("    Please run train.py first to generate the model.")
        return

    print("=" * 60)
    print(" DRONE ASSISTED ABNORMAL DETECTION — INFERENCE")
    print("=" * 60)
    print(f"\n  Input video : {video_path}")
    print(f"  Model path  : {model_path}\n")

    # Load model
    print("[1] Loading model …")
    model = tf.keras.models.load_model(model_path)
    print("    Model loaded successfully.\n")

    # Build output path
    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"result_{video_name}.mp4")

    # Annotate
    print("[2] Annotating video …")
    annotate_video(video_path, model, output_path)

    print(f"\n[✓] Output saved → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
