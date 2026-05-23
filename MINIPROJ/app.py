"""
=============================================================
 DRONE ASSISTED ABNORMAL DETECTION OF WOMEN IN PUBLIC PLACES
 Web Dashboard — Flask App
=============================================================
Run:
    python app.py
Then open:  http://localhost:5000
"""

import os
import uuid
import subprocess
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

app = Flask(__name__)

BASE_DIR   = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload an MP4/AVI/MOV/MKV file."}), 400

    # Save upload
    uid       = uuid.uuid4().hex[:8]
    filename  = f"{uid}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # Run inference subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "test_video.py"),
         "--video", save_path, "--model", MODEL_PATH],
        capture_output=True, text=True
    )

    stdout = result.stdout
    stderr = result.stderr

    # Parse result from stdout
    label      = "UNKNOWN"
    confidence = 0.0
    for line in stdout.splitlines():
        if "Prediction" in line and ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                pred_part = parts[1].strip()
                tokens    = pred_part.split()
                if tokens:
                    label = tokens[0].strip()
                # Extract confidence value
                if "(" in pred_part and "%" in pred_part:
                    try:
                        conf_str   = pred_part.split("(")[1].split("%")[0].strip()
                        confidence = float(conf_str)
                    except Exception:
                        pass

    # Find generated output video
    stem       = os.path.splitext(filename)[0]
    out_name   = f"result_{stem}.mp4"
    out_path   = os.path.join(OUTPUT_DIR, out_name)
    video_url  = url_for("serve_output", filename=out_name) if os.path.isfile(out_path) else None

    if result.returncode != 0:
        return jsonify({"error": f"Inference failed.\n{stderr}"}), 500

    return jsonify({
        "label":      label,
        "confidence": round(confidence, 1),
        "video_url":  video_url,
        "output":     stdout
    })


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
