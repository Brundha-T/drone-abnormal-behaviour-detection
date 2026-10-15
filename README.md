# Drone-Assisted Detection of Abnormal Behaviour Against Women in Public Places

## 📌 Project Overview

DroneWatch AI is an AI-based surveillance system designed to identify potentially abnormal behaviour in public places using drone-assisted video monitoring and deep learning.

Unlike traditional CCTV systems, drones provide a mobile aerial perspective that can help monitor larger areas. The proposed system analyses sequences of video frames to identify movement patterns that may indicate abnormal or aggressive behaviour.

The project combines deep learning, spatio-temporal attention, geographical information, and evidence integrity mechanisms to support public safety.

## 🎯 Objectives

- Detect potentially abnormal behaviour from video sequences.
- Improve behaviour recognition by analysing movements across multiple frames.
- Extract important visual features using deep learning.
- Incorporate geographical information into anomaly assessment.
- Maintain the integrity of recorded detection evidence.

## ✨ Key Features

- **Drone-Assisted Surveillance:** Uses aerial video to monitor public spaces.
- **Video Sequence Analysis:** Processes sequences of 16 frames to analyse movements over time.
- **MobileNetV3 Feature Extraction:** Extracts important visual features from video frames.
- **LHO-Based Feature Selection:** Selects relevant features to reduce redundant information.
- **Spatio-Temporal Transformer:** Uses Multi-Head Self-Attention to analyse relationships between frames.
- **Geo-Spatial Attention:** Incorporates location information to adjust anomaly scores in designated risk zones.
- **Evidence Integrity:** Uses SHA-256 hashing and a simulated blockchain ledger to support the integrity of detection records.

## 🔄 System Workflow

1. **Video Input:** Obtain video footage captured using a drone.
2. **Frame Processing:** Sample the video into sequences of 16 frames.
3. **Image Resizing:** Resize frames to 112 × 112 pixels.
4. **Feature Extraction:** Use MobileNetV3 to extract spatial features.
5. **Feature Selection:** Apply the LHO-inspired selection method to identify relevant features.
6. **Behaviour Analysis:** Use the Spatio-Temporal Transformer to analyse motion patterns across frames.
7. **Geo-Spatial Analysis:** Incorporate GPS information to adjust anomaly scores based on geographical context.
8. **Anomaly Detection:** Identify potentially abnormal behaviour patterns.
9. **Evidence Integrity:** Generate SHA-256 hashes for detection evidence and record them in the simulated ledger.
10. **Alert Generation:** Present potential detections for further review.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Deep Learning | Video-based behaviour analysis |
| MobileNetV3 | Spatial feature extraction |
| LHO | Feature selection and optimization |
| Spatio-Temporal Transformer | Analysing movements across video frames |
| Multi-Head Self-Attention | Learning temporal relationships |
| GPS | Providing geographical information |
| SHA-256 | Generating evidence-integrity hashes |
| Simulated Blockchain | Recording detection hashes |

## 🏗️ System Architecture

The proposed architecture follows a multi-stage video analysis pipeline.

**Drone Video Input → Frame Preprocessing → MobileNetV3 Feature Extraction → LHO Feature Selection → Spatio-Temporal Transformer → Geo-Spatial Attention → Anomaly Detection → SHA-256 Hashing → Alert Generation**

This approach combines visual intelligence with geographical context to support the identification of potentially abnormal behaviour.

## 📊 Reported Results

According to the project report:

- **Validation Accuracy:** 93.4%
- **Evaluation Dataset:** 1,634 scenarios
- **Frame Sequence:** 16 frames
- **Frame Resolution:** 112 × 112 pixels

These figures are reported in the project document and depend on its experimental setup. They do not guarantee the same performance in real-world environments.

## 💡 Applications

- Drone-assisted public-space monitoring.
- AI-based video surveillance research.
- Abnormal activity detection.
- Public safety monitoring.
- Video-based behaviour recognition.

## ⚠️ Limitations

- Detection accuracy may vary with lighting, camera angles, and video quality.
- Similar movements can have different meanings depending on the situation.
- AI predictions require human verification.
- GPS-based risk assessment depends on the accuracy of location information.
- Hashing supports evidence-integrity checks but does not independently establish the authenticity of footage.

## 🔮 Future Enhancements

- Improve detection accuracy using more diverse video datasets.
- Reduce false-positive alerts.
- Optimize the model for real-time drone deployment.
- Improve performance under different lighting and camera angles.
- Develop more detailed detection reports.
- Strengthen privacy and security safeguards.

## Author

**Brundha T**

B.Tech Information Technology

## 📄 Publication

The journal acceptance letter is uploaded as a PDF for your reference.[Acceptance Letter.pdf](https://github.com/user-attachments/files/33478267/Acceptance.Letter.pdf)
