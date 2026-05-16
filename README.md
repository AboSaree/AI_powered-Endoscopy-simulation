# 🩺 AI-Powered Endoscopy Simulation

A real-time endoscopy video analysis application that uses deep learning to **classify** and **detect** esophagitis in endoscopic video frames. The system employs a two-stage pipeline: an EfficientNetB0 classifier first determines whether a frame is **Normal** or **Esophagitis**, and if esophagitis is detected, a YOLOv8 object detector localizes the affected regions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This project simulates an AI-assisted endoscopy workflow. A clinician loads an endoscopic video into the desktop application, and the system processes it frame-by-frame, providing:

1. **Classification** — Each frame is classified as *Normal* or *Esophagitis* using an EfficientNetB0 model fine-tuned via transfer learning.
2. **Detection** — When esophagitis is identified, a YOLOv8 model draws bounding boxes around the affected regions.

Annotated frames are saved to separate output folders for review.

---

## Features

- 🎥 **Real-time video processing** with frame-by-frame analysis
- 🧠 **Two-stage AI pipeline** — classification followed by conditional detection
- 🖥️ **Desktop GUI** built with Tkinter — no web server required
- 📊 **Live statistics panel** showing classification results, confidence, FPS, and detection counts
- ⚙️ **Configurable settings** — adjust thresholds, frame skip rate, output paths, and display options
- 💾 **Automatic frame saving** — classification and detection frames saved separately
- ⏯️ **Playback controls** — start, pause/resume, and stop processing at any time

---

## Project Structure

```
AI_powered-Endoscopy-simulation/
│
├── endoscopy_app.py              # Main application (GUI + inference pipeline)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── Models/                       # Pre-trained model weights
│   ├── best_tl.keras             #   EfficientNetB0 classifier (transfer learning)
│   ├── best_cnn.keras            #   Custom CNN classifier (alternative)
│   └── Yolo model.pt             #   YOLOv8m esophagitis detector
│
├── Notebooks/                    # Training scripts & notebooks
│   ├── esophagitis_cnn_classifier.ipynb   # Classifier training notebook
│   ├── esophagitis_yolov8_kaggle.ipynb    # YOLOv8 training notebook
│   └── Yolo_script.py                    # Standalone YOLOv8 training script
│
├── Sample Video/                 # Sample endoscopy video for testing
│   └── endoscope.mp4
│
├── Dataset/                      # Training datasets (git-ignored)
│
└── output/                       # Inference output (generated at runtime)
    ├── classification_frames/    #   All processed frames with classification labels
    └── detection_frames/         #   Frames where esophagitis regions were detected
```

---

## Datasets

### Classification — Kvasir v2

The classification model was trained on the [**Kvasir v2**](https://datasets.simula.no/kvasir/) dataset, a large-scale gastrointestinal (GI) tract image dataset. For this project, only the **esophagitis** and **normal-z-line** classes were used to create a binary classification task.

- **Source:** Simula Research Laboratory
- **Classes used:** `esophagitis`, `normal-z-line`
- **Image size:** Resized to 224×224 for EfficientNetB0 input

### Detection — Custom Roboflow Dataset

The detection model was trained on a **custom dataset** created manually using [**Roboflow**](https://roboflow.com/). Esophagitis regions were manually annotated with bounding boxes in YOLO format.

> ⚠️ **Note:** Due to the small size and manual nature of this custom dataset, the detection model's accuracy is limited. The annotations were done by hand, and the dataset does not have the scale or diversity of professionally curated medical imaging datasets. As a result, **detection results should be interpreted with caution** and are intended for demonstration and simulation purposes only.

- **Format:** YOLO (train/valid/test split)
- **Annotations:** Manual bounding boxes around esophagitis regions
- **Tool:** Roboflow

---

## Models

| Model | Architecture | Task | Input Size | Dataset |
|-------|-------------|------|-----------|---------|
| `best_tl.keras` | EfficientNetB0 (transfer learning) | Binary Classification | 224×224 | Kvasir v2 |
| `best_cnn.keras` | Custom CNN | Binary Classification | 224×224 | Kvasir v2 |
| `Yolo model.pt` | YOLOv8m | Object Detection | 640×640 | Custom (Roboflow) |

**Classification output:**
- Sigmoid probability > threshold → **Normal**
- Sigmoid probability ≤ threshold → **Esophagitis**

**Detection output:**
- Bounding boxes with confidence scores around detected esophagitis regions

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AboSaree/AI_powered-Endoscopy-simulation.git
   cd AI_powered-Endoscopy-simulation
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv endoscopy_env
   endoscopy_env\Scripts\activate    # Windows
   # source endoscopy_env/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure model paths:**

   Open `endoscopy_app.py` and update the model paths at the top of the file:

   ```python
   CLASSIFIER_MODEL_PATH = r"Models\best_tl.keras"
   DETECTOR_MODEL_PATH   = r"Models\Yolo model.pt"
   ```

---

## Usage

1. **Run the application:**

   ```bash
   python endoscopy_app.py
   ```

2. **Load a video:** Click **📂 Open Video** and select an endoscopic video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`).

3. **Start processing:** Click **▶ Start** to begin frame-by-frame analysis.

4. **Monitor results:** The right panel shows live statistics including classification label, confidence, detection count, and processing FPS.

5. **Review output:** Annotated frames are saved to:
   - `output/classification_frames/` — all processed frames
   - `output/detection_frames/` — only frames with detected esophagitis regions

---

## Configuration

Click **⚙ Settings** in the app to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| Classifier threshold | `0.3` | Sigmoid threshold — below this value, the frame is classified as esophagitis |
| YOLO confidence | `0.25` | Minimum confidence for YOLO detections |
| YOLO NMS IoU | `0.45` | Non-maximum suppression IoU threshold |
| Frame skip | `5` | Process every Nth frame (1 = every frame) |
| Show confidence | `True` | Display confidence scores on annotated frames |
| Box thickness | `2` | Bounding box line thickness in pixels |

---

## Known Limitations

- **Detection accuracy is limited** — The YOLOv8 detection model was trained on a small, manually annotated custom dataset from Roboflow. It may produce false positives or miss esophagitis regions. This model is intended for **simulation and demonstration purposes only**.
- **Not for clinical use** — This application is an educational/research prototype and is **not validated for clinical diagnosis**.
- **Hardware requirements** — Running both TensorFlow and PyTorch models simultaneously requires significant RAM. A CUDA-capable GPU is recommended for real-time processing.

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- [Kvasir v2 Dataset](https://datasets.simula.no/kvasir/) — Simula Research Laboratory
- [Roboflow](https://roboflow.com/) — Dataset annotation and management
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — Classification model training