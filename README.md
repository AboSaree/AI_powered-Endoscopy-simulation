# 🩺 AI-Powered Endoscopy Simulation

A real-time endoscopy video analysis application that uses deep learning to **classify** and **detect** esophagitis in endoscopic video frames. The system employs a two-stage pipeline: an EfficientNetB0 classifier first determines whether a frame is **Normal** or **Esophagitis**, and if esophagitis is detected, a YOLOv8 object detector localizes the affected regions.

---

## 🎬 Demo

https://github.com/AboSaree/AI_powered-Endoscopy-simulation/blob/main/Demo/DEMO.mp4


---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Results & Model Comparison](#results--model-comparison)
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
├── Results/                      # Training results, plots & comparison figures
│   ├── Custom CNN vs Efficient Net.png
│   ├── AUC Graph for Custom CNN.png
│   ├── AUC Graph for Efficient Net.png
│   ├── Custom CNN Classification Report .png
│   ├── Custom CNN Confusion matrix .png
│   ├── Efficient Net Classification Report.png
│   ├── Efficient Net Confusion matrix .png
│   ├── Custom CNN Grad-CAM.png
│   ├── Plots for Yolo model.png
│   └── Confussion matrix for Yolo model .png
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

## Results & Model Comparison

Two classification architectures were trained and evaluated on the Kvasir v2 dataset: a **Custom CNN** built from scratch and an **EfficientNetB0** fine-tuned via transfer learning. Both were tested on 400 images (200 esophagitis, 200 normal-z-line).

### Overall Comparison

![Custom CNN vs EfficientNetB0 — Accuracy and AUC comparison](Results/Custom%20CNN%20vs%20Efficient%20Net.png)

| Metric | Custom CNN | EfficientNetB0 (fine-tuned) |
|--------|-----------|----------------------------|
| **Accuracy** | 78.75% | 77.75% |
| **ROC-AUC** | 0.863 | **0.874** |
| **Macro F1** | 0.79 | 0.78 |

Although the Custom CNN achieved a slightly higher accuracy (78.75% vs 77.75%), the **EfficientNetB0 model was selected** as the final classifier because it achieved a **higher AUC score (0.874 vs 0.863)**. AUC is a more robust metric for medical imaging tasks as it measures the model's ability to discriminate between classes across all thresholds, making it less sensitive to class imbalance and threshold selection.

---

### Custom CNN Results

<details>
<summary>📊 Click to expand Custom CNN details</summary>

#### Classification Report

![Custom CNN Classification Report](Results/Custom%20CNN%20Classification%20Report%20.png)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Esophagitis | 0.86 | 0.69 | 0.76 | 200 |
| Normal-z-line | 0.74 | 0.89 | 0.81 | 200 |
| **Accuracy** | | | **0.79** | **400** |

#### Confusion Matrix

![Custom CNN Confusion Matrix](Results/Custom%20CNN%20Confusion%20matrix%20.png)

- Correctly classified **138** esophagitis and **177** normal images
- Misclassified 62 esophagitis as normal (false negatives)
- Misclassified 23 normal as esophagitis (false positives)

#### ROC-AUC Curve

![Custom CNN AUC Curve — AUC = 0.863](Results/AUC%20Graph%20for%20Custom%20CNN.png)

</details>

---

### EfficientNetB0 Results (Selected Model ✅)

<details>
<summary>📊 Click to expand EfficientNetB0 details</summary>

#### Classification Report

![EfficientNetB0 Classification Report](Results/Efficient%20Net%20Classification%20Report.png)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Esophagitis | 0.84 | 0.69 | 0.75 | 200 |
| Normal-z-line | 0.73 | 0.87 | 0.80 | 200 |
| **Accuracy** | | | **0.78** | **400** |

#### Confusion Matrix

![EfficientNetB0 Confusion Matrix](Results/Efficient%20Net%20Confusion%20matrix%20.png)

- Correctly classified **137** esophagitis and **174** normal images
- Misclassified 63 esophagitis as normal (false negatives)
- Misclassified 26 normal as esophagitis (false positives)

#### ROC-AUC Curve

![EfficientNetB0 AUC Curve — AUC = 0.874](Results/AUC%20Graph%20for%20Efficient%20Net.png)

</details>

---

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) was applied to the Custom CNN to visualize which regions of the image the model focuses on when making predictions. The heatmaps confirm that the model attends to clinically relevant areas within the endoscopic images.

![Grad-CAM heatmap visualization for esophagitis and normal-z-line samples](Results/Custom%20CNN%20Grad-CAM.png)

---

## YOLOv8 Detection Results

### ⚠️ Dataset Limitation — Why Accuracy Is Low

One of the core challenges of this project was the **near-complete absence of publicly available annotated datasets** for esophagitis *detection* (bounding box localisation). While Kvasir v2 provides classification-level labels, it does not include bounding box annotations suitable for object detection training.

To overcome this, the detection dataset was **created entirely from scratch by manually annotating endoscopy images using [Roboflow](https://roboflow.com/)**. Every bounding box was drawn by hand, which introduced several unavoidable constraints:

- 📦 **Very small dataset** — the number of annotated images was far below what is typically required to train a reliable object detector.
- 🖊️ **Manual annotations** — bounding boxes were drawn by a non-expert, introducing inconsistencies in label quality.
- 🔁 **Limited diversity** — the images came from a narrow visual distribution, reducing the model's ability to generalise.
- 🚫 **No public benchmark** — there is no established detection dataset for this specific task to compare against or augment from.

As a direct consequence, the YOLOv8m model trained on this dataset shows **significantly lower performance** than the classifier, and its detections should be treated as **approximate and indicative only**.

---

### Training Curves

The plots below show training and validation losses alongside precision, recall, mAP50, and mAP50-95 across ~60 epochs. While losses decrease steadily, the detection metrics plateau at low values — a direct consequence of the small, manually annotated dataset.

![YOLOv8 training curves — loss, precision, recall, mAP50, mAP50-95](Results/Plots%20for%20Yolo%20model.png)

| Metric | Approximate Value |
|--------|------------------|
| Precision | ~0.40 |
| Recall | ~0.35 |
| **mAP50** | **~0.30** |
| **mAP50-95** | **~0.11** |

These values are substantially below production-level thresholds (typically mAP50 > 0.6 for reliable detection), which is expected given the dataset constraints.

---

### Confusion Matrix

![YOLOv8 confusion matrix — esophagitis vs background](Results/Confussion%20matrix%20for%20Yolo%20model%20.png)

The confusion matrix reveals the core weakness of the model:

- **96** esophagitis instances were correctly detected
- **74** esophagitis instances were **missed** (false negatives — predicted as background)
- **240** background regions were correctly rejected

A false-negative rate of ~44% (74 out of 170) is a direct reflection of the small and inconsistently annotated training set. The model has learned a general shape of the lesion but struggles with boundary cases due to the lack of training diversity.

> ⚠️ **Important:** These detection results are not suitable for clinical use. The model is included in this project as a **proof-of-concept simulation** of what a full detection pipeline would look like, pending a properly curated and annotated dataset.

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
