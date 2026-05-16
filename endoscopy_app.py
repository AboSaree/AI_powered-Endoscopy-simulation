"""
Endoscopy Real-Time Analysis App
=================================
Processes a video frame-by-frame:
  1. EfficientNetB0 classifier  →  Normal vs Esophagitis
  2. YOLOv8 detector            →  Locates esophagitis regions (only if step 1 flags it)

Annotated frames are saved to two separate output folders.
All settings are configurable through the Settings panel.

Dependencies:
    pip install tensorflow ultralytics opencv-python pillow numpy
"""

import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PATHS — edit these two lines before running
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL_PATH = r"D:\AI endoscopy\best_tl.keras"   # <-- CHANGE THIS
DETECTOR_MODEL_PATH   = r"D:\AI endoscopy\Yolo model.pt"                      # <-- CHANGE THIS


# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

tf   = None
YOLO = None


def _import_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf


def _import_yolo():
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Output folders
    "out_classification": "output/classification_frames",
    "out_detection":      "output/detection_frames",

    # Inference
    "clf_threshold": 0.3,    # sigmoid threshold: below → esophagitis
    "det_confidence": 0.25,   # YOLO confidence threshold
    "det_iou":        0.45,   # YOLO NMS IoU threshold
    "frame_skip":     5,      # process every Nth frame (1 = every frame)

    # Display
    "display_width":   800,
    "show_confidence": True,
    "box_thickness":   2,

    # Colors (BGR for OpenCV)
    "color_normal":      (0,  200,   0),   # green
    "color_esophagitis": (0,    0, 220),   # red
    "color_box":         (0,  165, 255),   # orange
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class Classifier:
    """
    Wraps the EfficientNetB0 Keras model.
    Class mapping: 0 = esophagitis, 1 = normal-z-line
    Preprocessing: tf.keras.applications.efficientnet.preprocess_input
    """
    IMG_SIZE = 224

    def __init__(self, model_path: str):
        _import_tf()
        self.model       = tf.keras.models.load_model(model_path)
        self._preprocess = tf.keras.applications.efficientnet.preprocess_input

    def predict(self, bgr_frame: np.ndarray, threshold: float = 0.5):
        """Returns (label, confidence). label is 'esophagitis' or 'normal'."""
        rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.IMG_SIZE, self.IMG_SIZE))
        inp     = self._preprocess(resized.astype("float32")[np.newaxis])
        prob    = float(self.model.predict(inp, verbose=0)[0][0])  # P(normal)
        if prob > threshold:
            return "normal", prob
        else:
            return "esophagitis", 1.0 - prob


class Detector:
    """Wraps the YOLOv8 detection model."""

    def __init__(self, model_path: str):
        _import_yolo()
        self.model = YOLO(model_path)

    def predict(self, bgr_frame: np.ndarray, conf: float, iou: float):
        """Returns list of dicts: {"box": (x1,y1,x2,y2), "conf": float, "cls": str}"""
        results    = self.model.predict(bgr_frame, conf=conf, iou=iou, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "box":  (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
                "cls":  results.names[int(box.cls[0])],
            })
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# FRAME ANNOTATOR
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame, clf_label, clf_conf, detections, settings):
    """Draw classification banner and YOLO boxes onto a copy of the frame."""
    out       = frame.copy()
    h, w      = out.shape[:2]
    show_conf = settings["show_confidence"]
    thickness = settings["box_thickness"]

    # Classification banner
    if clf_label == "esophagitis":
        color = settings["color_esophagitis"]
        text  = f"ESOPHAGITIS  {clf_conf:.0%}" if show_conf else "ESOPHAGITIS"
    else:
        color = settings["color_normal"]
        text  = f"NORMAL  {clf_conf:.0%}" if show_conf else "NORMAL"

    cv2.rectangle(out, (0, 0), (w, 40), color, -1)
    cv2.putText(out, text, (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # YOLO detection boxes
    box_color = settings["color_box"]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        label = f"{det['cls']} {det['conf']:.0%}" if show_conf else det["cls"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class EndoscopyApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Endoscopy Analysis App")
        self.root.resizable(True, True)

        self.settings     = DEFAULTS.copy()
        self.classifier   = None
        self.detector     = None

        self._video_path  = None
        self._cap         = None
        self._running     = False
        self._paused      = False
        self._thread      = None
        self._frame_count = 0
        self._saved_clf   = 0
        self._saved_det   = 0

        self._build_ui()
        # Auto-load models after UI is ready
        self.root.after(100, self._load_models_on_startup)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel — video + controls
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, bg="#1a1a2e", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        ctrl = ttk.Frame(left)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        ttk.Button(ctrl, text="📂 Open Video",  command=self._open_video).pack(side="left", padx=4)
        ttk.Button(ctrl, text="▶ Start",        command=self._start).pack(side="left", padx=4)
        ttk.Button(ctrl, text="⏸ Pause/Resume", command=self._pause).pack(side="left", padx=4)
        ttk.Button(ctrl, text="⏹ Stop",         command=self._stop).pack(side="left", padx=4)
        ttk.Button(ctrl, text="⚙ Settings",     command=self._open_settings).pack(side="right", padx=4)

        # Status bar
        self.status_var = tk.StringVar(value="Loading models…")
        ttk.Label(left, textvariable=self.status_var, relief="sunken",
                  anchor="w").grid(row=2, column=0, sticky="ew", pady=(4, 0))

        # Right panel — live stats + model status
        right = ttk.LabelFrame(self.root, text="Live Statistics", padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)

        stats = [
            ("Frame",          "_stat_frame"),
            ("Classification", "_stat_clf"),
            ("Confidence",     "_stat_conf"),
            ("Detections",     "_stat_dets"),
            ("Saved (clf)",    "_stat_saved_clf"),
            ("Saved (det)",    "_stat_saved_det"),
            ("Processing FPS", "_stat_fps"),
        ]
        for i, (label, attr) in enumerate(stats):
            ttk.Label(right, text=label + ":", anchor="w").grid(
                row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            ttk.Label(right, textvariable=var, anchor="w",
                      font=("Consolas", 10, "bold")).grid(
                row=i, column=1, sticky="w", padx=(8, 0))

        # Model status (read-only labels, no buttons)
        model_frame = ttk.LabelFrame(right, text="Models", padding=8)
        model_frame.grid(row=len(stats), column=0, columnspan=2,
                         sticky="ew", pady=(16, 0))

        ttk.Label(model_frame, text="Classifier:").grid(
            row=0, column=0, sticky="w")
        self.clf_status = ttk.Label(model_frame, text="Loading…",
                                    foreground="orange")
        self.clf_status.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(model_frame, text="Detector:").grid(
            row=1, column=0, sticky="w", pady=(4, 0))
        self.det_status = ttk.Label(model_frame, text="Loading…",
                                    foreground="orange")
        self.det_status.grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))

    # ── Auto model loading ────────────────────────────────────────────────────

    def _load_models_on_startup(self):
        threading.Thread(target=self._load_models_thread, daemon=True).start()

    def _load_models_thread(self):
        errors = []

        if not Path(CLASSIFIER_MODEL_PATH).exists():
            errors.append(f"Classifier not found:\n  {CLASSIFIER_MODEL_PATH}")
        if not Path(DETECTOR_MODEL_PATH).exists():
            errors.append(f"Detector not found:\n  {DETECTOR_MODEL_PATH}")

        if errors:
            msg = "\n\n".join(errors) + (
                "\n\nPlease update CLASSIFIER_MODEL_PATH and "
                "DETECTOR_MODEL_PATH at the top of the script."
            )
            self.root.after(0, lambda: [
                self.clf_status.config(text="✗ Not found", foreground="red"),
                self.det_status.config(text="✗ Not found", foreground="red"),
                self.status_var.set("Model paths not found — see script header."),
                messagebox.showerror("Model paths not found", msg),
            ])
            return

        # Load classifier
        self.root.after(0, lambda: self.status_var.set("Loading classifier…"))
        try:
            self.classifier = Classifier(CLASSIFIER_MODEL_PATH)
            self.root.after(0, lambda: self.clf_status.config(
                text="✓ Loaded", foreground="green"))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: [
                self.clf_status.config(text="✗ Failed", foreground="red"),
                messagebox.showerror("Classifier load error", err_msg),
            ])
            return

        # Load detector
        self.root.after(0, lambda: self.status_var.set("Loading detector…"))
        try:
            self.detector = Detector(DETECTOR_MODEL_PATH)
            self.root.after(0, lambda: self.det_status.config(
                text="✓ Loaded", foreground="green"))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: [
                self.det_status.config(text="✗ Failed", foreground="red"),
                messagebox.showerror("Detector load error", err_msg),
            ])
            return

        self.root.after(0, lambda: self.status_var.set(
            "Models loaded. Open a video and click Start."))

    # ── Video controls ────────────────────────────────────────────────────────

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select Endoscopy Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")]
        )
        if path:
            self._video_path = path
            self.status_var.set(f"Video loaded: {Path(path).name}")
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._show_frame(frame)

    def _start(self):
        if not self._video_path:
            messagebox.showwarning("No video", "Please open a video file first.")
            return
        if not self.classifier or not self.detector:
            messagebox.showwarning("Models not ready",
                                   "Models are still loading. Please wait.")
            return
        if self._running:
            return

        Path(self.settings["out_classification"]).mkdir(parents=True, exist_ok=True)
        Path(self.settings["out_detection"]).mkdir(parents=True, exist_ok=True)

        self._running     = True
        self._paused      = False
        self._frame_count = 0
        self._saved_clf   = 0
        self._saved_det   = 0

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _pause(self):
        if self._running:
            self._paused = not self._paused
            self.status_var.set("Paused" if self._paused else "Processing…")

    def _stop(self):
        self._running = False
        self._paused  = False
        if self._cap:
            self._cap.release()
        self.status_var.set(
            f"Stopped. Saved {self._saved_clf} classification frames, "
            f"{self._saved_det} detection frames."
        )

    # ── Processing loop ───────────────────────────────────────────────────────

    def _process_loop(self):
        s        = self.settings
        skip     = max(1, int(s["frame_skip"]))
        clf_dir  = Path(s["out_classification"])
        det_dir  = Path(s["out_detection"])
        clf_thr  = float(s["clf_threshold"])
        det_conf = float(s["det_confidence"])
        det_iou  = float(s["det_iou"])

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror(
                "Error", "Cannot open video file."))
            self._running = False
            return

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_idx      = 0

        self.root.after(0, lambda: self.status_var.set("Processing…"))

        while self._running:
            while self._paused and self._running:
                time.sleep(0.05)

            ret, frame = self._cap.read()
            if not ret:
                break

            raw_idx += 1

            if (raw_idx - 1) % skip != 0:
                continue

            self._frame_count += 1
            t0 = time.perf_counter()

            # 1. Classify
            try:
                clf_label, clf_conf = self.classifier.predict(frame, clf_thr)
            except Exception:
                clf_label, clf_conf = "error", 0.0

            # 2. Detect (only if esophagitis)
            detections = []
            if clf_label == "esophagitis":
                try:
                    detections = self.detector.predict(frame, det_conf, det_iou)
                except Exception:
                    pass

            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

            # 3. Annotate
            annotated = annotate_frame(frame, clf_label, clf_conf, detections, s)

            # 4. Save
            fname = f"frame_{raw_idx:06d}.jpg"
            cv2.imwrite(str(clf_dir / fname), annotated)
            self._saved_clf += 1

            if detections:
                cv2.imwrite(str(det_dir / fname), annotated)
                self._saved_det += 1

            # 5. Update UI
            ann_copy  = annotated.copy()
            n_dets    = len(detections)
            saved_clf = self._saved_clf
            saved_det = self._saved_det
            frame_num = self._frame_count

            def _update(ann=ann_copy, lbl=clf_label, conf=clf_conf,
                        nd=n_dets, sc=saved_clf, sd=saved_det,
                        fn=frame_num, f=fps, ri=raw_idx, tot=total_frames):
                self._show_frame(ann)
                self._stat_frame.set(f"{fn}  (raw {ri}/{tot})")
                self._stat_clf.set(lbl.upper())
                self._stat_conf.set(f"{conf:.1%}")
                self._stat_dets.set(str(nd))
                self._stat_saved_clf.set(str(sc))
                self._stat_saved_det.set(str(sd))
                self._stat_fps.set(f"{f:.1f}")

            self.root.after(0, _update)

        self._cap.release()
        self._running = False
        self.root.after(0, lambda: self.status_var.set(
            f"Done. Saved {self._saved_clf} classification frames, "
            f"{self._saved_det} detection frames."
        ))

    # ── Canvas display ────────────────────────────────────────────────────────

    def _show_frame(self, bgr_frame: np.ndarray):
        cw      = self.canvas.winfo_width()  or self.settings["display_width"]
        ch      = self.canvas.winfo_height() or int(cw * 9 / 16)
        h, w    = bgr_frame.shape[:2]
        scale   = min(cw / w, ch / h)
        nw, nh  = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr_frame, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas._img = img
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=img)

    # ── Settings dialog ───────────────────────────────────────────────────────

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.grab_set()
        win.resizable(False, False)

        s     = self.settings
        vars_ = {}

        fields = [
            ("── Inference ──────────────────────────", None,                 None),
            ("Classifier threshold (0–1)",              "clf_threshold",      "float"),
            ("YOLO confidence threshold (0–1)",         "det_confidence",     "float"),
            ("YOLO NMS IoU threshold (0–1)",            "det_iou",            "float"),
            ("Frame skip  (1 = every frame)",           "frame_skip",         "int"),
            ("── Output Folders ─────────────────────", None,                 None),
            ("Classification frames folder",            "out_classification", "str"),
            ("Detection frames folder",                 "out_detection",      "str"),
            ("── Display ────────────────────────────", None,                 None),
            ("Show confidence on frame",                "show_confidence",    "bool"),
            ("Bounding box thickness (px)",             "box_thickness",      "int"),
        ]

        row = 0
        for label, key, typ in fields:
            if key is None:
                ttk.Label(win, text=label, foreground="#555",
                          font=("TkDefaultFont", 9, "italic")).grid(
                    row=row, column=0, columnspan=2,
                    sticky="w", padx=12, pady=(10, 2))
                row += 1
                continue

            ttk.Label(win, text=label).grid(
                row=row, column=0, sticky="w", padx=12, pady=3)

            if typ == "bool":
                var = tk.BooleanVar(value=s[key])
                ttk.Checkbutton(win, variable=var).grid(
                    row=row, column=1, sticky="w", padx=8)
            else:
                var = tk.StringVar(value=str(s[key]))
                ttk.Entry(win, textvariable=var, width=32).grid(
                    row=row, column=1, sticky="ew", padx=8)

            vars_[key] = (var, typ)
            row += 1

        def _save():
            for key, (var, typ) in vars_.items():
                try:
                    if   typ == "float": s[key] = float(var.get())
                    elif typ == "int":   s[key] = int(var.get())
                    elif typ == "bool":  s[key] = bool(var.get())
                    else:                s[key] = var.get()
                except ValueError:
                    messagebox.showerror("Invalid value",
                                         f"Invalid value for '{key}'")
                    return
            win.destroy()
            self.status_var.set("Settings saved.")

        ttk.Button(win, text="Save", command=_save).grid(
            row=row, column=0, columnspan=2, pady=12)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x680")
    app = EndoscopyApp(root)
    root.mainloop()
