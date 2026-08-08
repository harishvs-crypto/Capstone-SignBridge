# Project Description: SignBridge (ISL-to-Speech Translation System)

**SignBridge** is an end-to-end, real-time Indian Sign Language (ISL) to speech translation system. It is designed to bridge the communication gap for deaf-mute individuals by converting hand gestures captured via a standard webcam into text and natural-sounding speech.

---

## 🎯 Project Objectives
1. **Real-Time Translation:** Capture, crop, and classify ISL alphabet signs (A–Z and Space) with minimal latency.
2. **Robust Hand Segmentation:** Isolate hand gestures from complex backgrounds, skin-like colors (such as the face), and varying room lightings.
3. **Text-to-Speech Synthesis:** Convert the compiled sign language words/sentences into audio outputs.
4. **Interactive Dashboard:** Provide a clean, user-friendly, responsive interface for real-time demonstrations.

---

## ⚙️ System Architecture

The project is structured into three primary pipelines:

```
[Webcam Video Stream] ──> [Hand Detector: MediaPipe Tasks]
                                      │
                                      ▼
                          [Geometric Convex Hull Masking]
                                      │
                                      ▼
                            [Black Background Crop]
                                      │
                                      ▼
                          [CNN Classifier: signbridge_best.h5]
                                      │
                                      ▼
                        [Stability Buffer & Text Accumulator]
                                      │
                                      ▼
                          [TTS Engine: Text-to-Speech]
```

### 1. High-Fidelity Hand Tracking
* **Engine:** MediaPipe Tasks API (`hand_landmarker.task`).
* **Capability:** Detects up to **2 hands** simultaneously. It maps 21 joint landmark coordinates per hand, completely ignoring the user's face and background clutter.
* **Fallback:** OpenCV skin-color and brightness trackers with real-time UI threshold sliders.

### 2. Geometric Background Masking (Core Innovation)
* **Problem Solved:** CNN models trained on clean datasets perform poorly when presented with colorful room backgrounds.
* **Mechanism:** The system computes the **Convex Hull (polygon)** of the 21 hand joints, dilates it slightly to cover the fingertips, and masks out everything else to solid black.
* **Result:** A clean silhouette of the hand shape on a black background, matching the exact format of the training dataset (81.5% black pixels) with 0% skin-tone or lighting dependence.

### 3. CNN Classification Model
* **Model Type:** Deep Convolutional Neural Network (CNN) built with TensorFlow/Keras.
* **Model Checkpoint:** `models/signbridge_best.h5` (optimal weights).
* **Classes:** 27 classes (A–Z and SPACE).
* **Input Resolution:** Resized to 64x64 pixels.

### 4. Translation & Speech Generation
* **Stability Buffer:** A hold-frame counter (e.g., 15 frames) prevents spelling errors. A letter is only appended to the text box if it is held stable.
* **Audio Synthesis:** Uses Python's Offline Text-to-Speech library (`pyttsx3`) for instant voice feedback.

---

## 💎 Key Features
* **Dual Hand Support:** Crops and classifies two-handed signs in a single, combined bounding box.
* **Zero-Calibrator MediaPipe Mode:** Uses the hand's geometry directly, making it immune to room lighting and skin tone changes.
* **Dynamic Fallback Calibrator:** Sliders are available on the sidebar to adjust skin redness/blueness or brightness if running outside the virtual environment.
* **Offline Operations:** The entire system—including hand landmarking, CNN prediction, and voice generation—runs fully offline on local CPU hardware.
