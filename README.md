## 👁️ Real‑Time People Counter (OpenCV)

Count people entering and exiting through a doorway using your webcam. The app detects people within a vertical band between two lines (Blue and Red) and counts when they cross from one side to the other. It’s designed to be robust in real scenes while staying simple to tune live.

## 🎥 Demo Video  
![Watch the demo video](/Output.gif)

## 🚀 Features
- Real‑time webcam processing (resized to 600px wide)
- HOG people detector + CLAHE pre‑processing + Non‑Max Suppression
- Doorway ROI detection (only detects inside the band between the two lines)
- Color‑aware crossing logic with debouncing
  - Blue → Red = IN (+1)
  - Red → Blue = OUT (−1)
- Directional trend check to filter jitter (requires small net movement)
- Lightweight tracking with dynamic association threshold
- Live on‑screen overlay: IN, OUT, INSIDE, SENS
- Live calibration hotkeys for sensitivity, lines, ROI band, and midpoint tolerance

## 📊 How it works
1. Preprocess each frame using CLAHE on the Y channel to stabilize lighting.
2. Run HOG people detection only within the vertical ROI band between the Blue and Red lines (reduces background false positives).
3. Filter tiny/odd‑shaped boxes, then apply non‑max suppression.
4. Track detections frame‑to‑frame with a simple nearest‑neighbor match.
5. Count when a tracked person crosses the midpoint between lines in a valid direction:
   - Blue→Red is IN; Red→Blue is OUT.
   - The object must have “visited” its starting color side first, and must show a small net movement across the midpoint (trend), to avoid jitter counts.

INSIDE is computed as IN − OUT.

## 🛠️ Requirements
Install dependencies:

```bash
pip install opencv-python imutils numpy
```

## ▶️ Run

```bash
python code.py
```

Position the camera so the doorway is visible. Use the controls below to place the lines and tune accuracy.

## 🎮 Controls
- q: Quit
- r: Reset counters
- d: Toggle debug logs (prints IN/OUT events)

Calibration and tuning:
- a / d: Move Blue line left / right
- j / l: Move Red line left / right
- [ / ]: Decrease / increase sensitivity (SENS = HOG weight threshold)
- z / x: Narrow / widen the ROI pad (vertical band width around the lines)
- c / v: Decrease / increase midpoint tolerance (reduces jitter around midpoint)


## 🧩 Display
- IN (Blue→Red)
- OUT (Red→Blue)
- INSIDE (IN − OUT)
- SENS (current sensitivity/weight threshold)

## 📦 Use cases
- Entrance/exit tracking for rooms, halls, stores
- Live occupancy monitoring
- Event attendance and capacity management