## 👁️ Real‑Time People Counter (OpenCV)

Count people entering and exiting through a doorway using your webcam. The app detects people within a vertical band between two lines (Blue and Red) and counts when they cross from one side to the other. It’s designed to be robust in real scenes while staying simple to tune live.

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

## 🎯 Recommended starting values
- SENS (weight_thresh): 0.50
  - Raise (0.60–0.75) if you see boxes on walls/edges
  - Lower (0.35–0.45) if people are missed in dim scenes
- ROI pad (z/x): 5–12
  - Shrink to ignore passers‑by outside the doorway
- Midpoint margin (c/v): 8–12
  - Increase if people hover near the midpoint and cause flips

Advanced (edit in code if needed):
- min_box_area: 1800–3000 — reject tiny detections
- min_aspect, max_aspect: 0.25–0.8 — acceptable width/height ratio
- min_dx_for_count: 12–20 — required net x‑movement for a count
- trend_len: 3 — recent points to validate direction
- frame_skip: 1–2 — processing cadence (2 = faster, 1 = smoother)

## 🧭 Quick calibration flow
1. Place Blue and Red lines to tightly bracket the doorway (a/d, j/l).
2. Set SENS ~ 0.50. Raise if you see false positives, lower if missing people.
3. Shrink ROI pad (z) until boxes outside the doorway disappear.
4. If counts are jittery near the midpoint, increase midpoint margin (v) to ~12.
5. If needed, increase min_dx_for_count in code to 15–20.

## 🧩 Display
- IN (Blue→Red)
- OUT (Red→Blue)
- INSIDE (IN − OUT)
- SENS (current sensitivity/weight threshold)

## 🐞 Troubleshooting
- Camera not opening: ensure no other app uses the webcam; try a different index than 0.
- Too many false positives: raise SENS, shrink ROI pad, increase min_box_area.
- Missing people: lower SENS, widen ROI pad slightly, set frame_skip to 1.
- Double counts: increase midpoint margin; raise min_dx_for_count.
- Module warning about “code”: the file name `code.py` shadows the stdlib `code` module; this is harmless. Rename to `people_counter.py` if you want to silence it.

## 📦 Use cases
- Entrance/exit tracking for rooms, halls, stores
- Live occupancy monitoring
- Event attendance and capacity management

