# Fiducial Markers Playground (UMI-style Gripper Width)

This repo is a **hands-on sandbox** to prototype how **fiducial markers (ArUco)** can recover **continuous gripper aperture (width)** from video — the same “missing DOF” that **UMI OG** adds on top of SLAM-based 6DoF pose tracking.

Even without a 3D-printed gripper, you can start today by:
- generating ArUco markers,
- showing/printing them,
- tracking two marker IDs in a webcam/video feed,
- measuring a “jaw distance” in pixels,
- calibrating it into a **normalized width** `[0, 1]`.

---

## 0) Links (sources & docs)

### UMI OG
- Project site: https://umi-gripper.github.io/
- Paper PDF: https://umi-gripper.github.io/umi.pdf
- arXiv: https://arxiv.org/abs/2402.10329

### FastUMI (comparison / clearer width recipe)
- arXiv HTML (easy to search): https://arxiv.org/html/2409.19499v2
- arXiv: https://arxiv.org/abs/2409.19499

### OpenCV ArUco
- Detection tutorial: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Module reference: https://docs.opencv.org/4.x/d9/d6a/group__aruco.html

### ArUco paper (fiducial theory)
- Garrido-Jurado et al., 2014 (Pattern Recognition): https://www.sciencedirect.com/science/article/pii/S0031320314000235

---

## 1) Why fiducials (UMI context)

UMI OG uses **visual-inertial SLAM** (GoPro video + IMU) to recover the **rigid 6DoF pose** of the wrist camera/gripper.  
But SLAM cannot infer **jaw aperture**, because aperture is an **internal tool DOF**, not rigid motion.

So UMI OG adds **continuous gripper width** tracked from video using **fiducial markers**:
- Continuous width via fiducials (UMI OG paper **p.5**)
- Calibration procedure via repeated open/close (UMI OG paper **p.16**)

This repo isolates only that extra channel: **marker-based gripper width**.

---

## 2) What you can do here

✅ Generate ArUco markers (PNG) for left/right “jaws”  
✅ Detect those markers from webcam or a video file  
✅ Compute pixel distance between markers → `d_px`  
✅ Smooth the signal and compute `width_norm` in `[0,1]`  
✅ Calibrate closed/open and export `gripper_calib.json`  

---

## 3) Install

> **Important:** you need `opencv-contrib-python` (not just `opencv-python`) for `cv2.aruco`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-contrib-python numpy
```

---

## 4) Quick start (no 3D print needed)

### Step A — Generate two markers (PNG)
1) Copy the script below into `generate_markers.py`  
2) Run it; it will create `./markers/*.png`

```bash
python generate_markers.py
```

**How to test without hardware**
- Open both PNGs on a phone/monitor **or print them**.
- Hold them like two “jaws” and move them closer/farther.

---

### Step B — Track width from webcam (or video)
1) Copy the script below into `track_gripper_width.py`  
2) Run it (webcam default). It overlays:
- marker detection,
- `d_px` raw/smoothed,
- `d_min`, `d_max`,
- `width_norm`.

```bash
python track_gripper_width.py
```

**Controls (inside the window)**
- `c` toggle capturing samples for **CLOSED**
- `o` toggle capturing samples for **OPEN**
- `s` save calibration (`d_min`, `d_max`) to `gripper_calib.json`
- `r` reset calibration + samples
- `q` quit

**Calibration procedure**
1) Put markers close together → press `c` (collect closed samples for a few seconds)  
2) Separate markers far apart → press `o` (collect open samples for a few seconds)  
3) Press `s` to save  
4) Now `width_norm` should track smoothly from 0 → 1 as you move markers

---

## 5) Scripts (copy/paste)

### 5.1 `generate_markers.py`
Generates two ArUco marker PNGs you can display or print.

```python
import cv2
from pathlib import Path

def save_aruco_marker(dict_name: str, marker_id: int, size_px: int, out_path: Path):
    aruco = cv2.aruco
    dictionary = getattr(aruco, dict_name)
    d = aruco.getPredefinedDictionary(dictionary)
    img = aruco.generateImageMarker(d, marker_id, size_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

if __name__ == "__main__":
    # Options: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, ...
    dict_name = "DICT_4X4_50"
    size_px = 600

    ids = [10, 11]  # Left / Right marker IDs
    for mid in ids:
        save_aruco_marker(dict_name, mid, size_px, Path(f"markers/aruco_{dict_name}_id{mid}.png"))

    print("Done. Check ./markers/")
```

---

### 5.2 `track_gripper_width.py`
Tracks two marker IDs and measures “aperture” as **pixel distance**, then calibrates it into `[0,1]`.

```python
import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
import argparse

@dataclass
class Calib:
    d_min: float | None = None
    d_max: float | None = None

def get_center(corners: np.ndarray) -> np.ndarray:
    # corners shape: (4,2)
    return corners.mean(axis=0)

def draw_text(img, lines, x=10, y=30, dy=28):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (x, y + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="ArUco-based 'gripper width' (pixel distance) tracker")
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to video file (e.g. demo.mp4). Default: 0")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name. Default: DICT_4X4_50")
    parser.add_argument("--id_left", type=int, default=10, help="Left marker ID")
    parser.add_argument("--id_right", type=int, default=11, help="Right marker ID")
    parser.add_argument("--out", default="gripper_calib.json", help="Where to save calibration JSON")
    parser.add_argument("--alpha", type=float, default=0.25, help="EMA smoothing factor (0..1). Default: 0.25")
    args = parser.parse_args()

    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dict))
    params = aruco.DetectorParameters()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video. Try --source 0 or --source path/to/video.mp4")

    out_calib = Path(args.out)
    calib = Calib()

    calibrating_closed = False
    calibrating_open = False
    d_samples_closed = []
    d_samples_open = []

    d_smoothed = None

    print("Controls:")
    print("  q: quit")
    print("  c: toggle CLOSED sample capture")
    print("  o: toggle OPEN sample capture")
    print("  s: save calibration (d_min, d_max)")
    print("  r: reset calibration")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        d_px = None
        conf = 0.0
        centers = {}

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners_list, ids)
            for c, mid in zip(corners_list, ids.flatten()):
                corners = c.reshape(4, 2)
                centers[int(mid)] = get_center(corners)

            if args.id_left in centers and args.id_right in centers:
                pL = centers[args.id_left]
                pR = centers[args.id_right]
                d_px = float(np.linalg.norm(pL - pR))
                conf = 1.0

                cv2.line(frame, tuple(pL.astype(int)), tuple(pR.astype(int)), (0, 255, 0), 3)
                cv2.circle(frame, tuple(pL.astype(int)), 6, (0, 255, 0), -1)
                cv2.circle(frame, tuple(pR.astype(int)), 6, (0, 255, 0), -1)

        # EMA smoothing
        if d_px is not None:
            d_smoothed = d_px if d_smoothed is None else (args.alpha * d_px + (1 - args.alpha) * d_smoothed)

        # Collect samples
        if d_smoothed is not None:
            if calibrating_closed:
                d_samples_closed.append(d_smoothed)
            if calibrating_open:
                d_samples_open.append(d_smoothed)

        # Compute normalized width if calibrated
        width_norm = None
        if calib.d_min is not None and calib.d_max is not None and d_smoothed is not None:
            denom = (calib.d_max - calib.d_min)
            if abs(denom) > 1e-6:
                width_norm = float(np.clip((d_smoothed - calib.d_min) / denom, 0.0, 1.0))

        lines = [
            f"dict={args.dict}  L={args.id_left} R={args.id_right}",
            f"detected_conf={conf:.1f}",
            f"d_px(raw)={d_px:.1f}" if d_px is not None else "d_px(raw)=None",
            f"d_px(smooth)={d_smoothed:.1f}" if d_smoothed is not None else "d_px(smooth)=None",
            f"calib d_min={calib.d_min:.1f}" if calib.d_min is not None else "calib d_min=None",
            f"calib d_max={calib.d_max:.1f}" if calib.d_max is not None else "calib d_max=None",
            f"width_norm={width_norm:.3f}" if width_norm is not None else "width_norm=None",
            f"[CAL] closed_samples={len(d_samples_closed)} open_samples={len(d_samples_open)}",
            "Keys: c(closed) o(open) s(save) r(reset) q(quit)"
        ]
        draw_text(frame, lines)

        cv2.imshow("UMI-style Gripper Width (ArUco)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            calibrating_closed = not calibrating_closed
            print("CLOSED capture:", "ON" if calibrating_closed else "OFF")
        elif key == ord("o"):
            calibrating_open = not calibrating_open
            print("OPEN capture:", "ON" if calibrating_open else "OFF")
        elif key == ord("r"):
            calib = Calib()
            d_samples_closed.clear()
            d_samples_open.clear()
            print("Calibration reset.")
        elif key == ord("s"):
            if len(d_samples_closed) >= 10 and len(d_samples_open) >= 10:
                dmin = float(np.percentile(d_samples_closed, 5))
                dmax = float(np.percentile(d_samples_open, 95))
                if dmax <= dmin + 1e-3:
                    print("Invalid calibration: d_max <= d_min. Re-capture samples.")
                else:
                    calib.d_min = dmin
                    calib.d_max = dmax
                    out = {
                        "dict": args.dict,
                        "id_left": args.id_left,
                        "id_right": args.id_right,
                        "d_min": calib.d_min,
                        "d_max": calib.d_max
                    }
                    out_calib.write_text(json.dumps(out, indent=2))
                    print("Saved calibration to:", out_calib.resolve())
            else:
                print("Not enough samples. Toggle 'c' and 'o' to capture CLOSED/OPEN first.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 6) What “width” means here (and how it matches UMI)

### Raw width signal
We use a simple proxy:
- detect **Left** and **Right** markers,
- compute pixel distance between their centers:
  `d_px = ||center_L - center_R||`.

### Normalized width
After calibration:
- `width_norm = clip((d_px - d_min) / (d_max - d_min), 0, 1)`.

This matches UMI’s concept:
- fiducials provide **continuous gripper width** (UMI OG **p.5**),
- a short open/close procedure defines min/max (UMI OG **p.16**).

---

## 7) Suggested tests (before you have the printed gripper)

### Detection robustness
- try different marker sizes (small vs large)
- lighting: bright / dim / warm
- motion blur: move fast

### Dropout behavior
- occlude one marker intentionally
- observe confidence and smoothing

### Repeatability
- calibrate today, test tomorrow
- watch drift due to exposure/focus changes

---

## 8) Known limitations (current prototype)

- ✅ Great for prototyping and learning the pipeline
- ❌ Does not convert to mm unless you define physical min/max widths
- ❌ No camera undistortion by default (GoPro wide FOV can distort)
- ❌ No multi-marker boards per jaw (better under occlusion)
- ❌ No SLAM alignment/timestamp fusion (this repo isolates width only)

---

## 9) Roadmap (UMI-ready upgrades)

- [ ] Add per-frame logging CSV/JSONL: `timestamp, d_px, width_norm, conf`
- [ ] Add camera calibration + undistort option (important for GoPro wide)
- [ ] Add **ArUco board per jaw** for occlusion robustness
- [ ] Add QA report: marker visibility %, dropout rate, longest gap
- [ ] Add dataset export stub compatible with our UMI pipeline

---

## 10) Notes on UMI OG vs FastUMI (gripper width)

- UMI OG states that width is tracked via fiducials (p.5) and calibrated via a procedure (p.16) but leaves implementation details open.
- FastUMI describes a clearer recipe: two markers on jaws → pixel distance → linear mapping to width, improving reproducibility.

FastUMI:
- https://arxiv.org/html/2409.19499v2
- https://arxiv.org/abs/2409.19499
