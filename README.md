# Fiducial Markers — UMI Width-Only Pipeline

This repo implements a **width-only** pipeline for UMI-style gripper aperture tracking using **ArUco fiducials**. It calibrates `d_min / d_max` and then extracts a normalized `width_norm` per frame from MP4 videos.

---

## Folder structure

```
fiducial_markers/
  scripts/
    utils_aruco.py
    calibrate_from_images.py
    calibrate_from_video.py
    extract_width_from_video.py
    batch_extract_width.py
  calib/
    gripper_calib.json        (generated; ignored by git)
  outputs/
    <episode_name>/
      width.csv
      qa.json
      preview.mp4             (optional)
  requirements.txt
  .gitignore
  README.md
```

`calib/gripper_calib.json` is **generated** and is **ignored by git** (see `.gitignore`).

---

## Install (Windows / PowerShell / Python 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Troubleshooting:** If `cv2.aruco` is missing, ensure you installed **opencv-contrib-python**.

---

## Calibration (images)

Use closed/open photos to estimate `d_min/d_max`.

```powershell
python scripts\calibrate_from_images.py `
  --closed "calib\closed\*.jpg" `
  --open "calib\open\*.jpg" `
  --out calib\gripper_calib.json
```

This computes:
- `d_min = percentile(closed, 10)`
- `d_max = percentile(open, 90)`

---

## Calibration (video)

### Auto
```powershell
python scripts\calibrate_from_video.py `
  --video calib.mp4 `
  --mode auto `
  --out calib\gripper_calib.json
```

Auto mode uses `d_min = p2`, `d_max = p98` of all detected frames.

### Interactive
```powershell
python scripts\calibrate_from_video.py `
  --video calib.mp4 `
  --mode interactive `
  --out calib\gripper_calib.json
```

Keys:
- `c` toggle capture **closed**
- `o` toggle capture **open**
- `s` save and exit
- `q` quit

Interactive percentiles:
`d_min = p5(closed)`, `d_max = p95(open)`.

---

## Extract width per episode

```powershell
python scripts\extract_width_from_video.py `
  --video episodes\demo.mp4 `
  --calib calib\gripper_calib.json `
  --out outputs\demo\width.csv `
  --smooth ema `
  --alpha 0.2 `
  --interpolate_missing true `
  --preview outputs\demo\preview.mp4
```

Output CSV columns:
```
frame_idx, timestamp_sec, conf, d_px_raw, d_px_smooth, width_norm
```

QA metrics are written to `outputs/<episode>/qa.json`.

---

## Batch extraction

```powershell
python scripts\batch_extract_width.py `
  --input_dir episodes `
  --calib calib\gripper_calib.json `
  --out_dir outputs `
  --interpolate_missing true `
  --preview false
```

Each MP4 in `episodes/` creates:
```
outputs/<video_stem>/width.csv
outputs/<video_stem>/qa.json
outputs/<video_stem>/preview.mp4 (if enabled)
```

---

## Notes

- This repo is **width-only**: no SLAM, no 6DoF pose.
- GoPro videos may be **HEVC**. If OpenCV fails to open them, convert to **H.264** first.
