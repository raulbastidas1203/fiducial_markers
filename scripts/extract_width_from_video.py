import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from scripts.utils_aruco import (
    Calibration,
    Smoother,
    compute_width_norm,
    detect_markers,
    get_aruco_dictionary,
    interpolate_nans,
)


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def compute_dropout(conf_values: np.ndarray) -> int:
    max_gap = 0
    current = 0
    for conf in conf_values:
        if conf == 0:
            current += 1
            max_gap = max(max_gap, current)
        else:
            current = 0
    return max_gap


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "timestamp_sec", "conf", "d_px_raw", "d_px_smooth", "width_norm"])
        writer.writerows(rows)


def extract_width_from_video(
    video_path: str,
    calib_path: str,
    out_csv: str,
    dict_name: str | None,
    id_left: int | None,
    id_right: int | None,
    smooth: str,
    alpha: float,
    window: int,
    interpolate_missing: bool,
    display: bool,
    preview_path: str | None,
) -> tuple[Path, Path]:
    calib = Calibration.from_file(Path(calib_path))

    dict_name = dict_name or calib.dict_name
    id_left = id_left if id_left is not None else calib.id_left
    id_right = id_right if id_right is not None else calib.id_right

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    dictionary = get_aruco_dictionary(dict_name)
    params = cv2.aruco.DetectorParameters()

    smoother = Smoother(smooth, alpha, window)

    timestamps: list[float] = []
    confs: list[float] = []
    d_px_raw_list: list[float] = []
    d_px_smooth_list: list[float] = []

    writer = None
    if preview_path:
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(preview_path), fps=fps if fps > 0 else 30)

    progress = tqdm(total=total_frames, desc="Extracting", unit="frame") if total_frames else None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_markers(gray, dictionary, params, id_left, id_right)

        d_px_smooth = smoother.update(detection.d_px_raw)

        if display or writer is not None:
            overlay = [
                f"frame={frame_idx}",
                f"t={timestamp_sec:.3f}s",
                f"conf={detection.conf:.1f}",
                f"d_px_raw={detection.d_px_raw:.2f}" if detection.conf > 0 else "d_px_raw=NaN",
                f"d_px_smooth={d_px_smooth:.2f}" if not np.isnan(d_px_smooth) else "d_px_smooth=NaN",
            ]
            for i, line in enumerate(overlay):
                y = 30 + i * 28
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if display:
            cv2.imshow("Width Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        timestamps.append(timestamp_sec)
        confs.append(detection.conf)
        d_px_raw_list.append(detection.d_px_raw)
        d_px_smooth_list.append(d_px_smooth)

        frame_idx += 1
        if progress:
            progress.update(1)

    cap.release()
    if progress:
        progress.close()
    if display:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.close()

    d_px_raw_arr = np.array(d_px_raw_list, dtype=float)
    d_px_smooth_arr = np.array(d_px_smooth_list, dtype=float)
    conf_arr = np.array(confs, dtype=float)
    timestamp_arr = np.array(timestamps, dtype=float)

    if interpolate_missing:
        d_px_raw_arr = interpolate_nans(d_px_raw_arr)
        d_px_smooth_arr = interpolate_nans(d_px_smooth_arr)

    width_norm_arr = np.array([compute_width_norm(val, calib) for val in d_px_smooth_arr], dtype=float)

    rows = []
    for idx in range(len(timestamp_arr)):
        rows.append(
            [
                str(idx),
                f"{timestamp_arr[idx]:.6f}",
                f"{conf_arr[idx]:.1f}",
                f"{d_px_raw_arr[idx]:.6f}" if not np.isnan(d_px_raw_arr[idx]) else "NaN",
                f"{d_px_smooth_arr[idx]:.6f}" if not np.isnan(d_px_smooth_arr[idx]) else "NaN",
                f"{width_norm_arr[idx]:.6f}" if not np.isnan(width_norm_arr[idx]) else "NaN",
            ]
        )

    out_csv_path = Path(out_csv)
    write_csv(out_csv_path, rows)

    qa = {
        "fps": fps,
        "total_frames": int(len(timestamp_arr)),
        "detected_ratio": float(conf_arr.mean()) if len(conf_arr) else 0.0,
        "max_dropout_frames": int(compute_dropout(conf_arr)),
        "n_missing_frames": int(np.sum(conf_arr == 0)),
        "min_width": float(np.nanmin(width_norm_arr)) if np.any(~np.isnan(width_norm_arr)) else None,
        "max_width": float(np.nanmax(width_norm_arr)) if np.any(~np.isnan(width_norm_arr)) else None,
    }

    qa_path = out_csv_path.parent / "qa.json"
    qa_path.write_text(json.dumps(qa, indent=2))
    print(f"Saved CSV to {out_csv_path}")
    print(f"Saved QA to {qa_path}")
    return out_csv_path, qa_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract width_norm from a video using ArUco markers.")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--calib", required=True, help="Calibration JSON (e.g. calib/gripper_calib.json)")
    parser.add_argument("--out", required=True, help="Output CSV (e.g. outputs/demo/width.csv)")
    parser.add_argument("--dict", default=None, help="Override ArUco dictionary name")
    parser.add_argument("--id_left", type=int, default=None, help="Override left marker ID")
    parser.add_argument("--id_right", type=int, default=None, help="Override right marker ID")
    parser.add_argument("--smooth", choices=["ema", "moving"], default="ema")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--interpolate_missing", type=parse_bool, default=False)
    parser.add_argument("--display", type=parse_bool, default=False)
    parser.add_argument("--preview", default=None, help="Optional preview MP4 with overlay")
    args = parser.parse_args()

    extract_width_from_video(
        video_path=args.video,
        calib_path=args.calib,
        out_csv=args.out,
        dict_name=args.dict,
        id_left=args.id_left,
        id_right=args.id_right,
        smooth=args.smooth,
        alpha=args.alpha,
        window=args.window,
        interpolate_missing=args.interpolate_missing,
        display=args.display,
        preview_path=args.preview,
    )


if __name__ == "__main__":
    main()
