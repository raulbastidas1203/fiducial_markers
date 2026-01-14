import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    d_min: float | None = None
    d_max: float | None = None


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def get_center(corners: np.ndarray) -> np.ndarray:
    return corners.mean(axis=0)


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    if values.size == 0:
        return values
    x = np.arange(values.size)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return values
    values[~mask] = np.interp(x[~mask], x[mask], values[mask])
    return values


def draw_overlay(frame, lines):
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UMI-style width_norm from an MP4 video.")
    parser.add_argument("--video", required=True, help="Path to input video (e.g. demo.mp4)")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name. Default: DICT_4X4_50")
    parser.add_argument("--id_left", type=int, default=10, help="Left marker ID")
    parser.add_argument("--id_right", type=int, default=11, help="Right marker ID")
    parser.add_argument("--calib_min_px", type=float, default=None, help="Optional calibrated min distance in pixels")
    parser.add_argument("--calib_max_px", type=float, default=None, help="Optional calibrated max distance in pixels")
    parser.add_argument("--calib_out", default="gripper_calib.json", help="Where to save calibration JSON")
    parser.add_argument("--save", required=True, help="Path to output CSV")
    parser.add_argument("--smooth", choices=["ema", "moving"], default="ema", help="Smoothing method")
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA alpha (0..1)")
    parser.add_argument("--window", type=int, default=5, help="Moving average window size")
    parser.add_argument(
        "--interpolate_missing",
        type=parse_bool,
        default=False,
        help="Interpolate missing values (true/false)",
    )
    parser.add_argument(
        "--display",
        type=parse_bool,
        default=True,
        help="Show interactive window (true/false). Required for interactive calibration.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dict))
    params = aruco.DetectorParameters()

    calib = Calibration(d_min=args.calib_min_px, d_max=args.calib_max_px)
    waiting_for_calib = calib.d_min is None or calib.d_max is None
    if waiting_for_calib and not args.display:
        print("Calibration requested but display disabled; continuing without interactive calibration.")

    d_px_raw_list = []
    d_px_smooth_list = []
    detected_conf_list = []
    timestamp_list = []
    d_min_list = []
    d_max_list = []

    ema_value = None
    moving_window = []

    frame_idx = 0

    print("Processing video:", args.video)
    if waiting_for_calib and args.display:
        print("Interactive calibration:")
        print("  c: set CLOSED (min) using current d_px")
        print("  o: set OPEN (max) using current d_px")
        print("  s: save calibration JSON and continue")
        print("  q: quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

        centers = {}
        d_px_raw = np.nan
        detected_conf = 0.0

        if ids is not None and len(ids) > 0:
            for c, mid in zip(corners_list, ids.flatten()):
                corners = c.reshape(4, 2)
                centers[int(mid)] = get_center(corners)

            if args.id_left in centers and args.id_right in centers:
                pL = centers[args.id_left]
                pR = centers[args.id_right]
                d_px_raw = float(np.linalg.norm(pL - pR))
                detected_conf = 1.0

        if np.isnan(d_px_raw):
            d_px_smooth = np.nan
        else:
            if args.smooth == "ema":
                if ema_value is None:
                    ema_value = d_px_raw
                else:
                    ema_value = args.alpha * d_px_raw + (1 - args.alpha) * ema_value
                d_px_smooth = float(ema_value)
            else:
                moving_window.append(d_px_raw)
                if len(moving_window) > args.window:
                    moving_window.pop(0)
                d_px_smooth = float(np.mean(moving_window))

        if waiting_for_calib and args.display:
            overlay_lines = [
                f"frame={frame_idx} t={timestamp_sec:.3f}s",
                f"d_px_raw={d_px_raw:.2f}" if not np.isnan(d_px_raw) else "d_px_raw=NaN",
                f"d_px_smooth={d_px_smooth:.2f}" if not np.isnan(d_px_smooth) else "d_px_smooth=NaN",
                f"calib d_min={calib.d_min if calib.d_min is not None else 'None'}",
                f"calib d_max={calib.d_max if calib.d_max is not None else 'None'}",
                "Keys: c(min) o(max) s(save) q(quit)",
            ]
            draw_overlay(frame, overlay_lines)
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c") and not np.isnan(d_px_smooth):
                calib.d_min = d_px_smooth
                print(f"Set d_min={calib.d_min:.2f}")
            if key == ord("o") and not np.isnan(d_px_smooth):
                calib.d_max = d_px_smooth
                print(f"Set d_max={calib.d_max:.2f}")
            if key == ord("s") and calib.d_min is not None and calib.d_max is not None:
                out_calib = {
                    "dict": args.dict,
                    "id_left": args.id_left,
                    "id_right": args.id_right,
                    "d_min": calib.d_min,
                    "d_max": calib.d_max,
                }
                Path(args.calib_out).write_text(json.dumps(out_calib, indent=2))
                print(f"Saved calibration to {args.calib_out}")
                waiting_for_calib = False
                cv2.destroyWindow("Calibration")
        else:
            if args.display and waiting_for_calib:
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        d_px_raw_list.append(d_px_raw)
        d_px_smooth_list.append(d_px_smooth)
        detected_conf_list.append(detected_conf)
        timestamp_list.append(timestamp_sec)
        d_min_list.append(calib.d_min if calib.d_min is not None else np.nan)
        d_max_list.append(calib.d_max if calib.d_max is not None else np.nan)

        frame_idx += 1

    cap.release()
    if args.display:
        cv2.destroyAllWindows()

    d_px_raw_arr = np.array(d_px_raw_list, dtype=float)
    d_px_smooth_arr = np.array(d_px_smooth_list, dtype=float)
    detected_conf_arr = np.array(detected_conf_list, dtype=float)
    timestamp_arr = np.array(timestamp_list, dtype=float)
    d_min_arr = np.array(d_min_list, dtype=float)
    d_max_arr = np.array(d_max_list, dtype=float)

    if args.interpolate_missing:
        d_px_raw_arr = interpolate_nans(d_px_raw_arr)
        d_px_smooth_arr = interpolate_nans(d_px_smooth_arr)

    width_norm_arr = np.full_like(d_px_smooth_arr, np.nan, dtype=float)
    if calib.d_min is not None and calib.d_max is not None:
        denom = calib.d_max - calib.d_min
        if abs(denom) > 1e-6:
            width_norm_arr = np.clip((d_px_smooth_arr - calib.d_min) / denom, 0.0, 1.0)

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "timestamp_sec",
                "detected_conf",
                "d_px_raw",
                "d_px_smooth",
                "d_min",
                "d_max",
                "width_norm",
            ]
        )
        for idx in range(len(timestamp_arr)):
            writer.writerow(
                [
                    idx,
                    f"{timestamp_arr[idx]:.6f}",
                    f"{detected_conf_arr[idx]:.3f}",
                    f"{d_px_raw_arr[idx]:.6f}" if not np.isnan(d_px_raw_arr[idx]) else "NaN",
                    f"{d_px_smooth_arr[idx]:.6f}" if not np.isnan(d_px_smooth_arr[idx]) else "NaN",
                    f"{d_min_arr[idx]:.6f}" if not np.isnan(d_min_arr[idx]) else "NaN",
                    f"{d_max_arr[idx]:.6f}" if not np.isnan(d_max_arr[idx]) else "NaN",
                    f"{width_norm_arr[idx]:.6f}" if not np.isnan(width_norm_arr[idx]) else "NaN",
                ]
            )

    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
