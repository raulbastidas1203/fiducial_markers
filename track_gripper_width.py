import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calib:
    d_min: float | None = None
    d_max: float | None = None


def get_center(corners: np.ndarray) -> np.ndarray:
    # corners shape: (4,2)
    return corners.mean(axis=0)


def draw_text(img, lines, x=10, y=30, dy=28):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="ArUco-based 'gripper width' (pixel distance) tracker")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0) or path to video file (e.g. demo.mp4). Default: 0",
    )
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
            denom = calib.d_max - calib.d_min
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
            "Keys: c(closed) o(open) s(save) r(reset) q(quit)",
        ]
        draw_text(frame, lines)

        cv2.imshow("UMI-style Gripper Width (ArUco)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c"):
            calibrating_closed = not calibrating_closed
            print("CLOSED capture:", "ON" if calibrating_closed else "OFF")
        if key == ord("o"):
            calibrating_open = not calibrating_open
            print("OPEN capture:", "ON" if calibrating_open else "OFF")
        if key == ord("r"):
            calib = Calib()
            d_samples_closed.clear()
            d_samples_open.clear()
            print("Calibration reset.")
        if key == ord("s"):
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
                        "d_max": calib.d_max,
                    }
                    out_calib.write_text(json.dumps(out, indent=2))
                    print("Saved calibration to:", out_calib.resolve())
            else:
                print("Not enough samples. Toggle 'c' and 'o' to capture CLOSED/OPEN first.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
