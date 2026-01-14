import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.utils_aruco import (
    build_calibration,
    detect_markers,
    get_aruco_dictionary,
    percentile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate gripper width from a video.")
    parser.add_argument("--video", required=True, help="Path to calibration video")
    parser.add_argument("--mode", choices=["auto", "interactive"], default="auto")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--id_left", type=int, default=10, help="Left marker ID")
    parser.add_argument("--id_right", type=int, default=11, help="Right marker ID")
    parser.add_argument("--out", default="calib/gripper_calib.json", help="Output calibration JSON")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    dictionary = get_aruco_dictionary(args.dict)
    params = cv2.aruco.DetectorParameters()

    closed_samples: list[float] = []
    open_samples: list[float] = []
    all_samples: list[float] = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = tqdm(total=total_frames, desc="Calibrating", unit="frame") if args.mode == "auto" else None

    capture_closed = False
    capture_open = False

    if args.mode == "interactive":
        print("Interactive calibration:")
        print("  c: toggle capture CLOSED")
        print("  o: toggle capture OPEN")
        print("  s: save calibration and exit")
        print("  q: quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_markers(gray, dictionary, params, args.id_left, args.id_right)

        if detection.conf > 0:
            all_samples.append(detection.d_px_raw)
            if capture_closed:
                closed_samples.append(detection.d_px_raw)
            if capture_open:
                open_samples.append(detection.d_px_raw)

        if args.mode == "interactive":
            overlay = [
                f"conf={detection.conf:.1f}",
                f"d_px={detection.d_px_raw:.2f}" if detection.conf > 0 else "d_px=NaN",
                f"closed_samples={len(closed_samples)} open_samples={len(open_samples)}",
                "Keys: c(closed) o(open) s(save) q(quit)",
            ]
            for i, line in enumerate(overlay):
                y = 30 + i * 28
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                capture_closed = not capture_closed
                print("CLOSED capture:", "ON" if capture_closed else "OFF")
            if key == ord("o"):
                capture_open = not capture_open
                print("OPEN capture:", "ON" if capture_open else "OFF")
            if key == ord("s"):
                if len(closed_samples) < 5 or len(open_samples) < 5:
                    print("Not enough samples to save calibration.")
                else:
                    break
        else:
            if progress:
                progress.update(1)

    cap.release()
    if progress:
        progress.close()
    if args.mode == "interactive":
        cv2.destroyAllWindows()

    if args.mode == "auto":
        if not all_samples:
            raise RuntimeError("No valid detections found for calibration.")
        d_min = percentile(all_samples, 2)
        d_max = percentile(all_samples, 98)
    else:
        if len(closed_samples) < 5 or len(open_samples) < 5:
            raise RuntimeError("Not enough interactive samples to calibrate.")
        d_min = percentile(closed_samples, 5)
        d_max = percentile(open_samples, 95)

    if d_max <= d_min:
        raise RuntimeError("Invalid calibration: d_max <= d_min.")

    sources = [args.video]
    calib = build_calibration(args.dict, args.id_left, args.id_right, d_min, d_max, "video", sources)
    calib.save(Path(args.out))
    print(f"Saved calibration to {args.out}")


if __name__ == "__main__":
    main()
