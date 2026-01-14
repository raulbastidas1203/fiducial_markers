import argparse
from pathlib import Path

import cv2

from scripts.utils_aruco import build_calibration, detect_markers, get_aruco_dictionary, percentile


def expand_glob(pattern: str) -> list[Path]:
    return sorted(Path().glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate gripper width from closed/open images.")
    parser.add_argument("--closed", required=True, help="Glob for closed images (e.g. calib/closed/*.jpg)")
    parser.add_argument("--open", required=True, help="Glob for open images (e.g. calib/open/*.jpg)")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--id_left", type=int, default=10, help="Left marker ID")
    parser.add_argument("--id_right", type=int, default=11, help="Right marker ID")
    parser.add_argument("--out", default="calib/gripper_calib.json", help="Output calibration JSON")
    args = parser.parse_args()

    closed_paths = expand_glob(args.closed)
    open_paths = expand_glob(args.open)

    if not closed_paths:
        raise RuntimeError(f"No closed images found for pattern: {args.closed}")
    if not open_paths:
        raise RuntimeError(f"No open images found for pattern: {args.open}")

    dictionary = get_aruco_dictionary(args.dict)
    params = cv2.aruco.DetectorParameters()

    def collect_distances(paths: list[Path]) -> list[float]:
        distances = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"[WARN] Could not read {path}, skipping.")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detection = detect_markers(gray, dictionary, params, args.id_left, args.id_right)
            if detection.conf > 0:
                distances.append(detection.d_px_raw)
            else:
                print(f"[WARN] Missing markers in {path}, skipping.")
        return distances

    closed_distances = collect_distances(closed_paths)
    open_distances = collect_distances(open_paths)

    if not closed_distances or not open_distances:
        raise RuntimeError("Not enough detections to calibrate. Check marker visibility.")

    d_min = percentile(closed_distances, 10)
    d_max = percentile(open_distances, 90)
    if d_max <= d_min:
        raise RuntimeError("Invalid calibration: d_max <= d_min.")

    sources = [str(p) for p in closed_paths + open_paths]
    calib = build_calibration(args.dict, args.id_left, args.id_right, d_min, d_max, "images", sources)
    calib.save(Path(args.out))
    print(f"Saved calibration to {args.out}")


if __name__ == "__main__":
    main()
