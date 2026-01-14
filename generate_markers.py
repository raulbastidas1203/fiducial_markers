import cv2
from pathlib import Path


def save_aruco_marker(dict_name: str, marker_id: int, size_px: int, out_path: Path) -> None:
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
