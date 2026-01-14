import argparse
from pathlib import Path

from tqdm import tqdm

from scripts.extract_width_from_video import extract_width_from_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract width_norm from a folder of MP4s.")
    parser.add_argument("--input_dir", required=True, help="Directory with MP4 episodes")
    parser.add_argument("--calib", required=True, help="Calibration JSON")
    parser.add_argument("--out_dir", required=True, help="Output directory for episode results")
    parser.add_argument("--smooth", choices=["ema", "moving"], default="ema")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--interpolate_missing", default="false")
    parser.add_argument("--preview", default="false", help="Generate preview MP4s (true/false)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    videos = sorted(input_dir.glob("*.mp4"))

    if not videos:
        raise RuntimeError(f"No MP4 files found in {input_dir}")

    preview_enabled = args.preview.lower() in {"true", "1", "yes", "y"}
    interpolate_missing = args.interpolate_missing.lower() in {"true", "1", "yes", "y"}

    for video_path in tqdm(videos, desc="Episodes", unit="video"):
        episode_dir = out_dir / video_path.stem
        out_csv = episode_dir / "width.csv"
        preview_path = episode_dir / "preview.mp4" if preview_enabled else None

        extract_width_from_video(
            video_path=str(video_path),
            calib_path=args.calib,
            out_csv=str(out_csv),
            dict_name=None,
            id_left=None,
            id_right=None,
            smooth=args.smooth,
            alpha=args.alpha,
            window=args.window,
            interpolate_missing=interpolate_missing,
            display=False,
            preview_path=str(preview_path) if preview_path else None,
        )

    print(f"Processed {len(videos)} video(s).")


if __name__ == "__main__":
    main()
