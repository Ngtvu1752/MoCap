from __future__ import annotations

import argparse
from pathlib import Path

from src.io.video_reader import VideoReader, _load_cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/frames"),
        help="Directory where extracted frames are saved",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of frames to extract",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv2 = _load_cv2()
    saved = 0
    with VideoReader(args.video) as reader:
        for frame in reader.iter_frames():
            if args.limit is not None and saved >= args.limit:
                break

            frame_path = args.output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(frame_path), frame.image)
            saved += 1

    print(f"Saved {saved} frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
