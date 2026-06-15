from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.renderer.mesh_renderer import SkeletonRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D pose to a skeleton video")
    parser.add_argument("--input", type=Path, default=Path("output/dance/pose3d.npy"), help="Input pose3d .npy")
    parser.add_argument("--skeleton-output", type=Path, default=Path("output/dance/skeleton3d.mp4"))
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose3d = np.load(args.input)
    renderer = SkeletonRenderer()
    renderer.render_skeleton(pose3d, args.skeleton_output, fps=args.fps)
    print(f"Saved skeleton video to {args.skeleton_output}")


if __name__ == "__main__":
    main()
