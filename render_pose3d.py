from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.renderer.mesh_renderer import SkeletonRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D pose to skeleton and tube-mesh videos")
    parser.add_argument("--input", type=Path, default=Path("output/pose3d.npy"), help="Input pose3d .npy")
    parser.add_argument("--skeleton-output", type=Path, default=Path("output/skeleton3d.mp4"))
    parser.add_argument("--mesh-output", type=Path, default=Path("output/tube_mesh.mp4"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mode", choices=["both", "skeleton", "mesh"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose3d = np.load(args.input)
    renderer = SkeletonRenderer()

    if args.mode in {"both", "skeleton"}:
        renderer.render_skeleton(pose3d, args.skeleton_output, fps=args.fps)
        print(f"Saved skeleton video to {args.skeleton_output}")
    if args.mode in {"both", "mesh"}:
        renderer.render_mesh(pose3d, args.mesh_output, fps=args.fps)
        print(f"Saved tube-mesh video to {args.mesh_output}")


if __name__ == "__main__":
    main()
