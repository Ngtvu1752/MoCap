from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.pose3d.adapters import KeypointFormat, Pose2DFormatConverter
from src.pose3d.motionbert_estimator import MotionBERTEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift RTMPose 2D keypoints to 3D with MotionBERT")
    parser.add_argument("--input", type=Path, default=Path("output/pose2d.npy"), help="Input pose2d .npy")
    parser.add_argument("--output", type=Path, default=Path("output/pose3d.npy"), help="Output pose3d .npy")
    parser.add_argument("--input-format", choices=["rtmpose_raw", "human36m_17"], default="rtmpose_raw")
    parser.add_argument("--motionbert-repo", type=Path, default=Path("MotionBERT"))
    parser.add_argument("--config", type=Path, default=Path("MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose2d = np.load(args.input)

    if args.input_format == "rtmpose_raw":
        converter = Pose2DFormatConverter(KeypointFormat.RTMPOSE_RAW, KeypointFormat.HUMAN36M_17)
        pose2d = converter.convert(pose2d)

    estimator = MotionBERTEstimator(
        repo_path=args.motionbert_repo,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    result = estimator.lift(pose2d)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.keypoints)
    print(f"Saved {result.keypoints.shape} to {args.output}")


if __name__ == "__main__":
    main()
