from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import MocapPipeline, PipelineConfig
from src.pose2d.rtmpose_estimator import RTMPoseEstimator
from src.pose3d.adapters import KeypointFormat, Pose2DFormatConverter
from src.pose3d.posemamba_estimator import PoseMambaEstimator
from src.renderer.mesh_renderer import SkeletonRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to 3D skeleton demo pipeline")
    parser.add_argument("video", type=Path, help="Path to an input .mp4 video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for pose2d.npy, pose3d.npy, and mesh.mp4",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only read video metadata. Useful for Phase 1 validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = MocapPipeline(
        config=PipelineConfig(
            video_path=args.video,
            output_dir=args.output_dir,
            metadata_only=args.metadata_only,
        ),
        pose2d_estimator=RTMPoseEstimator(),
        pose_converter=Pose2DFormatConverter(
            source_format=KeypointFormat.RTMPOSE_RAW,
            target_format=KeypointFormat.HUMAN36M_17,
        ),
        pose3d_estimator=PoseMambaEstimator(),
        renderer=SkeletonRenderer(),
    )

    result = pipeline.run()
    print(result.summary())


if __name__ == "__main__":
    main()
