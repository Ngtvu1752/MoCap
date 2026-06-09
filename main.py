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
    parser.add_argument(
        "--pose2d-only",
        action="store_true",
        help="Run RTMPose and save pose2d.npy, then stop before 3D lifting.",
    )
    parser.add_argument(
        "--pose2d-config",
        type=Path,
        default=Path("checkpoints/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"),
        help="RTMPose config file path",
    )
    parser.add_argument(
        "--pose2d-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
        ),
        help="RTMPose checkpoint .pth path",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = MocapPipeline(
        config=PipelineConfig(
            video_path=args.video,
            output_dir=args.output_dir,
            metadata_only=args.metadata_only,
            pose2d_only=args.pose2d_only,
        ),
        pose2d_estimator=RTMPoseEstimator(
            config_path=args.pose2d_config,
            checkpoint_path=args.pose2d_checkpoint,
            device=args.device,
        ),
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
