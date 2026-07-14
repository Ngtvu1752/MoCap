from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import PipelineConfig
from src.pipeline_defaults import (
    DEFAULT_DEVICE,
    DEFAULT_MESH_CHECKPOINT,
    DEFAULT_MESH_CLIP_LEN,
    DEFAULT_MESH_CLIP_STRIDE,
    DEFAULT_MESH_CONFIG,
    DEFAULT_MOTIONBERT_REPO,
    DEFAULT_POSE2D_CHECKPOINT,
    DEFAULT_POSE2D_CONFIG,
    DEFAULT_POSE2D_FORMAT,
    DEFAULT_POSE3D_CHECKPOINT,
    DEFAULT_POSE3D_CONFIG,
    DEFAULT_SMPL_DATA_ROOT,
    SUPPORTED_POSE2D_FORMATS,
)
from src.pipeline_factory import create_mocap_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to 3D skeleton/demo human mesh pipeline")
    parser.add_argument("video", type=Path, help="Path to an input .mp4 video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for per-video output folders",
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
        "--pose2d-format",
        choices=SUPPORTED_POSE2D_FORMATS,
        default=DEFAULT_POSE2D_FORMAT,
        help="RTMPose keypoint layout emitted by the 2D model",
    )
    parser.add_argument(
        "--pose2d-config",
        type=Path,
        default=DEFAULT_POSE2D_CONFIG,
        help="RTMPose config file path",
    )
    parser.add_argument(
        "--pose2d-checkpoint",
        type=Path,
        default=DEFAULT_POSE2D_CHECKPOINT,
        help="RTMPose checkpoint .pth path",
    )
    parser.add_argument(
        "--pose3d-only",
        action="store_true",
        help="Run 2D pose + 3D lifting, then stop before rendering/HMR.",
    )
    parser.add_argument(
        "--motionbert-repo",
        type=Path,
        default=DEFAULT_MOTIONBERT_REPO,
        help="MotionBERT repository path",
    )
    parser.add_argument(
        "--pose3d-config",
        type=Path,
        default=DEFAULT_POSE3D_CONFIG,
        help="MotionBERT pose3d config file path",
    )
    parser.add_argument(
        "--pose3d-checkpoint",
        type=Path,
        default=DEFAULT_POSE3D_CHECKPOINT,
        help="MotionBERT pose3d checkpoint .bin path",
    )
    parser.add_argument(
        "--human-mesh",
        action="store_true",
        help="Run MotionBERT Human Mesh Recovery and save SMPL outputs.",
    )
    parser.add_argument(
        "--mesh-config",
        type=Path,
        default=DEFAULT_MESH_CONFIG,
        help="MotionBERT mesh config file path",
    )
    parser.add_argument(
        "--mesh-checkpoint",
        type=Path,
        default=DEFAULT_MESH_CHECKPOINT,
        help="MotionBERT mesh checkpoint .bin path",
    )
    parser.add_argument(
        "--smpl-data-root",
        type=Path,
        default=DEFAULT_SMPL_DATA_ROOT,
        help="Directory containing SMPL_NEUTRAL.pkl and MotionBERT mesh regressor assets",
    )
    parser.add_argument(
        "--skip-human-mesh-render",
        action="store_true",
        help="Save SMPL .npy outputs but skip rendering human_mesh.mp4.",
    )
    parser.add_argument(
        "--mesh-clip-len",
        type=int,
        default=DEFAULT_MESH_CLIP_LEN,
        help="MotionBERT mesh clip length.",
    )
    parser.add_argument(
        "--mesh-clip-stride",
        type=int,
        default=DEFAULT_MESH_CLIP_STRIDE,
        help=(
            "Number of overlapping frames between adjacent MotionBERT mesh clips. "
            "For example, with --mesh-clip-len 243 and --mesh-clip-stride 121, "
            "adjacent clips start 122 frames apart."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Inference device, for example cuda:0 or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = create_mocap_pipeline(
        PipelineConfig(
            video_path=args.video,
            output_dir=args.output_dir,
            metadata_only=args.metadata_only,
            pose2d_only=args.pose2d_only,
            pose3d_only=args.pose3d_only,
            human_mesh=args.human_mesh,
            render_human_mesh=not args.skip_human_mesh_render,
            pose2d_format=args.pose2d_format,
            pose2d_config_path=args.pose2d_config,
            pose2d_checkpoint_path=args.pose2d_checkpoint,
            motionbert_repo=args.motionbert_repo,
            pose3d_config_path=args.pose3d_config,
            pose3d_checkpoint_path=args.pose3d_checkpoint,
            mesh_config_path=args.mesh_config,
            mesh_checkpoint_path=args.mesh_checkpoint,
            smpl_data_root=args.smpl_data_root,
            mesh_clip_len=args.mesh_clip_len,
            mesh_clip_stride=args.mesh_clip_stride,
            device=args.device,
        )
    )

    result = pipeline.run()
    print(result.summary())


if __name__ == "__main__":
    main()
