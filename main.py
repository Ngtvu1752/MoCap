from __future__ import annotations

import argparse
from pathlib import Path

from src.mesh.motionbert_mesh_estimator import MotionBERTMeshEstimator
from src.mesh.smpl_mesh_renderer import SMPLMeshRenderer
from src.pipeline import MocapPipeline, PipelineConfig
from src.pose2d.rtmpose_estimator import RTMPoseEstimator
from src.pose3d.adapters import KeypointFormat, Pose2DFormatConverter
from src.pose3d.motionbert_estimator import MotionBERTEstimator
from src.renderer.mesh_renderer import SkeletonRenderer


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
        choices=["whole_body133", "coco_body17", "halpe26"],
        default="whole_body133",
        help="RTMPose keypoint layout emitted by the 2D model",
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
        "--pose3d-only",
        action="store_true",
        help="Run 2D pose + 3D lifting, then stop before rendering/HMR.",
    )
    parser.add_argument(
        "--motionbert-repo",
        type=Path,
        default=Path("MotionBERT"),
        help="MotionBERT repository path",
    )
    parser.add_argument(
        "--pose3d-config",
        type=Path,
        default=Path("MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml"),
        help="MotionBERT pose3d config file path",
    )
    parser.add_argument(
        "--pose3d-checkpoint",
        type=Path,
        default=Path("checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"),
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
        default=Path("MotionBERT/configs/mesh/MB_ft_pw3d.yaml"),
        help="MotionBERT mesh config file path",
    )
    parser.add_argument(
        "--mesh-checkpoint",
        type=Path,
        default=Path("checkpoints/MotionBERT/FT_MB_release_MB_ft_pw3d/best_epoch.bin"),
        help="MotionBERT mesh checkpoint .bin path",
    )
    parser.add_argument(
        "--smpl-data-root",
        type=Path,
        default=Path("checkpoints/Mesh"),
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
        default=None,
        help="Optional override for MotionBERT mesh clip length.",
    )
    parser.add_argument(
        "--mesh-clip-stride",
        type=int,
        default=None,
        help=(
            "Number of overlapping frames between adjacent MotionBERT mesh clips. "
            "For example, with --mesh-clip-len 243 and --mesh-clip-stride 121, "
            "adjacent clips start 122 frames apart."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose2d_format = KeypointFormat(args.pose2d_format)

    human_mesh_estimator = None
    human_mesh_renderer = None
    if args.human_mesh:
        human_mesh_estimator = MotionBERTMeshEstimator(
            repo_path=args.motionbert_repo,
            config_path=args.mesh_config,
            checkpoint_path=args.mesh_checkpoint,
            smpl_data_root=args.smpl_data_root,
            source_format=pose2d_format,
            device=args.device,
            clip_len=args.mesh_clip_len,
            clip_stride=args.mesh_clip_stride,
        )
        if not args.skip_human_mesh_render:
            human_mesh_renderer = SMPLMeshRenderer()

    pipeline = MocapPipeline(
        config=PipelineConfig(
            video_path=args.video,
            output_dir=args.output_dir,
            metadata_only=args.metadata_only,
            pose2d_only=args.pose2d_only,
            pose3d_only=args.pose3d_only,
            human_mesh=args.human_mesh,
        ),
        pose2d_estimator=RTMPoseEstimator(
            config_path=args.pose2d_config,
            checkpoint_path=args.pose2d_checkpoint,
            device=args.device,
        ),
        pose_converter=Pose2DFormatConverter(
            source_format=pose2d_format,
            target_format=KeypointFormat.HUMAN36M_17,
        ),
        pose3d_estimator=MotionBERTEstimator(
            repo_path=args.motionbert_repo,
            config_path=args.pose3d_config,
            checkpoint_path=args.pose3d_checkpoint,
            device=args.device,
        ),
        renderer=SkeletonRenderer(),
        human_mesh_estimator=human_mesh_estimator,
        human_mesh_renderer=human_mesh_renderer,
    )

    result = pipeline.run()
    print(result.summary())


if __name__ == "__main__":
    main()
