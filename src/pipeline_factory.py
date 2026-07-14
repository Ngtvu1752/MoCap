from __future__ import annotations

from src.mesh.motionbert_mesh_estimator import MotionBERTMeshEstimator
from src.mesh.smpl_mesh_renderer import SMPLMeshRenderer
from src.pipeline import MocapPipeline, PipelineConfig
from src.pose2d.rtmpose_estimator import RTMPoseEstimator
from src.pose3d.adapters import KeypointFormat, Pose2DFormatConverter
from src.pose3d.motionbert_estimator import MotionBERTEstimator
from src.renderer.mesh_renderer import SkeletonRenderer


def create_mocap_pipeline(config: PipelineConfig) -> MocapPipeline:
    pose2d_format = KeypointFormat(config.pose2d_format)
    human_mesh_estimator = None
    human_mesh_renderer = None

    if config.human_mesh:
        human_mesh_estimator = MotionBERTMeshEstimator(
            repo_path=config.motionbert_repo,
            config_path=config.mesh_config_path,
            checkpoint_path=config.mesh_checkpoint_path,
            smpl_data_root=config.smpl_data_root,
            source_format=pose2d_format,
            device=config.device,
            clip_len=config.mesh_clip_len,
            clip_stride=config.mesh_clip_stride,
        )
        if config.render_human_mesh:
            human_mesh_renderer = SMPLMeshRenderer()

    return MocapPipeline(
        config=config,
        pose2d_estimator=RTMPoseEstimator(
            config_path=config.pose2d_config_path,
            checkpoint_path=config.pose2d_checkpoint_path,
            device=config.device,
        ),
        pose_converter=Pose2DFormatConverter(
            source_format=pose2d_format,
            target_format=KeypointFormat.HUMAN36M_17,
        ),
        pose3d_estimator=MotionBERTEstimator(
            repo_path=config.motionbert_repo,
            config_path=config.pose3d_config_path,
            checkpoint_path=config.pose3d_checkpoint_path,
            device=config.device,
        ),
        renderer=SkeletonRenderer(),
        human_mesh_estimator=human_mesh_estimator,
        human_mesh_renderer=human_mesh_renderer,
    )
