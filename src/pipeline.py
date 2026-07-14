from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.io.video_reader import VideoMetadata, VideoReader
from src.mesh.base import HumanMeshEstimator, MeshResult
from src.mesh.smpl_mesh_renderer import SMPLMeshRenderer
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
)
from src.pose2d.base import Pose2DEstimator
from src.pose3d.adapters import Pose2DFormatConverter
from src.pose3d.base import Pose3DEstimator, Pose3DResult
from src.renderer.base import PoseRenderer, RenderResult


@dataclass(frozen=True)
class PipelineConfig:
    video_path: Path
    output_dir: Path = Path("output")
    metadata_only: bool = False
    pose2d_only: bool = False
    pose3d_only: bool = False
    human_mesh: bool = False
    render_human_mesh: bool = True
    pose2d_format: str = DEFAULT_POSE2D_FORMAT
    pose2d_config_path: Path = DEFAULT_POSE2D_CONFIG
    pose2d_checkpoint_path: Path = DEFAULT_POSE2D_CHECKPOINT
    motionbert_repo: Path = DEFAULT_MOTIONBERT_REPO
    pose3d_config_path: Path = DEFAULT_POSE3D_CONFIG
    pose3d_checkpoint_path: Path = DEFAULT_POSE3D_CHECKPOINT
    mesh_config_path: Path = DEFAULT_MESH_CONFIG
    mesh_checkpoint_path: Path = DEFAULT_MESH_CHECKPOINT
    smpl_data_root: Path = DEFAULT_SMPL_DATA_ROOT
    mesh_clip_len: int | None = DEFAULT_MESH_CLIP_LEN
    mesh_clip_stride: int | None = DEFAULT_MESH_CLIP_STRIDE
    device: str = DEFAULT_DEVICE


@dataclass(frozen=True)
class Pose2DStageResult:
    keypoints: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any] | None
    pose2d_path: Path
    pose2d_scores_path: Path


@dataclass(frozen=True)
class Pose3DStageResult:
    result: Pose3DResult
    pose3d_path: Path


@dataclass(frozen=True)
class HumanMeshStageResult:
    mesh_result: MeshResult | None = None
    smpl_vertices_path: Path | None = None
    smpl_joints3d_path: Path | None = None
    smpl_theta_path: Path | None = None
    human_mesh_video_path: Path | None = None


@dataclass(frozen=True)
class PipelineResult:
    metadata: VideoMetadata
    metadata_path: Path | None = None
    pose2d_path: Path | None = None
    pose2d_scores_path: Path | None = None
    pose3d_path: Path | None = None
    skeleton_video_path: Path | None = None
    smpl_vertices_path: Path | None = None
    smpl_joints3d_path: Path | None = None
    smpl_theta_path: Path | None = None
    human_mesh_video_path: Path | None = None

    def summary(self) -> str:
        lines = [
            f"Video: {self.metadata.path}",
            f"Resolution: {self.metadata.width}x{self.metadata.height}",
            f"FPS: {self.metadata.fps:.2f}",
            f"Frames: {self.metadata.frame_count}",
            f"Duration: {self.metadata.duration_seconds:.2f}s",
        ]

        if self.metadata_path is not None:
            lines.append(f"Metadata: {self.metadata_path}")
        if self.pose2d_path is not None:
            lines.append(f"2D pose: {self.pose2d_path}")
        if self.pose2d_scores_path is not None:
            lines.append(f"2D pose scores: {self.pose2d_scores_path}")
        if self.pose3d_path is not None:
            lines.append(f"3D pose: {self.pose3d_path}")
        if self.skeleton_video_path is not None:
            lines.append(f"Skeleton video: {self.skeleton_video_path}")
        if self.smpl_vertices_path is not None:
            lines.append(f"SMPL vertices: {self.smpl_vertices_path}")
        if self.smpl_joints3d_path is not None:
            lines.append(f"SMPL joints3d: {self.smpl_joints3d_path}")
        if self.smpl_theta_path is not None:
            lines.append(f"SMPL theta: {self.smpl_theta_path}")
        if self.human_mesh_video_path is not None:
            lines.append(f"Human mesh video: {self.human_mesh_video_path}")

        return "\n".join(lines)


class MocapPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        pose2d_estimator: Pose2DEstimator,
        pose_converter: Pose2DFormatConverter,
        pose3d_estimator: Pose3DEstimator,
        renderer: PoseRenderer,
        human_mesh_estimator: HumanMeshEstimator | None = None,
        human_mesh_renderer: SMPLMeshRenderer | None = None,
    ) -> None:
        self.config = config
        self.pose2d_estimator = pose2d_estimator
        self.pose_converter = pose_converter
        self.pose3d_estimator = pose3d_estimator
        self.renderer = renderer
        self.human_mesh_estimator = human_mesh_estimator
        self.human_mesh_renderer = human_mesh_renderer

    def run(self) -> PipelineResult:
        run_output_dir = self.prepare_output_dir()
        metadata = self.read_metadata()
        if self.config.metadata_only:
            metadata_path = self.write_metadata(run_output_dir, metadata, stage="metadata_only")
            return PipelineResult(metadata=metadata, metadata_path=metadata_path)

        pose2d_stage = self.run_pose2d_stage(run_output_dir)
        if self.config.pose2d_only:
            metadata_path = self.write_metadata(
                run_output_dir,
                metadata,
                stage="pose2d_only",
                pose2d_stage=pose2d_stage,
            )
            return PipelineResult(
                metadata=metadata,
                metadata_path=metadata_path,
                pose2d_path=pose2d_stage.pose2d_path,
                pose2d_scores_path=pose2d_stage.pose2d_scores_path,
            )

        pose3d_stage = self.run_pose3d_stage(run_output_dir, pose2d_stage)
        if self.config.pose3d_only:
            metadata_path = self.write_metadata(
                run_output_dir,
                metadata,
                stage="pose3d_only",
                pose2d_stage=pose2d_stage,
                pose3d_stage=pose3d_stage,
            )
            return PipelineResult(
                metadata=metadata,
                metadata_path=metadata_path,
                pose2d_path=pose2d_stage.pose2d_path,
                pose2d_scores_path=pose2d_stage.pose2d_scores_path,
                pose3d_path=pose3d_stage.pose3d_path,
            )

        render_result = self.run_render_stage(run_output_dir, pose3d_stage, metadata)
        mesh_stage = self.run_human_mesh_stage(run_output_dir, pose2d_stage, metadata)
        metadata_path = self.write_metadata(
            run_output_dir,
            metadata,
            stage="complete_with_human_mesh" if self.config.human_mesh else "complete",
            pose2d_stage=pose2d_stage,
            pose3d_stage=pose3d_stage,
            render_result=render_result,
            mesh_stage=mesh_stage,
        )

        return PipelineResult(
            metadata=metadata,
            metadata_path=metadata_path,
            pose2d_path=pose2d_stage.pose2d_path,
            pose2d_scores_path=pose2d_stage.pose2d_scores_path,
            pose3d_path=pose3d_stage.pose3d_path,
            skeleton_video_path=render_result.skeleton_video_path,
            smpl_vertices_path=mesh_stage.smpl_vertices_path,
            smpl_joints3d_path=mesh_stage.smpl_joints3d_path,
            smpl_theta_path=mesh_stage.smpl_theta_path,
            human_mesh_video_path=mesh_stage.human_mesh_video_path,
        )

    def prepare_output_dir(self) -> Path:
        run_output_dir = self._run_output_dir()
        run_output_dir.mkdir(parents=True, exist_ok=True)
        return run_output_dir

    def read_metadata(self) -> VideoMetadata:
        with VideoReader(self.config.video_path) as reader:
            return reader.metadata

    def run_pose2d_stage(self, output_dir: Path) -> Pose2DStageResult:
        pose2d_sequence = []
        pose2d_scores_sequence = []
        pose2d_metadata: dict[str, Any] | None = None
        with VideoReader(self.config.video_path) as reader:
            for frame in reader.iter_frames():
                result = self.pose2d_estimator.estimate_frame(frame.image, frame.index)
                pose2d_sequence.append(result.keypoints)
                if result.scores is not None:
                    pose2d_scores_sequence.append(result.scores)
                else:
                    pose2d_scores_sequence.append(np.ones(result.keypoints.shape[0], dtype=np.float32))
                if pose2d_metadata is None:
                    pose2d_metadata = result.metadata

        if not pose2d_sequence:
            raise RuntimeError(f"No frames were decoded from {self.config.video_path}.")

        pose2d = np.stack(pose2d_sequence, axis=0)
        pose2d_path = output_dir / "pose2d.npy"
        np.save(pose2d_path, pose2d)

        pose2d_scores = np.stack(pose2d_scores_sequence, axis=0)
        pose2d_scores_path = output_dir / "pose2d_scores.npy"
        np.save(pose2d_scores_path, pose2d_scores)

        return Pose2DStageResult(
            keypoints=pose2d,
            scores=pose2d_scores,
            metadata=pose2d_metadata,
            pose2d_path=pose2d_path,
            pose2d_scores_path=pose2d_scores_path,
        )

    def run_pose3d_stage(self, output_dir: Path, pose2d_stage: Pose2DStageResult) -> Pose3DStageResult:
        model_pose2d = self.pose_converter.convert(pose2d_stage.keypoints)
        pose3d_result = self.pose3d_estimator.lift(model_pose2d)
        pose3d_path = output_dir / "pose3d.npy"
        np.save(pose3d_path, pose3d_result.keypoints)
        return Pose3DStageResult(result=pose3d_result, pose3d_path=pose3d_path)

    def run_render_stage(
        self,
        output_dir: Path,
        pose3d_stage: Pose3DStageResult,
        metadata: VideoMetadata,
    ) -> RenderResult:
        return self.renderer.render(pose3d_stage.result.keypoints, output_dir / "mesh.mp4", metadata.fps)

    def run_human_mesh_stage(
        self,
        output_dir: Path,
        pose2d_stage: Pose2DStageResult,
        metadata: VideoMetadata,
    ) -> HumanMeshStageResult:
        if not self.config.human_mesh:
            return HumanMeshStageResult()
        if self.human_mesh_estimator is None:
            raise RuntimeError("Human mesh recovery was requested but no mesh estimator was configured.")

        mesh_result = self.human_mesh_estimator.recover(pose2d_stage.keypoints, pose2d_stage.scores)
        smpl_vertices_path = output_dir / "smpl_vertices.npy"
        smpl_joints3d_path = output_dir / "smpl_joints3d.npy"
        smpl_theta_path = output_dir / "smpl_theta.npy"
        np.save(smpl_vertices_path, mesh_result.vertices)
        np.save(smpl_joints3d_path, mesh_result.joints3d)
        np.save(smpl_theta_path, mesh_result.theta)

        human_mesh_video_path: Path | None = None
        if self.human_mesh_renderer is not None:
            human_mesh_video_path = self.human_mesh_renderer.render(
                mesh_result.vertices,
                mesh_result.faces,
                output_dir / "human_mesh.mp4",
                fps=metadata.fps,
            )

        return HumanMeshStageResult(
            mesh_result=mesh_result,
            smpl_vertices_path=smpl_vertices_path,
            smpl_joints3d_path=smpl_joints3d_path,
            smpl_theta_path=smpl_theta_path,
            human_mesh_video_path=human_mesh_video_path,
        )

    def write_metadata(
        self,
        output_dir: Path,
        video_metadata: VideoMetadata,
        stage: str,
        pose2d_stage: Pose2DStageResult | None = None,
        pose3d_stage: Pose3DStageResult | None = None,
        render_result: RenderResult | None = None,
        mesh_stage: HumanMeshStageResult | None = None,
    ) -> Path:
        metadata_path = output_dir / "metadata.json"
        outputs: dict[str, str] = {"metadata": str(metadata_path)}
        if pose2d_stage is not None:
            outputs["pose2d"] = str(pose2d_stage.pose2d_path)
            outputs["pose2d_scores"] = str(pose2d_stage.pose2d_scores_path)
        if pose3d_stage is not None:
            outputs["pose3d"] = str(pose3d_stage.pose3d_path)
        if render_result is not None:
            outputs["skeleton_video"] = str(render_result.skeleton_video_path)
        if mesh_stage is not None:
            if mesh_stage.smpl_vertices_path is not None:
                outputs["smpl_vertices"] = str(mesh_stage.smpl_vertices_path)
            if mesh_stage.smpl_joints3d_path is not None:
                outputs["smpl_joints3d"] = str(mesh_stage.smpl_joints3d_path)
            if mesh_stage.smpl_theta_path is not None:
                outputs["smpl_theta"] = str(mesh_stage.smpl_theta_path)
            if mesh_stage.human_mesh_video_path is not None:
                outputs["human_mesh_video"] = str(mesh_stage.human_mesh_video_path)

        payload = {
            "video": {
                "path": str(video_metadata.path),
                "width": video_metadata.width,
                "height": video_metadata.height,
                "fps": video_metadata.fps,
                "frame_count": video_metadata.frame_count,
                "duration_seconds": video_metadata.duration_seconds,
            },
            "pipeline": {
                "stage": stage,
                "output_dir": str(output_dir),
                "metadata_only": self.config.metadata_only,
                "pose2d_only": self.config.pose2d_only,
                "pose3d_only": self.config.pose3d_only,
                "human_mesh": self.config.human_mesh,
                "render_human_mesh": self.config.render_human_mesh,
                "pose2d_format": self.config.pose2d_format,
                "pose2d_config_path": str(self.config.pose2d_config_path),
                "pose2d_checkpoint_path": str(self.config.pose2d_checkpoint_path),
                "motionbert_repo": str(self.config.motionbert_repo),
                "pose3d_config_path": str(self.config.pose3d_config_path),
                "pose3d_checkpoint_path": str(self.config.pose3d_checkpoint_path),
                "mesh_config_path": str(self.config.mesh_config_path),
                "mesh_checkpoint_path": str(self.config.mesh_checkpoint_path),
                "smpl_data_root": str(self.config.smpl_data_root),
                "mesh_clip_len": self.config.mesh_clip_len,
                "mesh_clip_stride": self.config.mesh_clip_stride,
                "device": self.config.device,
            },
            "outputs": outputs,
            "pose2d": self._json_safe(pose2d_stage.metadata if pose2d_stage is not None else {}),
            "pose3d": self._json_safe(pose3d_stage.result.metadata if pose3d_stage is not None else {}),
            "human_mesh": self._json_safe(
                mesh_stage.mesh_result.metadata
                if mesh_stage is not None and mesh_stage.mesh_result is not None
                else {}
            ),
        }
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata_path

    def _run_output_dir(self) -> Path:
        return self.config.output_dir / self.config.video_path.stem

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value
