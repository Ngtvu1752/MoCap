from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.io.video_reader import VideoMetadata, VideoReader
from src.pose2d.base import Pose2DEstimator
from src.pose3d.adapters import Pose2DFormatConverter
from src.pose3d.base import Pose3DEstimator
from src.renderer.base import PoseRenderer


@dataclass(frozen=True)
class PipelineConfig:
    video_path: Path
    output_dir: Path = Path("output")
    metadata_only: bool = False
    pose2d_only: bool = False


@dataclass(frozen=True)
class PipelineResult:
    metadata: VideoMetadata
    pose2d_path: Path | None = None
    pose3d_path: Path | None = None
    mesh_video_path: Path | None = None

    def summary(self) -> str:
        lines = [
            f"Video: {self.metadata.path}",
            f"Resolution: {self.metadata.width}x{self.metadata.height}",
            f"FPS: {self.metadata.fps:.2f}",
            f"Frames: {self.metadata.frame_count}",
            f"Duration: {self.metadata.duration_seconds:.2f}s",
        ]

        if self.pose2d_path is not None:
            lines.append(f"2D pose: {self.pose2d_path}")
        if self.pose3d_path is not None:
            lines.append(f"3D pose: {self.pose3d_path}")
        if self.mesh_video_path is not None:
            lines.append(f"Rendered video: {self.mesh_video_path}")

        return "\n".join(lines)


class MocapPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        pose2d_estimator: Pose2DEstimator,
        pose_converter: Pose2DFormatConverter,
        pose3d_estimator: Pose3DEstimator,
        renderer: PoseRenderer,
    ) -> None:
        self.config = config
        self.pose2d_estimator = pose2d_estimator
        self.pose_converter = pose_converter
        self.pose3d_estimator = pose3d_estimator
        self.renderer = renderer

    def run(self) -> PipelineResult:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        with VideoReader(self.config.video_path) as reader:
            metadata = reader.metadata
            if self.config.metadata_only:
                return PipelineResult(metadata=metadata)

            pose2d_sequence = []
            for frame in reader.iter_frames():
                result = self.pose2d_estimator.estimate_frame(frame.image, frame.index)
                pose2d_sequence.append(result.keypoints)

        pose2d = np.stack(pose2d_sequence, axis=0)
        pose2d_path = self.config.output_dir / "pose2d.npy"
        np.save(pose2d_path, pose2d)

        if self.config.pose2d_only:
            return PipelineResult(metadata=metadata, pose2d_path=pose2d_path)

        model_pose2d = self.pose_converter.convert(pose2d)
        pose3d_result = self.pose3d_estimator.lift(model_pose2d)
        pose3d_path = self.config.output_dir / "pose3d.npy"
        np.save(pose3d_path, pose3d_result.keypoints)

        mesh_video_path = self.config.output_dir / "mesh.mp4"
        self.renderer.render(pose3d_result.keypoints, mesh_video_path, metadata.fps)

        return PipelineResult(
            metadata=metadata,
            pose2d_path=pose2d_path,
            pose3d_path=pose3d_path,
            mesh_video_path=mesh_video_path,
        )
