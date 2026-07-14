from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.pipeline_defaults import DEFAULT_MESH_CLIP_LEN, DEFAULT_MESH_CLIP_STRIDE, DEFAULT_POSE2D_FORMAT


JOB_STATUSES = (
    "queued",
    "validating",
    "pose2d",
    "pose3d",
    "smpl",
    "fbx_export",
    "fbx_preview",
    "packaging",
    "done",
    "failed",
    "expired",
    "canceled",
)
TERMINAL_STATUSES = ("done", "failed", "expired", "canceled")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_status(status: str) -> str:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown job status {status!r}.")
    return status


@dataclass(frozen=True)
class JobRequest:
    profile: str = "unity_humanoid_fbx"
    pose2d_format: str = DEFAULT_POSE2D_FORMAT
    mesh_clip_len: int = DEFAULT_MESH_CLIP_LEN
    mesh_clip_stride: int = DEFAULT_MESH_CLIP_STRIDE
    root_trajectory: str = "pose3d"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "pose2d_format": self.pose2d_format,
            "mesh_clip_len": self.mesh_clip_len,
            "mesh_clip_stride": self.mesh_clip_stride,
            "root_trajectory": self.root_trajectory,
        }


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    input_path: str | None = None
    output_fbx_path: str | None = None
    report_path: str | None = None
    preview_path: str | None = None
    error: str | None = None
    rq_job_id: str | None = None
    request: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "input_path": self.input_path,
            "output_fbx_path": self.output_fbx_path,
            "report_path": self.report_path,
            "preview_path": self.preview_path,
            "error": self.error,
            "rq_job_id": self.rq_job_id,
            "request": self.request,
        }

