from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from service.config import ServiceConfig
from service.core.hashing import sha256_file
from service.core.models import JobRecord, utc_now_iso


def artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_report(
    job: JobRecord,
    config: ServiceConfig,
    *,
    input_metadata: dict[str, Any] | None = None,
    stage_timings: dict[str, float] | None = None,
    artifacts: dict[str, Path] | None = None,
    root_trajectory: dict[str, Any] | None = None,
    error: str | None = None,
    log_tail: str | None = None,
) -> dict[str, Any]:
    artifact_payload = {}
    for name, path in (artifacts or {}).items():
        if path.exists():
            artifact_payload[name] = artifact_info(path)
    input_payload: dict[str, Any] = input_metadata.copy() if input_metadata else {}
    if job.input_path and Path(job.input_path).exists():
        input_payload["source"] = artifact_info(Path(job.input_path))
    return {
        "job": job.to_dict(),
        "generated_at": utc_now_iso(),
        "input": input_payload,
        "service": {
            "profile": job.request.get("profile"),
            "device": config.device,
            "pose2d_format": job.request.get("pose2d_format"),
            "mesh_clip_len": job.request.get("mesh_clip_len"),
            "mesh_clip_stride": job.request.get("mesh_clip_stride"),
            "root_trajectory": job.request.get("root_trajectory"),
        },
        "models": {
            "pose2d_config": str(config.resolve(config.pose2d_config)),
            "pose2d_checkpoint": str(config.resolve(config.pose2d_checkpoint)),
            "pose3d_config": str(config.resolve(config.pose3d_config)),
            "pose3d_checkpoint": str(config.resolve(config.pose3d_checkpoint)),
            "mesh_config": str(config.resolve(config.mesh_config)),
            "mesh_checkpoint": str(config.resolve(config.mesh_checkpoint)),
            "motionbert_repo": str(config.resolve(config.motionbert_repo)),
            "smpl_data_root": str(config.resolve(config.smpl_data_root)),
            "base_fbx": str(config.resolve(config.base_fbx)),
        },
        "stage_timings_seconds": stage_timings or {},
        "root_trajectory": root_trajectory or {},
        "artifacts": artifact_payload,
        "error": error,
        "log_tail": log_tail,
    }


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path

