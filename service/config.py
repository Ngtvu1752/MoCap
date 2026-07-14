from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.pipeline_defaults import (
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


_ENV_LOADED = False


def load_env_file(path=None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = path or Path(".env")
    if not env_path.exists():
        _ENV_LOADED = True
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"\x27")
        os.environ.setdefault(key, value)
    _ENV_LOADED = True


def _path_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class ServiceConfig:
    repo_root: Path = Path(".")
    storage_root: Path = Path("service_data")
    database_path: Path = Path("service_data/jobs.sqlite3")
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "mocap-fbx"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-internal"
    device: str = "cuda:0"
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_duration_seconds: float = 300.0
    max_width: int = 1920
    max_height: int = 1080
    retention_days: int = 7
    validation_timeout_seconds: int = 120
    job_timeout_seconds: int = 3600
    blender_executable: str = "blender"
    base_fbx: Path = Path("assets/retarget/smpl_base_tpose.fbx")
    pose2d_format: str = DEFAULT_POSE2D_FORMAT
    pose2d_config: Path = DEFAULT_POSE2D_CONFIG
    pose2d_checkpoint: Path = DEFAULT_POSE2D_CHECKPOINT
    motionbert_repo: Path = DEFAULT_MOTIONBERT_REPO
    pose3d_config: Path = DEFAULT_POSE3D_CONFIG
    pose3d_checkpoint: Path = DEFAULT_POSE3D_CHECKPOINT
    mesh_config: Path = DEFAULT_MESH_CONFIG
    mesh_checkpoint: Path = DEFAULT_MESH_CHECKPOINT
    smpl_data_root: Path = DEFAULT_SMPL_DATA_ROOT
    mesh_clip_len: int = DEFAULT_MESH_CLIP_LEN
    mesh_clip_stride: int = DEFAULT_MESH_CLIP_STRIDE
    root_trajectory: str = "pose3d"
    root_scale: float = 0.1

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        load_env_file()
        repo_root = _path_env("MOCAP_REPO_ROOT", ".")
        storage_root = _path_env("MOCAP_SERVICE_STORAGE", "service_data")
        return cls(
            repo_root=repo_root,
            storage_root=storage_root,
            database_path=_path_env("MOCAP_SERVICE_DB", str(storage_root / "jobs.sqlite3")),
            redis_url=os.environ.get("MOCAP_REDIS_URL", "redis://localhost:6379/0"),
            queue_name=os.environ.get("MOCAP_QUEUE_NAME", "mocap-fbx"),
            api_host=os.environ.get("MOCAP_API_HOST", "0.0.0.0"),
            api_port=_int_env("MOCAP_API_PORT", 8000),
            api_key=os.environ.get("MOCAP_SERVICE_API_KEY", "change-me-internal"),
            device=os.environ.get("MOCAP_DEVICE", "cuda:0"),
            max_upload_bytes=_int_env("MOCAP_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024),
            max_duration_seconds=_float_env("MOCAP_MAX_DURATION_SECONDS", 300.0),
            max_width=_int_env("MOCAP_MAX_WIDTH", 1920),
            max_height=_int_env("MOCAP_MAX_HEIGHT", 1080),
            retention_days=_int_env("MOCAP_RETENTION_DAYS", 7),
            validation_timeout_seconds=_int_env("MOCAP_VALIDATION_TIMEOUT_SECONDS", 120),
            job_timeout_seconds=_int_env("MOCAP_JOB_TIMEOUT_SECONDS", 3600),
            blender_executable=os.environ.get("MOCAP_BLENDER", "blender"),
            base_fbx=_path_env("MOCAP_BASE_FBX", "assets/retarget/smpl_base_tpose.fbx"),
            pose2d_format=os.environ.get("MOCAP_POSE2D_FORMAT", DEFAULT_POSE2D_FORMAT),
            pose2d_config=_path_env("MOCAP_POSE2D_CONFIG", str(DEFAULT_POSE2D_CONFIG)),
            pose2d_checkpoint=_path_env("MOCAP_POSE2D_CHECKPOINT", str(DEFAULT_POSE2D_CHECKPOINT)),
            motionbert_repo=_path_env("MOCAP_MOTIONBERT_REPO", str(DEFAULT_MOTIONBERT_REPO)),
            pose3d_config=_path_env("MOCAP_POSE3D_CONFIG", str(DEFAULT_POSE3D_CONFIG)),
            pose3d_checkpoint=_path_env("MOCAP_POSE3D_CHECKPOINT", str(DEFAULT_POSE3D_CHECKPOINT)),
            mesh_config=_path_env("MOCAP_MESH_CONFIG", str(DEFAULT_MESH_CONFIG)),
            mesh_checkpoint=_path_env("MOCAP_MESH_CHECKPOINT", str(DEFAULT_MESH_CHECKPOINT)),
            smpl_data_root=_path_env("MOCAP_SMPL_DATA_ROOT", str(DEFAULT_SMPL_DATA_ROOT)),
            mesh_clip_len=_int_env("MOCAP_MESH_CLIP_LEN", DEFAULT_MESH_CLIP_LEN),
            mesh_clip_stride=_int_env("MOCAP_MESH_CLIP_STRIDE", DEFAULT_MESH_CLIP_STRIDE),
            root_trajectory=os.environ.get("MOCAP_ROOT_TRAJECTORY", "pose3d"),
            root_scale=_float_env("MOCAP_ROOT_SCALE", 0.1),
        )

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()


def get_config() -> ServiceConfig:
    return ServiceConfig.from_env()

