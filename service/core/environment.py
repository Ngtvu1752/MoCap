from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from service.config import ServiceConfig


SMPL_REQUIRED_FILES = (
    "SMPL_NEUTRAL.pkl",
    "J_regressor_h36m_correct.npy",
    "J_regressor_extra.npy",
    "smpl_mean_params.npz",
)


def required_asset_paths(config: ServiceConfig) -> list[Path]:
    smpl_root = config.resolve(config.smpl_data_root)
    return [
        config.resolve(config.pose2d_config),
        config.resolve(config.pose2d_checkpoint),
        config.resolve(config.motionbert_repo),
        config.resolve(config.pose3d_config),
        config.resolve(config.pose3d_checkpoint),
        config.resolve(config.mesh_config),
        config.resolve(config.mesh_checkpoint),
        config.resolve(config.base_fbx),
        *(smpl_root / name for name in SMPL_REQUIRED_FILES),
    ]


def missing_asset_paths(config: ServiceConfig) -> list[Path]:
    return [path for path in required_asset_paths(config) if not path.exists()]


def check_runtime_environment(config: ServiceConfig) -> list[str]:
    errors: list[str] = []
    missing = missing_asset_paths(config)
    if missing:
        errors.append("Missing required asset(s): " + ", ".join(str(path) for path in missing))
    if shutil.which("ffprobe") is None:
        errors.append("ffprobe was not found in PATH.")
    if shutil.which(config.blender_executable) is None and not Path(config.blender_executable).exists():
        errors.append(
            f"Blender executable {config.blender_executable!r} was not found. "
            "Set MOCAP_BLENDER=/absolute/path/to/blender."
        )
    for package in ("fastapi", "uvicorn", "redis", "rq", "multipart"):
        if importlib.util.find_spec(package) is None:
            errors.append(f"Python package {package!r} is not installed.")
    if config.device.startswith("cuda"):
        if importlib.util.find_spec("torch") is None:
            errors.append("PyTorch is not installed, cannot use CUDA device.")
        else:
            import torch

            if not torch.cuda.is_available():
                errors.append(f"CUDA device requested ({config.device}) but torch.cuda.is_available() is false.")
    return errors

