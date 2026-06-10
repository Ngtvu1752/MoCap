from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.pose3d.base import Pose3DEstimator, Pose3DResult


DEFAULT_REPO_PATH = Path("MotionBERT")
DEFAULT_CONFIG_PATH = Path("MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml")
DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
)


class MotionBERTEstimator(Pose3DEstimator):
    """MotionBERT 2D-to-3D pose lifter for Human3.6M 17-joint input."""

    def __init__(
        self,
        repo_path: Path | str = DEFAULT_REPO_PATH,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda:0",
        clip_len: int | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.clip_len = clip_len

        self._torch: Any | None = None
        self._args: Any | None = None
        self._model: Any | None = None
        self._crop_scale: Any | None = None
        self._flip_data: Any | None = None

    def lift(self, pose2d: np.ndarray) -> Pose3DResult:
        pose2d = np.asarray(pose2d, dtype=np.float32)
        self._validate_pose2d(pose2d)
        self._ensure_model_loaded()

        motion = self._with_confidence(pose2d)
        motion = self._crop_scale(motion, scale_range=[1, 1]).astype(np.float32)

        clip_len = self.clip_len or int(self._args.clip_len)
        outputs = []
        with self._torch.no_grad():
            for clip in self._iter_clips(motion, clip_len):
                batch_input = self._torch.from_numpy(clip[None]).to(self._device())

                if self._args.no_conf:
                    batch_input = batch_input[:, :, :, :2]

                if self._args.flip:
                    batch_input_flip = self._flip_data(batch_input)
                    pred = self._model(batch_input)
                    pred_flip = self._model(batch_input_flip)
                    pred_flip = self._flip_data(pred_flip)
                    predicted_3d = (pred + pred_flip) / 2.0
                else:
                    predicted_3d = self._model(batch_input)

                if self._args.rootrel:
                    predicted_3d[:, :, 0, :] = 0
                else:
                    predicted_3d[:, 0, 0, 2] = 0

                if self._args.gt_2d:
                    predicted_3d[..., :2] = batch_input[..., :2]

                outputs.append(predicted_3d.detach().cpu().numpy()[0])

        pose3d = np.concatenate(outputs, axis=0).astype(np.float32)
        return Pose3DResult(
            keypoints=pose3d,
            metadata={
                "backend": "motionbert",
                "repo_path": str(self.repo_path),
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "device": str(self._device()),
                "clip_len": clip_len,
                "output_space": "motionbert_normalized",
            },
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.repo_path.exists():
            raise FileNotFoundError(f"MotionBERT repo does not exist: {self.repo_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"MotionBERT config does not exist: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MotionBERT checkpoint does not exist: {self.checkpoint_path}")

        repo_path = str(self.repo_path.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            import torch
            from lib.utils.learning import load_backbone
            from lib.utils.tools import get_config
            from lib.utils.utils_data import crop_scale, flip_data
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MotionBERT dependencies are missing. Install the minimal MotionBERT "
                "dependencies in the current environment before running Phase 3."
            ) from exc

        self._torch = torch
        self._crop_scale = crop_scale
        self._flip_data = flip_data
        self._args = get_config(str(self.config_path))

        model = load_backbone(self._args)
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model_pos"] if "model_pos" in checkpoint else checkpoint
        state_dict = self._strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device())
        model.eval()
        self._model = model

    def _device(self) -> Any:
        torch = self._torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _with_confidence(self, pose2d: np.ndarray) -> np.ndarray:
        confidence = (~np.all(np.isclose(pose2d, 0.0), axis=-1)).astype(np.float32)
        return np.concatenate([pose2d, confidence[..., None]], axis=-1)

    def _iter_clips(self, motion: np.ndarray, clip_len: int):
        for start in range(0, len(motion), clip_len):
            yield motion[start : start + clip_len]

    def _validate_pose2d(self, pose2d: np.ndarray) -> None:
        if pose2d.ndim != 3:
            raise ValueError(f"Expected pose2d shape (T, 17, 2), got {pose2d.shape}")
        if pose2d.shape[1] != 17:
            raise ValueError(f"Expected 17 Human3.6M joints, got {pose2d.shape[1]}")
        if pose2d.shape[2] != 2:
            raise ValueError(f"Expected xy coordinates in last dimension, got {pose2d.shape}")
        if pose2d.shape[0] == 0:
            raise ValueError("Expected at least one frame for MotionBERT inference")

    def _strip_module_prefix(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not state_dict:
            return state_dict
        if all(key.startswith("module.") for key in state_dict.keys()):
            return {key[7:]: value for key, value in state_dict.items()}
        return state_dict
