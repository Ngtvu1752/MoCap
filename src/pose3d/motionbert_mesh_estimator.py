from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REPO_PATH = Path("MotionBERT")
DEFAULT_CONFIG_PATH = Path("MotionBERT/configs/mesh/MB_ft_pw3d.yaml")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin")


@dataclass(frozen=True)
class MeshRecoveryResult:
    vertices: np.ndarray
    keypoints3d: np.ndarray
    theta: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class MotionBERTMeshEstimator:
    """MotionBERT mesh recovery backend for Human3.6M 17-joint 2D input."""

    def __init__(
        self,
        repo_path: Path | str = DEFAULT_REPO_PATH,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda:0",
        clip_len: int = 243,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.clip_len = clip_len

        self._torch: Any | None = None
        self._args: Any | None = None
        self._model: Any | None = None
        self._smpl: Any | None = None
        self._j_regressor: Any | None = None
        self._crop_scale: Any | None = None
        self._flip_data: Any | None = None
        self._flip_thetas_batch: Any | None = None

    def recover(self, pose2d: np.ndarray) -> MeshRecoveryResult:
        pose2d = np.asarray(pose2d, dtype=np.float32)
        self._validate_pose2d(pose2d)
        self._ensure_loaded()

        motion = self._with_confidence(pose2d)
        motion = self._crop_scale(motion, scale_range=[1, 1]).astype(np.float32)

        verts_all = []
        kp3d_all = []
        theta_all = []
        device = self._device()

        with self._torch.no_grad():
            for clip in self._iter_clips(motion, self.clip_len):
                batch_input = self._torch.from_numpy(clip[None]).to(device).float()
                output = self._model(batch_input)

                if bool(getattr(self._args, "flip", False)):
                    output_flip = self._model(self._flip_data(batch_input))
                    output_flip_pose = output_flip[0]["theta"][:, :, :72]
                    output_flip_shape = output_flip[0]["theta"][:, :, 72:]
                    output_flip_pose = self._flip_thetas_batch(output_flip_pose).reshape(-1, 72)
                    output_flip_shape = output_flip_shape.reshape(-1, 10)
                    output_flip_smpl = self._smpl(
                        betas=output_flip_shape,
                        body_pose=output_flip_pose[:, 3:],
                        global_orient=output_flip_pose[:, :3],
                        pose2rot=True,
                    )
                    output_flip_verts = output_flip_smpl.vertices.detach()
                    regressor = self._j_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(device)
                    output_flip_kp3d = self._torch.matmul(regressor, output_flip_verts)
                    batch_size, clip_frames = batch_input.shape[:2]
                    output[0]["verts"] = (
                        output[0]["verts"]
                        + output_flip_verts.reshape(batch_size, clip_frames, -1, 3) * 1000.0
                    ) / 2.0
                    output[0]["kp_3d"] = (
                        output[0]["kp_3d"]
                        + output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3)
                    ) / 2.0

                verts_all.append(output[0]["verts"].detach().cpu().numpy()[0])
                kp3d_all.append(output[0]["kp_3d"].detach().cpu().numpy()[0])
                theta_all.append(output[0]["theta"].detach().cpu().numpy()[0])

        vertices = np.concatenate(verts_all, axis=0).astype(np.float32)
        keypoints3d = np.concatenate(kp3d_all, axis=0).astype(np.float32)
        theta = np.concatenate(theta_all, axis=0).astype(np.float32)

        return MeshRecoveryResult(
            vertices=vertices,
            keypoints3d=keypoints3d,
            theta=theta,
            faces=np.asarray(self._smpl.faces, dtype=np.int32),
            metadata={
                "backend": "motionbert_mesh",
                "repo_path": str(self.repo_path),
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "device": str(device),
                "clip_len": self.clip_len,
            },
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.repo_path.exists():
            raise FileNotFoundError(f"MotionBERT repo does not exist: {self.repo_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"MotionBERT mesh config does not exist: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"MotionBERT mesh checkpoint does not exist: {self.checkpoint_path}")

        self._patch_numpy_for_chumpy()
        repo_path = str(self.repo_path.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            import torch
            from lib.model.model_mesh import MeshRegressor
            from lib.utils.learning import load_backbone
            from lib.utils.tools import get_config
            from lib.utils.utils_data import crop_scale, flip_data
            from lib.utils.utils_mesh import flip_thetas_batch
            from lib.utils.utils_smpl import SMPL
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MotionBERT mesh dependencies are missing. Ensure smplx and MotionBERT mesh deps are installed."
            ) from exc

        self._torch = torch
        self._crop_scale = crop_scale
        self._flip_data = flip_data
        self._flip_thetas_batch = flip_thetas_batch
        self._args = get_config(str(self.config_path))
        self._args.data_root = str((self.repo_path / self._args.data_root).resolve())

        device = self._device()
        self._smpl = SMPL(self._args.data_root, batch_size=1, create_transl=False).to(device)
        self._j_regressor = self._smpl.J_regressor_h36m

        backbone = load_backbone(self._args)
        model = MeshRegressor(
            self._args,
            backbone=backbone,
            dim_rep=self._args.dim_rep,
            hidden_dim=self._args.hidden_dim,
            dropout_ratio=self._args.dropout,
        )
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(self._strip_module_prefix(state_dict), strict=True)
        model = model.to(device)
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
        if pose2d.shape[1:] != (17, 2):
            raise ValueError(f"Expected pose2d shape (T, 17, 2), got {pose2d.shape}")
        if pose2d.shape[0] == 0:
            raise ValueError("Expected at least one frame for mesh recovery")

    def _strip_module_prefix(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not state_dict:
            return state_dict
        if all(key.startswith("module.") for key in state_dict.keys()):
            return {key[7:]: value for key, value in state_dict.items()}
        return state_dict

    def _patch_numpy_for_chumpy(self) -> None:
        # chumpy imports aliases removed in numpy>=1.24 while loading older SMPL pkls.
        aliases = {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "unicode": str,
            "str": str,
        }
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)
