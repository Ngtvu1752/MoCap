from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.mesh.base import HumanMeshEstimator, MeshResult
from src.pose3d.adapters import CocoBody17, Halpe26, Human36M17, KeypointFormat


DEFAULT_REPO_PATH = Path("MotionBERT")
DEFAULT_CONFIG_PATH = Path("MotionBERT/configs/mesh/MB_ft_pw3d.yaml")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/MotionBERT/FT_MB_release_MB_ft_pw3d/best_epoch.bin")
DEFAULT_SMPL_DATA_ROOT = Path("checkpoints/Mesh")


class MotionBERTMeshEstimator(HumanMeshEstimator):
    """MotionBERT MeshRegressor wrapper for SMPL human mesh recovery."""

    def __init__(
        self,
        repo_path: Path | str = DEFAULT_REPO_PATH,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        smpl_data_root: Path | str = DEFAULT_SMPL_DATA_ROOT,
        source_format: KeypointFormat = KeypointFormat.HALPE26,
        device: str = "cuda:0",
        clip_len: int | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.smpl_data_root = Path(smpl_data_root)
        self.source_format = KeypointFormat(source_format)
        self.device = device
        self.clip_len = clip_len

        self._torch: Any | None = None
        self._args: Any | None = None
        self._model: Any | None = None
        self._smpl: Any | None = None
        self._j_regressor: Any | None = None
        self._faces: np.ndarray | None = None
        self._crop_scale: Any | None = None
        self._flip_data: Any | None = None
        self._flip_thetas_batch: Any | None = None

    def recover(self, pose2d: np.ndarray, scores: np.ndarray | None = None) -> MeshResult:
        pose2d = np.asarray(pose2d, dtype=np.float32)
        scores = self._prepare_scores(pose2d, scores)
        self._validate_pose2d(pose2d, scores)
        self._ensure_model_loaded()

        motion = self._to_h36m_with_confidence(pose2d, scores)
        motion = self._crop_scale(motion, scale_range=[1, 1]).astype(np.float32)

        clip_len = self.clip_len or int(self._args.clip_len)
        vertices = []
        joints3d = []
        theta = []

        with self._torch.no_grad():
            for clip in self._iter_clips(motion, clip_len):
                batch_input = self._torch.from_numpy(clip[None]).to(self._device()).float()
                output = self._predict(batch_input)
                vertices.append(output["verts"])
                joints3d.append(output["kp_3d"])
                theta.append(output["theta"])

        vertices_np = np.concatenate(vertices, axis=0).astype(np.float32)
        joints3d_np = np.concatenate(joints3d, axis=0).astype(np.float32)
        theta_np = np.concatenate(theta, axis=0).astype(np.float32)

        return MeshResult(
            vertices=vertices_np,
            joints3d=joints3d_np,
            theta=theta_np,
            faces=np.asarray(self._faces, dtype=np.int32),
            metadata={
                "backend": "motionbert_mesh",
                "repo_path": str(self.repo_path),
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "smpl_data_root": str(self.smpl_data_root),
                "source_format": self.source_format.value,
                "device": str(self._device()),
                "clip_len": clip_len,
                "input_space": "motionbert_crop_scaled_h36m17_with_confidence",
                "vertices_shape": list(vertices_np.shape),
                "theta_shape": list(theta_np.shape),
            },
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        self._patch_numpy_legacy_aliases()
        self._validate_paths()
        repo_path = str(self.repo_path.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            import torch
            import torch.nn as nn
            from lib.model.model_mesh import MeshRegressor
            from lib.utils.learning import load_backbone
            from lib.utils.tools import get_config
            from lib.utils.utils_data import crop_scale, flip_data
            from lib.utils.utils_mesh import flip_thetas_batch
            from lib.utils.utils_smpl import SMPL
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MotionBERT mesh dependencies are missing. Install MotionBERT/OpenMMLab "
                "dependencies before running Human Mesh Recovery."
            ) from exc

        self._torch = torch
        self._crop_scale = crop_scale
        self._flip_data = flip_data
        self._flip_thetas_batch = flip_thetas_batch
        self._args = get_config(str(self.config_path))
        self._args.data_root = str(self.smpl_data_root)

        backbone = load_backbone(self._args)
        model = MeshRegressor(
            self._args,
            backbone=backbone,
            dim_rep=self._args.dim_rep,
            hidden_dim=self._args.hidden_dim,
            dropout_ratio=self._args.dropout,
        )
        if self._device().type == "cuda":
            model = nn.DataParallel(model)
        model = model.to(self._device())

        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self._load_state_dict(model, state_dict)
        model.eval()

        smpl = SMPL(str(self.smpl_data_root), batch_size=1).to(self._device())
        smpl.eval()
        self._smpl = smpl
        self._j_regressor = smpl.J_regressor_h36m
        self._faces = np.asarray(smpl.faces, dtype=np.int32)
        self._model = model

    def _predict(self, batch_input: Any) -> dict[str, np.ndarray]:
        output = self._model(batch_input)
        final = {
            "theta": output[0]["theta"],
            "verts": output[0]["verts"],
            "kp_3d": output[0]["kp_3d"],
        }

        if bool(getattr(self._args, "flip", False)):
            output_flip = self._model(self._flip_data(batch_input))
            flipped = self._unflip_mesh_output(output_flip, batch_input.shape[0], batch_input.shape[1])
            final["verts"] = (final["verts"] + flipped["verts"]) / 2.0
            final["kp_3d"] = (final["kp_3d"] + flipped["kp_3d"]) / 2.0

        return {key: value.detach().cpu().numpy()[0] for key, value in final.items()}

    def _unflip_mesh_output(self, output_flip: Any, batch_size: int, clip_frames: int) -> dict[str, Any]:
        output_flip_pose = output_flip[0]["theta"][:, :, :72]
        output_flip_shape = output_flip[0]["theta"][:, :, 72:]
        output_flip_pose = self._flip_thetas_batch(output_flip_pose)
        output_flip_pose = output_flip_pose.reshape(-1, 72)
        output_flip_shape = output_flip_shape.reshape(-1, 10)
        output_flip_smpl = self._smpl(
            betas=output_flip_shape,
            body_pose=output_flip_pose[:, 3:],
            global_orient=output_flip_pose[:, :3],
            pose2rot=True,
        )
        output_flip_verts = output_flip_smpl.vertices.detach() * 1000.0
        j_regressor = self._j_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
        output_flip_kp3d = self._torch.matmul(j_regressor, output_flip_verts)
        return {
            "theta": output_flip[0]["theta"],
            "verts": output_flip_verts.reshape(batch_size, clip_frames, -1, 3),
            "kp_3d": output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3),
        }

    def _load_state_dict(self, model: Any, state_dict: dict[str, Any]) -> None:
        try:
            model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass

        model_is_parallel = hasattr(model, "module")
        keys_have_module = all(key.startswith("module.") for key in state_dict.keys())
        if keys_have_module and not model_is_parallel:
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        elif model_is_parallel and not keys_have_module:
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    def _to_h36m_with_confidence(self, pose2d: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.source_format == KeypointFormat.HALPE26:
            return self._halpe26_to_h36m_with_confidence(pose2d, scores)
        if self.source_format in {KeypointFormat.WHOLE_BODY133, KeypointFormat.COCO_BODY17, KeypointFormat.RTMPOSE_RAW}:
            return self._coco_to_h36m_with_confidence(pose2d, scores)
        if self.source_format == KeypointFormat.HUMAN36M_17:
            return np.concatenate([pose2d, scores[..., None]], axis=-1).astype(np.float32)
        raise ValueError(f"Unsupported mesh source format: {self.source_format}")

    def _coco_to_h36m_with_confidence(self, pose2d: np.ndarray, scores: np.ndarray) -> np.ndarray:
        self._validate_min_joints(pose2d, scores, 17, "COCO body 17")
        coco_xy = pose2d[:, :17]
        coco_score = scores[:, :17]
        h36m = np.zeros((pose2d.shape[0], 17, 3), dtype=np.float32)

        left_hip = coco_xy[:, CocoBody17.LEFT_HIP]
        right_hip = coco_xy[:, CocoBody17.RIGHT_HIP]
        left_shoulder = coco_xy[:, CocoBody17.LEFT_SHOULDER]
        right_shoulder = coco_xy[:, CocoBody17.RIGHT_SHOULDER]
        pelvis = (left_hip + right_hip) * 0.5
        neck = (left_shoulder + right_shoulder) * 0.5
        torso = (pelvis + neck) * 0.5
        nose = coco_xy[:, CocoBody17.NOSE]
        head = nose + (nose - neck) * 0.35

        self._assign(h36m, Human36M17.PELVIS, pelvis, self._mean_score(coco_score, CocoBody17.LEFT_HIP, CocoBody17.RIGHT_HIP))
        self._assign(h36m, Human36M17.RIGHT_HIP, right_hip, coco_score[:, CocoBody17.RIGHT_HIP])
        self._assign(h36m, Human36M17.RIGHT_KNEE, coco_xy[:, CocoBody17.RIGHT_KNEE], coco_score[:, CocoBody17.RIGHT_KNEE])
        self._assign(h36m, Human36M17.RIGHT_ANKLE, coco_xy[:, CocoBody17.RIGHT_ANKLE], coco_score[:, CocoBody17.RIGHT_ANKLE])
        self._assign(h36m, Human36M17.LEFT_HIP, left_hip, coco_score[:, CocoBody17.LEFT_HIP])
        self._assign(h36m, Human36M17.LEFT_KNEE, coco_xy[:, CocoBody17.LEFT_KNEE], coco_score[:, CocoBody17.LEFT_KNEE])
        self._assign(h36m, Human36M17.LEFT_ANKLE, coco_xy[:, CocoBody17.LEFT_ANKLE], coco_score[:, CocoBody17.LEFT_ANKLE])
        self._assign(h36m, Human36M17.TORSO, torso, self._mean_score(coco_score, CocoBody17.LEFT_HIP, CocoBody17.RIGHT_HIP, CocoBody17.LEFT_SHOULDER, CocoBody17.RIGHT_SHOULDER))
        self._assign(h36m, Human36M17.NECK, neck, self._mean_score(coco_score, CocoBody17.LEFT_SHOULDER, CocoBody17.RIGHT_SHOULDER))
        self._assign(h36m, Human36M17.NOSE, nose, coco_score[:, CocoBody17.NOSE])
        self._assign(h36m, Human36M17.HEAD, head, self._mean_score(coco_score, CocoBody17.NOSE, CocoBody17.LEFT_SHOULDER, CocoBody17.RIGHT_SHOULDER))
        self._assign(h36m, Human36M17.LEFT_SHOULDER, left_shoulder, coco_score[:, CocoBody17.LEFT_SHOULDER])
        self._assign(h36m, Human36M17.LEFT_ELBOW, coco_xy[:, CocoBody17.LEFT_ELBOW], coco_score[:, CocoBody17.LEFT_ELBOW])
        self._assign(h36m, Human36M17.LEFT_WRIST, coco_xy[:, CocoBody17.LEFT_WRIST], coco_score[:, CocoBody17.LEFT_WRIST])
        self._assign(h36m, Human36M17.RIGHT_SHOULDER, right_shoulder, coco_score[:, CocoBody17.RIGHT_SHOULDER])
        self._assign(h36m, Human36M17.RIGHT_ELBOW, coco_xy[:, CocoBody17.RIGHT_ELBOW], coco_score[:, CocoBody17.RIGHT_ELBOW])
        self._assign(h36m, Human36M17.RIGHT_WRIST, coco_xy[:, CocoBody17.RIGHT_WRIST], coco_score[:, CocoBody17.RIGHT_WRIST])
        return h36m

    def _halpe26_to_h36m_with_confidence(self, pose2d: np.ndarray, scores: np.ndarray) -> np.ndarray:
        self._validate_min_joints(pose2d, scores, 26, "Halpe 26")
        h36m = np.zeros((pose2d.shape[0], 17, 3), dtype=np.float32)
        mapping = {
            Human36M17.PELVIS: Halpe26.HIP,
            Human36M17.RIGHT_HIP: Halpe26.RIGHT_HIP,
            Human36M17.RIGHT_KNEE: Halpe26.RIGHT_KNEE,
            Human36M17.RIGHT_ANKLE: Halpe26.RIGHT_ANKLE,
            Human36M17.LEFT_HIP: Halpe26.LEFT_HIP,
            Human36M17.LEFT_KNEE: Halpe26.LEFT_KNEE,
            Human36M17.LEFT_ANKLE: Halpe26.LEFT_ANKLE,
            Human36M17.NECK: Halpe26.NECK,
            Human36M17.NOSE: Halpe26.NOSE,
            Human36M17.HEAD: Halpe26.HEAD,
            Human36M17.LEFT_SHOULDER: Halpe26.LEFT_SHOULDER,
            Human36M17.LEFT_ELBOW: Halpe26.LEFT_ELBOW,
            Human36M17.LEFT_WRIST: Halpe26.LEFT_WRIST,
            Human36M17.RIGHT_SHOULDER: Halpe26.RIGHT_SHOULDER,
            Human36M17.RIGHT_ELBOW: Halpe26.RIGHT_ELBOW,
            Human36M17.RIGHT_WRIST: Halpe26.RIGHT_WRIST,
        }
        for dst, src in mapping.items():
            self._assign(h36m, dst, pose2d[:, src], scores[:, src])
        torso = (pose2d[:, Halpe26.NECK] + pose2d[:, Halpe26.HIP]) * 0.5
        torso_score = self._mean_score(scores, Halpe26.NECK, Halpe26.HIP)
        self._assign(h36m, Human36M17.TORSO, torso, torso_score)
        return h36m

    def _prepare_scores(self, pose2d: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
        if scores is None:
            return np.ones(pose2d.shape[:2], dtype=np.float32)
        return np.asarray(scores, dtype=np.float32)

    def _validate_pose2d(self, pose2d: np.ndarray, scores: np.ndarray) -> None:
        if pose2d.ndim != 3 or pose2d.shape[-1] != 2:
            raise ValueError(f"Expected pose2d shape (T, K, 2), got {pose2d.shape}")
        if scores.shape != pose2d.shape[:2]:
            raise ValueError(f"Expected scores shape {pose2d.shape[:2]}, got {scores.shape}")
        if pose2d.shape[0] == 0:
            raise ValueError("Expected at least one frame for mesh recovery")
        if not np.isfinite(pose2d).all() or not np.isfinite(scores).all():
            raise ValueError("pose2d or scores contain NaN or infinite values")

    def _validate_min_joints(self, pose2d: np.ndarray, scores: np.ndarray, min_joints: int, name: str) -> None:
        if pose2d.shape[1] < min_joints or scores.shape[1] < min_joints:
            raise ValueError(f"Expected at least {min_joints} {name} joints, got {pose2d.shape[1]}")

    def _patch_numpy_legacy_aliases(self) -> None:
        # chumpy is unmaintained and still imports NumPy aliases removed after
        # NumPy 1.23. SMPL pickle loading can import chumpy, so patch before
        # constructing MotionBERT's SMPL wrapper.
        aliases = {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "str": str,
            "unicode": str,
        }
        for name, value in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, value)

    def _validate_paths(self) -> None:
        for path, label in [
            (self.repo_path, "MotionBERT repo"),
            (self.config_path, "MotionBERT mesh config"),
            (self.checkpoint_path, "MotionBERT mesh checkpoint"),
            (self.smpl_data_root, "SMPL data root"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        for name in ["SMPL_NEUTRAL.pkl", "smpl_mean_params.npz", "J_regressor_extra.npy", "J_regressor_h36m_correct.npy"]:
            path = self.smpl_data_root / name
            if not path.exists():
                raise FileNotFoundError(f"Required SMPL mesh asset does not exist: {path}")

    def _iter_clips(self, motion: np.ndarray, clip_len: int):
        for start in range(0, len(motion), clip_len):
            yield motion[start : start + clip_len]

    def _device(self) -> Any:
        torch = self._torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _assign(self, target: np.ndarray, joint: int, xy: np.ndarray, score: np.ndarray) -> None:
        target[:, joint, :2] = xy
        target[:, joint, 2] = score

    def _mean_score(self, scores: np.ndarray, *indices: int) -> np.ndarray:
        return np.mean([scores[:, index] for index in indices], axis=0).astype(np.float32)
