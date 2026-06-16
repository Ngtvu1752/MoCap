from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SMPLMotion:
    pose_axis_angle: np.ndarray
    pose_quat: np.ndarray
    pose_euler_xyz: np.ndarray
    betas: np.ndarray
    root_translation: np.ndarray
    fps: float
    frame_count: int
    metadata: dict[str, Any]

    @property
    def num_frames(self) -> int:
        return int(self.pose_axis_angle.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.pose_axis_angle.shape[1])


def decode_smpl_motion(
    smpl_theta_path: Path | str,
    smpl_joints3d_path: Path | str,
    metadata_path: Path | str,
    *,
    rotation_order: str = "xyz",
) -> SMPLMotion:
    """Decode MotionBERT SMPL theta output into retarget-friendly motion arrays."""
    smpl_theta_path = Path(smpl_theta_path)
    smpl_joints3d_path = Path(smpl_joints3d_path)
    metadata_path = Path(metadata_path)

    theta = np.load(smpl_theta_path).astype(np.float32)
    joints3d = np.load(smpl_joints3d_path).astype(np.float32)
    metadata = _load_metadata(metadata_path)

    _validate_theta(theta, smpl_theta_path)
    _validate_joints3d(joints3d, smpl_joints3d_path)
    if theta.shape[0] != joints3d.shape[0]:
        raise ValueError(
            f"Frame count mismatch: {smpl_theta_path} has {theta.shape[0]} frames, "
            f"but {smpl_joints3d_path} has {joints3d.shape[0]} frames."
        )

    pose_axis_angle = theta[:, :72].reshape(theta.shape[0], 24, 3)
    betas = theta[:, 72:82]
    root_translation = joints3d[:, 0, :]

    rotations = Rotation.from_rotvec(pose_axis_angle.reshape(-1, 3))
    pose_quat = rotations.as_quat().reshape(theta.shape[0], 24, 4).astype(np.float32)
    pose_euler_xyz = rotations.as_euler(rotation_order).reshape(theta.shape[0], 24, 3).astype(np.float32)

    fps = _metadata_float(metadata, "fps")
    frame_count = _metadata_int(metadata, "frame_count")
    if frame_count != theta.shape[0]:
        raise ValueError(
            f"Metadata frame_count is {frame_count}, but SMPL theta contains {theta.shape[0]} frames."
        )

    _validate_finite("pose_axis_angle", pose_axis_angle)
    _validate_finite("pose_quat", pose_quat)
    _validate_finite("pose_euler_xyz", pose_euler_xyz)
    _validate_finite("betas", betas)
    _validate_finite("root_translation", root_translation)

    return SMPLMotion(
        pose_axis_angle=pose_axis_angle.astype(np.float32),
        pose_quat=pose_quat,
        pose_euler_xyz=pose_euler_xyz,
        betas=betas.astype(np.float32),
        root_translation=root_translation.astype(np.float32),
        fps=fps,
        frame_count=frame_count,
        metadata=metadata,
    )


def decode_smpl_motion_from_dir(input_dir: Path | str, *, rotation_order: str = "xyz") -> SMPLMotion:
    input_dir = Path(input_dir)
    return decode_smpl_motion(
        input_dir / "smpl_theta.npy",
        input_dir / "smpl_joints3d.npy",
        input_dir / "metadata.json",
        rotation_order=rotation_order,
    )


def save_source_motion(motion: SMPLMotion, output_path: Path | str, *, rotation_order: str = "xyz") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pose_axis_angle=motion.pose_axis_angle,
        pose_quat=motion.pose_quat,
        pose_euler_xyz=motion.pose_euler_xyz,
        betas=motion.betas,
        root_translation=motion.root_translation,
        fps=np.array(motion.fps, dtype=np.float32),
        frame_count=np.array(motion.frame_count, dtype=np.int32),
        joint_count=np.array(motion.num_joints, dtype=np.int32),
        rotation_order=np.array(rotation_order),
        source=np.array("motionbert_smpl"),
        coordinate_system=np.array("motionbert_smpl"),
    )
    return output_path


def build_source_motion(
    input_dir: Path | str,
    output_path: Path | str | None = None,
    *,
    rotation_order: str = "xyz",
) -> tuple[SMPLMotion, Path]:
    input_dir = Path(input_dir)
    if output_path is None:
        output_path = input_dir / "retarget" / "source_motion.npz"
    motion = decode_smpl_motion_from_dir(input_dir, rotation_order=rotation_order)
    saved_path = save_source_motion(motion, output_path, rotation_order=rotation_order)
    return motion, saved_path


def describe_npz(path: Path | str) -> str:
    path = Path(path)
    data = np.load(path)
    lines = [f"File: {path}"]
    for key in data.files:
        value = data[key]
        if value.shape == ():
            lines.append(f"{key}: {value.item()!r} dtype={value.dtype}")
        else:
            detail = f"{key}: shape={value.shape} dtype={value.dtype}"
            if np.issubdtype(value.dtype, np.number):
                detail += f" min={float(np.nanmin(value)):.6g} max={float(np.nanmax(value)):.6g}"
            lines.append(detail)
    return "\n".join(lines)


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_theta(theta: np.ndarray, path: Path) -> None:
    if theta.ndim != 2 or theta.shape[1] != 82:
        raise ValueError(f"Expected {path} to have shape (T,82), got {theta.shape}.")


def _validate_joints3d(joints3d: np.ndarray, path: Path) -> None:
    if joints3d.ndim != 3 or joints3d.shape[1:] != (17, 3):
        raise ValueError(f"Expected {path} to have shape (T,17,3), got {joints3d.shape}.")


def _metadata_float(metadata: dict[str, Any], key: str) -> float:
    try:
        return float(metadata["video"][key])
    except KeyError as exc:
        raise KeyError(f"metadata.json is missing video.{key}") from exc


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        return int(metadata["video"][key])
    except KeyError as exc:
        raise KeyError(f"metadata.json is missing video.{key}") from exc


def _validate_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode MotionBERT SMPL outputs into retarget source motion.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing smpl_theta.npy, smpl_joints3d.npy, and metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <input-dir>/retarget/source_motion.npz.",
    )
    parser.add_argument(
        "--rotation-order",
        default="xyz",
        help="Euler rotation order for pose_euler_xyz. Default: xyz.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print saved .npz keys, shapes, dtypes, and numeric ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion, output_path = build_source_motion(
        args.input_dir,
        args.output,
        rotation_order=args.rotation_order,
    )
    print(f"Saved source motion: {output_path}")
    print(f"Frames: {motion.frame_count}")
    print(f"FPS: {motion.fps:.6g}")
    print(f"Pose axis-angle: {motion.pose_axis_angle.shape}")
    print(f"Pose quaternion: {motion.pose_quat.shape}")
    print(f"Pose Euler {args.rotation_order}: {motion.pose_euler_xyz.shape}")
    print(f"Betas: {motion.betas.shape}")
    print(f"Root translation: {motion.root_translation.shape}")
    if args.inspect:
        print(describe_npz(output_path))


if __name__ == "__main__":
    main()
