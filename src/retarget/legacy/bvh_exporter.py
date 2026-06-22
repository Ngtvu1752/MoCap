from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.retarget.legacy.coordinates import (
    BVH_COORDINATE_SYSTEM,
    SOURCE_COORDINATE_SYSTEM,
    motionbert_points_to_bvh,
    motionbert_rotations_to_bvh,
)
from src.retarget.legacy.smpl_motion import build_source_motion
from src.retarget.legacy.smpl_skeleton import DEFAULT_SMPL_SKELETON_PATH, SMPLSkeleton, load_smpl_skeleton


DEFAULT_ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
ROOT_ROTATION_MODES = ("relative", "relative_yaw", "zero", "source")
ROOT_TRANSLATION_MODES = ("relative", "source", "zero")
DEFAULT_BVH_SCALE = 0.1
DEFAULT_BVH_UNITS = "centimeters"


@dataclass(frozen=True)
class SourceMotionForBVH:
    pose_axis_angle: np.ndarray | None
    pose_quat: np.ndarray | None
    pose_euler_xyz: np.ndarray
    root_translation: np.ndarray
    fps: float
    frame_count: int
    joint_count: int

    @property
    def frame_time(self) -> float:
        return 1.0 / self.fps


def load_source_motion(path: Path | str) -> SourceMotionForBVH:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source motion file does not exist: {path}")

    data = np.load(path)
    required = ["pose_euler_xyz", "root_translation", "fps", "frame_count", "joint_count"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")

    motion = SourceMotionForBVH(
        pose_axis_angle=np.asarray(data["pose_axis_angle"], dtype=np.float32) if "pose_axis_angle" in data.files else None,
        pose_quat=np.asarray(data["pose_quat"], dtype=np.float32) if "pose_quat" in data.files else None,
        pose_euler_xyz=np.asarray(data["pose_euler_xyz"], dtype=np.float32),
        root_translation=np.asarray(data["root_translation"], dtype=np.float32),
        fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        joint_count=int(data["joint_count"]),
    )
    validate_source_motion(motion, path)
    return motion


def validate_source_motion(motion: SourceMotionForBVH, path: Path | str) -> None:
    if motion.pose_axis_angle is not None and motion.pose_axis_angle.shape != motion.pose_euler_xyz.shape:
        raise ValueError(
            f"Expected pose_axis_angle shape {motion.pose_euler_xyz.shape} in {path}, got {motion.pose_axis_angle.shape}."
        )
    if motion.pose_quat is not None and motion.pose_quat.shape != (*motion.pose_euler_xyz.shape[:2], 4):
        raise ValueError(
            f"Expected pose_quat shape {(*motion.pose_euler_xyz.shape[:2], 4)} in {path}, got {motion.pose_quat.shape}."
        )
    if motion.pose_euler_xyz.ndim != 3 or motion.pose_euler_xyz.shape[1:] != (24, 3):
        raise ValueError(f"Expected pose_euler_xyz shape (T,24,3) in {path}, got {motion.pose_euler_xyz.shape}.")
    if motion.root_translation.shape != (motion.pose_euler_xyz.shape[0], 3):
        raise ValueError(
            f"Expected root_translation shape ({motion.pose_euler_xyz.shape[0]},3) in {path}, "
            f"got {motion.root_translation.shape}."
        )
    if motion.frame_count != motion.pose_euler_xyz.shape[0]:
        raise ValueError(
            f"frame_count is {motion.frame_count}, but pose_euler_xyz has {motion.pose_euler_xyz.shape[0]} frames."
        )
    if motion.joint_count != 24:
        raise ValueError(f"Expected joint_count 24 in {path}, got {motion.joint_count}.")
    if motion.fps <= 0:
        raise ValueError(f"Expected positive fps in {path}, got {motion.fps}.")
    if motion.pose_axis_angle is not None and not np.isfinite(motion.pose_axis_angle).all():
        raise ValueError(f"pose_axis_angle in {path} contains NaN or Inf values.")
    if motion.pose_quat is not None and not np.isfinite(motion.pose_quat).all():
        raise ValueError(f"pose_quat in {path} contains NaN or Inf values.")
    if not np.isfinite(motion.pose_euler_xyz).all():
        raise ValueError(f"pose_euler_xyz in {path} contains NaN or Inf values.")
    if not np.isfinite(motion.root_translation).all():
        raise ValueError(f"root_translation in {path} contains NaN or Inf values.")


def export_source_smpl_bvh(
    source_motion_path: Path | str,
    skeleton_path: Path | str,
    output_path: Path | str,
    *,
    scale: float = DEFAULT_BVH_SCALE,
    root_rotation: str = "relative",
    root_translation: str = "relative",
) -> Path:
    motion = load_source_motion(source_motion_path)
    skeleton = load_smpl_skeleton(skeleton_path)
    validate_motion_matches_skeleton(motion, skeleton)
    return write_bvh(
        motion,
        skeleton,
        output_path,
        scale=scale,
        root_rotation=root_rotation,
        root_translation=root_translation,
    )


def write_bvh(
    motion: SourceMotionForBVH,
    skeleton: SMPLSkeleton,
    output_path: Path | str,
    *,
    scale: float = DEFAULT_BVH_SCALE,
    root_rotation: str = "relative",
    root_translation: str = "relative",
) -> Path:
    _validate_mode("root_rotation", root_rotation, ROOT_ROTATION_MODES)
    _validate_mode("root_translation", root_translation, ROOT_TRANSLATION_MODES)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["HIERARCHY"]
    _append_joint_hierarchy(lines, skeleton, skeleton.root_index, indent=0, scale=scale)
    lines.append("MOTION")
    lines.append(f"Frames: {motion.frame_count}")
    lines.append(f"Frame Time: {motion.frame_time:.10f}")

    euler_degrees = _rotation_degrees_for_bvh(motion, root_rotation)
    root_positions = _root_positions_for_bvh(motion, root_translation) * scale
    traversal = hierarchy_traversal(skeleton)

    for frame_index in range(motion.frame_count):
        values: list[float] = []
        root = root_positions[frame_index]
        values.extend([float(root[0]), float(root[1]), float(root[2])])
        for joint_index in traversal:
            rotation = euler_degrees[frame_index, joint_index]
            values.extend([float(rotation[0]), float(rotation[1]), float(rotation[2])])
        lines.append(" ".join(f"{value:.6f}" for value in values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def validate_motion_matches_skeleton(motion: SourceMotionForBVH, skeleton: SMPLSkeleton) -> None:
    if motion.joint_count != skeleton.joint_count:
        raise ValueError(f"Motion has {motion.joint_count} joints, skeleton has {skeleton.joint_count}.")


def validate_source_motion_is_fresh(input_dir: Path | str, source_motion_path: Path | str) -> None:
    input_dir = Path(input_dir)
    source_motion_path = Path(source_motion_path)
    if not source_motion_path.exists():
        raise FileNotFoundError(
            f"Source motion file does not exist: {source_motion_path}. Run again with --build-source-motion."
        )

    source_mtime = source_motion_path.stat().st_mtime
    stale_inputs = [path for path in _source_motion_inputs(input_dir) if path.stat().st_mtime > source_mtime]
    if stale_inputs:
        names = ", ".join(str(path) for path in stale_inputs)
        raise RuntimeError(
            f"Source motion file is older than source input(s): {names}. "
            "Run export_smpl_bvh.py with --build-source-motion to rebuild it."
        )


def _source_motion_inputs(input_dir: Path) -> list[Path]:
    paths = [input_dir / "smpl_theta.npy", input_dir / "smpl_joints3d.npy", input_dir / "metadata.json"]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source input(s): " + ", ".join(str(path) for path in missing))
    return paths


def _rotation_degrees_for_bvh(motion: SourceMotionForBVH, root_rotation: str) -> np.ndarray:
    rotations = motionbert_rotations_to_bvh(_source_rotation_matrices(motion))
    if root_rotation == "zero":
        rotations[:, 0, :, :] = np.eye(3, dtype=np.float32)
    elif root_rotation == "relative":
        rotations[:, 0, :, :] = _relative_root_rotations(rotations)
    elif root_rotation == "relative_yaw":
        rotations[:, 0, :, :] = _relative_root_yaw_rotations(rotations)
    elif root_rotation == "source":
        pass
    else:
        _validate_mode("root_rotation", root_rotation, ROOT_ROTATION_MODES)

    euler = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_euler("xyz").reshape(motion.frame_count, 24, 3)
    euler = np.unwrap(euler, axis=0)
    return np.rad2deg(euler).astype(np.float32)


def _source_rotation_matrices(motion: SourceMotionForBVH) -> np.ndarray:
    if motion.pose_axis_angle is not None:
        rotations = Rotation.from_rotvec(motion.pose_axis_angle.reshape(-1, 3))
    elif motion.pose_quat is not None:
        rotations = Rotation.from_quat(motion.pose_quat.reshape(-1, 4))
    else:
        rotations = Rotation.from_euler("xyz", motion.pose_euler_xyz.reshape(-1, 3))
    return rotations.as_matrix().reshape(motion.frame_count, motion.joint_count, 3, 3).astype(np.float32)


def _relative_root_rotations(rotations: np.ndarray) -> np.ndarray:
    root_rotations = Rotation.from_matrix(rotations[:, 0, :, :])
    first_inverse = root_rotations[0].inv()
    relative = first_inverse * root_rotations
    return relative.as_matrix().astype(np.float32)


def _relative_root_yaw_rotations(rotations: np.ndarray) -> np.ndarray:
    root_rotations = Rotation.from_matrix(rotations[:, 0, :, :])
    first_inverse = root_rotations[0].inv()
    relative = first_inverse * root_rotations
    yaw = np.unwrap(relative.as_euler("xyz")[:, 2])
    return Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)


def _root_positions_for_bvh(motion: SourceMotionForBVH, root_translation: str) -> np.ndarray:
    converted = motionbert_points_to_bvh(motion.root_translation)
    if root_translation == "source":
        return np.array(converted, dtype=np.float32, copy=True)
    if root_translation == "relative":
        return (converted - converted[0]).astype(np.float32)
    if root_translation == "zero":
        return np.zeros_like(converted, dtype=np.float32)
    _validate_mode("root_translation", root_translation, ROOT_TRANSLATION_MODES)
    raise AssertionError("unreachable")


def _validate_mode(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"Unsupported {name}: {value!r}. Expected one of: {', '.join(allowed)}.")


def hierarchy_traversal(skeleton: SMPLSkeleton) -> list[int]:
    ordered: list[int] = []

    def visit(joint_index: int) -> None:
        ordered.append(joint_index)
        for child in skeleton.children(joint_index):
            visit(child)

    visit(skeleton.root_index)
    return ordered


def _append_joint_hierarchy(
    lines: list[str],
    skeleton: SMPLSkeleton,
    joint_index: int,
    *,
    indent: int,
    scale: float,
) -> None:
    prefix = "\t" * indent
    joint_name = skeleton.joints[joint_index]
    label = "ROOT" if skeleton.parents[joint_index] == -1 else "JOINT"

    lines.append(f"{prefix}{label} {joint_name}")
    lines.append(f"{prefix}{{")
    offset = motionbert_points_to_bvh(skeleton.offsets[joint_index]) * scale
    lines.append(f"{prefix}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    if label == "ROOT":
        lines.append(f"{prefix}\tCHANNELS 6 Xposition Yposition Zposition {' '.join(DEFAULT_ROTATION_CHANNELS)}")
    else:
        lines.append(f"{prefix}\tCHANNELS 3 {' '.join(DEFAULT_ROTATION_CHANNELS)}")

    children = skeleton.children(joint_index)
    if children:
        for child_index in children:
            _append_joint_hierarchy(lines, skeleton, child_index, indent=indent + 1, scale=scale)
    else:
        end_offset = _leaf_end_site_offset(offset / scale) * scale
        lines.append(f"{prefix}\tEnd Site")
        lines.append(f"{prefix}\t{{")
        lines.append(f"{prefix}\t\tOFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}")
        lines.append(f"{prefix}\t}}")

    lines.append(f"{prefix}}}")


def _leaf_end_site_offset(joint_offset: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(joint_offset))
    if length <= 1e-6:
        return np.array([0.0, 50.0, 0.0], dtype=np.float32)
    return (joint_offset / length * min(length * 0.4, 80.0)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized SMPL source motion to BVH.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Pipeline output directory. Uses <input-dir>/retarget/source_motion.npz by default.",
    )
    parser.add_argument(
        "--source-motion",
        type=Path,
        default=None,
        help="Path to source_motion.npz. Overrides --input-dir source motion path.",
    )
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=DEFAULT_SMPL_SKELETON_PATH,
        help="Path to smpl_skeleton.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output BVH path. Defaults to <input-dir>/retarget/source_smpl.bvh.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_BVH_SCALE,
        help=(
            "Scale applied to root translation and skeleton offsets. Default: 0.1, "
            "which converts MotionBERT/SMPL millimeters to BVH centimeters."
        ),
    )
    parser.add_argument(
        "--root-rotation",
        choices=ROOT_ROTATION_MODES,
        default="relative",
        help=(
            "How to write pelvis/root rotation after coordinate conversion. 'relative' removes the "
            "first-frame global orientation, 'relative_yaw' keeps only heading around BVH Z-up, "
            "'zero' disables root rotation, 'source' writes converted source root rotation. "
            "Default: relative."
        ),
    )
    parser.add_argument(
        "--root-translation",
        choices=ROOT_TRANSLATION_MODES,
        default="relative",
        help=(
            "How to write pelvis/root translation. 'relative' starts at the origin, 'source' writes "
            "raw MotionBERT translation, 'zero' disables root translation. Default: relative."
        ),
    )
    parser.add_argument(
        "--build-source-motion",
        action="store_true",
        help="Rebuild and overwrite <input-dir>/retarget/source_motion.npz before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_motion is None and args.input_dir is None:
        raise SystemExit("Either --input-dir or --source-motion is required.")

    input_dir = args.input_dir
    source_motion_path = args.source_motion
    if source_motion_path is None:
        source_motion_path = input_dir / "retarget" / "source_motion.npz"

    source_motion_rebuilt = False
    if args.build_source_motion:
        if input_dir is None:
            raise SystemExit("--build-source-motion requires --input-dir.")
        build_source_motion(input_dir, source_motion_path)
        source_motion_rebuilt = True
    elif args.source_motion is None and input_dir is not None:
        try:
            validate_source_motion_is_fresh(input_dir, source_motion_path)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc

    output_path = args.output
    if output_path is None:
        if input_dir is None:
            output_path = source_motion_path.with_name("source_smpl.bvh")
        else:
            output_path = input_dir / "retarget" / "source_smpl.bvh"

    written_path = export_source_smpl_bvh(
        source_motion_path,
        args.skeleton,
        output_path,
        scale=args.scale,
        root_rotation=args.root_rotation,
        root_translation=args.root_translation,
    )
    motion = load_source_motion(source_motion_path)
    print(f"Saved BVH: {written_path}")
    print(f"Frames: {motion.frame_count}")
    print(f"Frame Time: {motion.frame_time:.10f}")
    print(f"Joints: {motion.joint_count}")
    print(f"Scale: {args.scale}")
    print(f"Output units: {DEFAULT_BVH_UNITS if args.scale == DEFAULT_BVH_SCALE else 'custom'}")
    print(f"Root rotation: {args.root_rotation}")
    print(f"Root translation: {args.root_translation}")
    print(f"Source coordinate: {SOURCE_COORDINATE_SYSTEM}")
    print(f"Target coordinate: {BVH_COORDINATE_SYSTEM}")
    print("Rotation conversion: enabled")
    print("Euler unwrap: enabled")
    print(f"Source motion rebuilt: {str(source_motion_rebuilt).lower()}")


if __name__ == "__main__":
    main()
