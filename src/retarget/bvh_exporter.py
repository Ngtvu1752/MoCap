from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.retarget.smpl_motion import build_source_motion
from src.retarget.smpl_skeleton import DEFAULT_SMPL_SKELETON_PATH, SMPLSkeleton, load_smpl_skeleton


DEFAULT_ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
ROOT_ROTATION_MODES = ("relative", "zero", "source")
ROOT_TRANSLATION_MODES = ("relative", "source", "zero")
DEFAULT_BVH_SCALE = 0.1
DEFAULT_BVH_UNITS = "centimeters"


@dataclass(frozen=True)
class SourceMotionForBVH:
    pose_axis_angle: np.ndarray | None
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


def _rotation_degrees_for_bvh(motion: SourceMotionForBVH, root_rotation: str) -> np.ndarray:
    euler = np.array(motion.pose_euler_xyz, dtype=np.float32, copy=True)
    if root_rotation == "zero":
        euler[:, 0, :] = 0.0
    elif root_rotation == "relative":
        euler[:, 0, :] = _relative_root_euler_xyz(motion)
    elif root_rotation == "source":
        pass
    else:
        _validate_mode("root_rotation", root_rotation, ROOT_ROTATION_MODES)
    return np.rad2deg(euler)


def _relative_root_euler_xyz(motion: SourceMotionForBVH) -> np.ndarray:
    if motion.pose_axis_angle is not None:
        root_rotations = Rotation.from_rotvec(motion.pose_axis_angle[:, 0, :])
    else:
        root_rotations = Rotation.from_euler("xyz", motion.pose_euler_xyz[:, 0, :])
    first_inverse = root_rotations[0].inv()
    relative = first_inverse * root_rotations
    return relative.as_euler("xyz").astype(np.float32)


def _root_positions_for_bvh(motion: SourceMotionForBVH, root_translation: str) -> np.ndarray:
    if root_translation == "source":
        return np.array(motion.root_translation, dtype=np.float32, copy=True)
    if root_translation == "relative":
        return (motion.root_translation - motion.root_translation[0]).astype(np.float32)
    if root_translation == "zero":
        return np.zeros_like(motion.root_translation, dtype=np.float32)
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
    offset = skeleton.offsets[joint_index] * scale
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
        end_offset = _leaf_end_site_offset(skeleton.offsets[joint_index]) * scale
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
            "How to write pelvis/root rotation. 'relative' removes the first-frame SMPL global "
            "orientation, 'zero' disables root rotation, 'source' writes raw SMPL root rotation. "
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
        help="Build <input-dir>/retarget/source_motion.npz first if it is missing.",
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

    if args.build_source_motion and input_dir is not None and not source_motion_path.exists():
        build_source_motion(input_dir, source_motion_path)

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


if __name__ == "__main__":
    main()
