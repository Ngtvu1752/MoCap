from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.retarget.root_motion import RootMotionResult, build_pose3d_root_motion


DEFAULT_BASE_FBX = Path("assets/retarget/smpl_base_tpose.fbx")
DEFAULT_BLENDER_SCRIPT = Path("src/retarget/blender/bake_smpl_fbx.py")
ROOT_TRAJECTORY_MODES = ("pose3d", "smpl", "zero")


@dataclass(frozen=True)
class BlenderExportConfig:
    input_dir: Path
    base_fbx: Path = DEFAULT_BASE_FBX
    blender_script: Path = DEFAULT_BLENDER_SCRIPT
    output: Path | None = None
    blender_executable: str = "blender"
    root_scale: float = 0.1
    root_trajectory: str = "pose3d"
    pose3d_scale_mm: float | None = None


def export_animated_smpl_fbx(config: BlenderExportConfig) -> Path:
    input_dir = config.input_dir.resolve()
    theta_path = input_dir / "smpl_theta.npy"
    joints_path = input_dir / "smpl_joints3d.npy"
    pose3d_path = input_dir / "pose3d.npy"
    metadata_path = input_dir / "metadata.json"
    output_path = (config.output or input_dir / "retarget" / "animated_smpl.fbx").resolve()
    root_trajectory_path = output_path.parent / "root_trajectory.npy"
    base_fbx = config.base_fbx.resolve()
    blender_script = config.blender_script.resolve()

    _validate_paths(theta_path, joints_path, metadata_path, base_fbx, blender_script)
    _validate_root_trajectory_mode(config.root_trajectory)
    if config.root_trajectory == "pose3d":
        _validate_paths(pose3d_path)
    fps = _validate_inputs(
        theta_path,
        joints_path,
        metadata_path,
        pose3d_path if config.root_trajectory == "pose3d" else None,
    )
    blender = _resolve_blender(config.blender_executable)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_trajectory = _prepare_root_trajectory(
        config.root_trajectory,
        pose3d_path,
        joints_path,
        fps,
        pose3d_scale_mm=config.pose3d_scale_mm,
    )
    np.save(root_trajectory_path, root_trajectory)

    command = build_blender_command(
        blender,
        blender_script,
        base_fbx,
        theta_path,
        root_trajectory_path,
        output_path,
        fps=fps,
        root_scale=config.root_scale,
    )
    subprocess.run(command, check=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Blender completed without creating a valid FBX: {output_path}")
    return output_path


def build_blender_command(
    blender: str,
    blender_script: Path,
    base_fbx: Path,
    theta_path: Path,
    root_trajectory_path: Path,
    output_path: Path,
    *,
    fps: float,
    root_scale: float,
) -> list[str]:
    return [
        blender,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--base-fbx",
        str(base_fbx),
        "--theta",
        str(theta_path),
        "--root-translation",
        str(root_trajectory_path),
        "--output",
        str(output_path),
        "--fps",
        f"{fps:.10g}",
        "--root-scale",
        f"{root_scale:.10g}",
    ]


def _validate_paths(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(str(path) for path in missing))


def _validate_inputs(
    theta_path: Path,
    joints_path: Path,
    metadata_path: Path,
    pose3d_path: Path | None = None,
) -> float:
    theta = np.load(theta_path, mmap_mode="r")
    joints = np.load(joints_path, mmap_mode="r")
    if theta.ndim != 2 or theta.shape[1] != 82:
        raise ValueError(f"Expected {theta_path} shape (T,82), got {theta.shape}.")
    if joints.ndim != 3 or joints.shape[1:] != (17, 3):
        raise ValueError(f"Expected {joints_path} shape (T,17,3), got {joints.shape}.")
    if theta.shape[0] != joints.shape[0]:
        raise ValueError(f"Frame count mismatch: theta={theta.shape[0]}, joints={joints.shape[0]}.")
    if theta.shape[0] == 0:
        raise ValueError("Cannot export an empty motion sequence.")
    if not np.isfinite(theta).all() or not np.isfinite(joints).all():
        raise ValueError("SMPL motion arrays contain NaN or Inf values.")

    if pose3d_path is not None:
        pose3d = np.load(pose3d_path, mmap_mode="r")
        if pose3d.ndim != 3 or pose3d.shape[1:] != (17, 3):
            raise ValueError(f"Expected {pose3d_path} shape (T,17,3), got {pose3d.shape}.")
        if pose3d.shape[0] != theta.shape[0]:
            raise ValueError(
                f"Frame count mismatch: theta={theta.shape[0]}, pose3d={pose3d.shape[0]}."
            )
        if not np.isfinite(pose3d).all():
            raise ValueError(f"{pose3d_path} contains NaN or Inf values.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata["video"]["fps"])
    frame_count = int(metadata["video"]["frame_count"])
    if fps <= 0:
        raise ValueError(f"Expected positive video FPS, got {fps}.")
    if frame_count != theta.shape[0]:
        raise ValueError(f"Metadata frame_count={frame_count}, but motion contains {theta.shape[0]} frames.")
    return fps


def _prepare_root_trajectory(
    mode: str,
    pose3d_path: Path,
    joints_path: Path,
    fps: float,
    *,
    pose3d_scale_mm: float | None,
) -> np.ndarray:
    joints = np.load(joints_path).astype(np.float32)
    if mode == "pose3d":
        result = build_pose3d_root_motion(
            np.load(pose3d_path),
            joints,
            fps,
            scale_mm_per_pose_unit=pose3d_scale_mm,
        )
        _print_root_motion_summary(result)
        return result.translation_source_mm
    if pose3d_scale_mm is not None:
        raise ValueError("--pose3d-scale-mm is only valid with --root-trajectory pose3d.")
    if mode == "smpl":
        trajectory = joints[:, 0] - joints[0:1, 0]
    elif mode == "zero":
        trajectory = np.zeros((joints.shape[0], 3), dtype=np.float32)
    else:
        _validate_root_trajectory_mode(mode)
        raise AssertionError("unreachable")
    span_m = np.ptp(trajectory, axis=0) * 0.001
    print(f"Root trajectory: {mode}")
    print(f"Root span XYZ (m): {_format_vector(span_m)}")
    return trajectory.astype(np.float32)


def _print_root_motion_summary(result: RootMotionResult) -> None:
    print("Root trajectory: pose3d")
    print(f"Pose3D scale (mm/unit): {result.scale_mm_per_pose_unit:.3f}")
    print(f"Raw root span XYZ (m): {_format_vector(result.raw_span_m)}")
    print(f"Clean root span XYZ (m): {_format_vector(result.cleaned_span_m)}")
    print(f"Foot contacts: left={int(result.left_contact.sum())}, right={int(result.right_contact.sum())}")


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def _validate_root_trajectory_mode(mode: str) -> None:
    if mode not in ROOT_TRAJECTORY_MODES:
        raise ValueError(
            f"Unknown root trajectory mode {mode!r}; expected one of {', '.join(ROOT_TRAJECTORY_MODES)}."
        )


def _resolve_blender(executable: str) -> str:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Blender executable does not exist: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"Blender executable {executable!r} was not found in PATH. "
            "Install Blender or pass --blender /absolute/path/to/blender."
        )
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake MotionBERT SMPL motion into an animated FBX using Blender.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing SMPL .npy files and metadata.json.")
    parser.add_argument("--base-fbx", type=Path, default=DEFAULT_BASE_FBX, help="SMPL T-pose source armature FBX.")
    parser.add_argument("--blender-script", type=Path, default=DEFAULT_BLENDER_SCRIPT, help="Blender Python bake script.")
    parser.add_argument("--output", type=Path, default=None, help="Output FBX. Defaults to <input-dir>/retarget/animated_smpl.fbx.")
    parser.add_argument("--blender", default="blender", help="Blender executable name or absolute path.")
    parser.add_argument("--root-scale", type=float, default=0.1, help="Scale from MotionBERT millimeters to the base rig centimeter units.")
    parser.add_argument(
        "--root-trajectory",
        choices=ROOT_TRAJECTORY_MODES,
        default="pose3d",
        help="Root translation source. Default: pose3d.",
    )
    parser.add_argument(
        "--pose3d-scale-mm",
        type=float,
        default=None,
        help="Manual millimeters per Pose3D unit. Default: estimate from SMPL bone lengths.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output = export_animated_smpl_fbx(
            BlenderExportConfig(
                input_dir=args.input_dir,
                base_fbx=args.base_fbx,
                blender_script=args.blender_script,
                output=args.output,
                blender_executable=args.blender,
                root_scale=args.root_scale,
                root_trajectory=args.root_trajectory,
                pose3d_scale_mm=args.pose3d_scale_mm,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved animated SMPL FBX: {output}")


if __name__ == "__main__":
    main()
