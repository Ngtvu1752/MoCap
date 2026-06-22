from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_BASE_FBX = Path("assets/retarget/smpl_base_tpose.fbx")
DEFAULT_BLENDER_SCRIPT = Path("src/retarget/blender/bake_smpl_fbx.py")


@dataclass(frozen=True)
class BlenderExportConfig:
    input_dir: Path
    base_fbx: Path = DEFAULT_BASE_FBX
    blender_script: Path = DEFAULT_BLENDER_SCRIPT
    output: Path | None = None
    blender_executable: str = "blender"
    root_scale: float = 0.001


def export_animated_smpl_fbx(config: BlenderExportConfig) -> Path:
    input_dir = config.input_dir.resolve()
    theta_path = input_dir / "smpl_theta.npy"
    joints_path = input_dir / "smpl_joints3d.npy"
    metadata_path = input_dir / "metadata.json"
    output_path = (config.output or input_dir / "retarget" / "animated_smpl.fbx").resolve()
    base_fbx = config.base_fbx.resolve()
    blender_script = config.blender_script.resolve()

    _validate_paths(theta_path, joints_path, metadata_path, base_fbx, blender_script)
    fps = _validate_inputs(theta_path, joints_path, metadata_path)
    blender = _resolve_blender(config.blender_executable)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_blender_command(
        blender,
        blender_script,
        base_fbx,
        theta_path,
        joints_path,
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
    joints_path: Path,
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
        "--joints",
        str(joints_path),
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


def _validate_inputs(theta_path: Path, joints_path: Path, metadata_path: Path) -> float:
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

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata["video"]["fps"])
    frame_count = int(metadata["video"]["frame_count"])
    if fps <= 0:
        raise ValueError(f"Expected positive video FPS, got {fps}.")
    if frame_count != theta.shape[0]:
        raise ValueError(f"Metadata frame_count={frame_count}, but motion contains {theta.shape[0]} frames.")
    return fps


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
    parser.add_argument("--root-scale", type=float, default=0.001, help="Scale from MotionBERT millimeters to Blender units.")
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
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved animated SMPL FBX: {output}")


if __name__ == "__main__":
    main()
