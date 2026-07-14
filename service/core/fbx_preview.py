from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DEFAULT_PREVIEW_SCRIPT = Path("src/retarget/blender/render_fbx_preview.py")


def render_fbx_preview(
    *,
    blender_executable: str,
    fbx_path: Path,
    output_path: Path,
    script_path: Path = DEFAULT_PREVIEW_SCRIPT,
    fps: float | None = None,
) -> Path:
    blender = _resolve_blender(blender_executable)
    script = script_path.resolve()
    if not script.exists():
        raise FileNotFoundError(f"FBX preview Blender script does not exist: {script}")
    if not fbx_path.exists() or fbx_path.stat().st_size == 0:
        raise FileNotFoundError(f"FBX file does not exist or is empty: {fbx_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        blender,
        "--background",
        "--python",
        str(script),
        "--",
        "--fbx",
        str(fbx_path.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    if fps is not None:
        command.extend(["--fps", f"{fps:.10g}"])
    subprocess.run(command, check=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Blender completed without creating a valid FBX preview: {output_path}")
    return output_path


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
            "Install Blender or set MOCAP_BLENDER=/absolute/path/to/blender."
        )
    return resolved

