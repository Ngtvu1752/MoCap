from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from service.config import ServiceConfig
from service.core.storage import ALLOWED_VIDEO_EXTENSIONS


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None
    codec_name: str | None
    pixel_format: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "codec_name": self.codec_name,
            "pixel_format": self.pixel_format,
        }


def validate_upload_name(filename: str) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension {extension!r}; expected one of "
            f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
        )


def probe_video(video_path: Path) -> VideoProbe:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found in PATH.")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,codec_name,pix_fmt:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {video_path}.")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = _parse_fps(stream.get("avg_frame_rate", "0/0"))
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    frame_count = stream.get("nb_frames")
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        frame_count=int(frame_count) if frame_count and str(frame_count).isdigit() else None,
        codec_name=stream.get("codec_name"),
        pixel_format=stream.get("pix_fmt"),
    )


def validate_video_for_service(video_path: Path, config: ServiceConfig) -> VideoProbe:
    probe = probe_video(video_path)
    if probe.fps <= 0:
        raise ValueError(f"Could not read a positive FPS from {video_path}.")
    if probe.duration_seconds <= 0:
        raise ValueError(f"Could not read a positive duration from {video_path}.")
    if probe.duration_seconds > config.max_duration_seconds:
        raise ValueError(
            f"Video duration {probe.duration_seconds:.2f}s exceeds limit "
            f"{config.max_duration_seconds:.2f}s."
        )
    if probe.width > config.max_width or probe.height > config.max_height:
        raise ValueError(
            f"Video resolution {probe.width}x{probe.height} exceeds limit "
            f"{config.max_width}x{config.max_height}."
        )
    return probe


def _parse_fps(raw: str) -> float:
    numerator, _, denominator = raw.partition("/")
    num = float(numerator)
    den = float(denominator or "1")
    if den == 0:
        return 0.0
    return num / den

