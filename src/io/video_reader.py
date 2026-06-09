from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import cv2

@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_seconds: float
    image: np.ndarray


class VideoReader:
    """Small OpenCV-backed reader that exposes metadata and frame iteration."""

    def __init__(self, video_path: Path | str) -> None:
        self.video_path = Path(video_path)
        self._capture: Any | None = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def open(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self._capture = capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def metadata(self) -> VideoMetadata:
        capture = self._require_capture()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=self.video_path,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )

    def iter_frames(self) -> Iterator[VideoFrame]:
        capture = self._require_capture()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        index = 0

        while True:
            ok, image = capture.read()
            if not ok:
                break

            timestamp = index / fps if fps > 0 else 0.0
            yield VideoFrame(index=index, timestamp_seconds=timestamp, image=image)
            index += 1

    def _require_capture(self) -> Any:
        if self._capture is None:
            raise RuntimeError("VideoReader is not open. Use `with VideoReader(...)` or call open().")
        return self._capture
