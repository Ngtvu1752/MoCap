from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RenderResult:
    skeleton_video_path: Path
    mesh_video_path: Path


class PoseRenderer(ABC):
    @abstractmethod
    def render(self, pose3d: np.ndarray, output_path: Path, fps: float) -> RenderResult:
        """Render a 3D pose sequence to video files."""
