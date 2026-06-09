from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class PoseRenderer(ABC):
    @abstractmethod
    def render(self, pose3d: np.ndarray, output_path: Path, fps: float) -> Path:
        """Render a 3D pose sequence to a video file."""
