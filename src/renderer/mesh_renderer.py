from __future__ import annotations

from pathlib import Path

import numpy as np

from src.renderer.base import PoseRenderer


class SkeletonRenderer(PoseRenderer):
    """Placeholder renderer boundary for skeleton or mesh video output."""

    def render(self, pose3d: np.ndarray, output_path: Path, fps: float) -> Path:
        raise NotImplementedError(
            "Renderer is not implemented yet. Start with a simple 3D skeleton renderer, "
            "then replace or extend it with mesh rendering later."
        )
