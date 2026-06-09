from __future__ import annotations

import numpy as np

from src.pose3d.base import Pose3DEstimator, Pose3DResult


class PoseMambaEstimator(Pose3DEstimator):
    """Adapter boundary for PoseMamba or compatible 2D-to-3D pose lifters."""

    def lift(self, pose2d: np.ndarray) -> Pose3DResult:
        raise NotImplementedError(
            "PoseMamba backend is not configured yet. Implement checkpoint loading "
            "and sequence inference inside PoseMambaEstimator."
        )
