from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pose3DResult:
    keypoints: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class Pose3DEstimator(ABC):
    @abstractmethod
    def lift(self, pose2d: np.ndarray) -> Pose3DResult:
        """Lift normalized/model-specific 2D pose into 3D pose."""
