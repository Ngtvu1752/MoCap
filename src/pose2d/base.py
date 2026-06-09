from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pose2DResult:
    keypoints: np.ndarray
    scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Pose2DEstimator(ABC):
    @abstractmethod
    def estimate_frame(self, image: np.ndarray, frame_index: int) -> Pose2DResult:
        """Estimate raw 2D pose for one frame."""
