from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshResult:
    vertices: np.ndarray
    joints3d: np.ndarray
    theta: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanMeshEstimator(ABC):
    @abstractmethod
    def recover(self, pose2d: np.ndarray, scores: np.ndarray | None = None) -> MeshResult:
        """Recover a SMPL mesh sequence from 2D keypoints."""
