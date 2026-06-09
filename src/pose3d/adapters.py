from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class KeypointFormat(str, Enum):
    RTMPOSE_RAW = "rtmpose_raw"
    HUMAN36M_17 = "human36m_17"


@dataclass(frozen=True)
class Pose2DFormatConverter:
    source_format: KeypointFormat
    target_format: KeypointFormat

    def convert(self, pose2d: np.ndarray) -> np.ndarray:
        if self.source_format == self.target_format:
            return pose2d

        if (
            self.source_format == KeypointFormat.RTMPOSE_RAW
            and self.target_format == KeypointFormat.HUMAN36M_17
        ):
            return self._rtmpose_raw_to_human36m_17(pose2d)

        raise ValueError(f"Unsupported conversion: {self.source_format} -> {self.target_format}")

    def _rtmpose_raw_to_human36m_17(self, pose2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Define the RTMPose raw keypoint layout and mapping to Human3.6M 17 joints."
        )
