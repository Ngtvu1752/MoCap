from __future__ import annotations

import numpy as np

from src.pose2d.base import Pose2DEstimator, Pose2DResult


class RTMPoseEstimator(Pose2DEstimator):
    """Adapter boundary for RTMPose.

    This class intentionally does not lock the project to one RTMPose checkpoint
    or runtime. Add the concrete backend here later, while the pipeline keeps
    depending only on Pose2DEstimator.
    """

    def estimate_frame(self, image: np.ndarray, frame_index: int) -> Pose2DResult:
        raise NotImplementedError(
            "RTMPose backend is not configured yet. Implement checkpoint loading "
            "and inference inside RTMPoseEstimator."
        )
