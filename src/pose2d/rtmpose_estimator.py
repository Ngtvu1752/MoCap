from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.pose2d.base import Pose2DEstimator, Pose2DResult


DEFAULT_CONFIG_PATH = Path("/workspace/MoCap/checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py")
DEFAULT_CHECKPOINT_PATH = Path(
    "/workspace/MoCap/checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth"
)


class RTMPoseEstimator(Pose2DEstimator):
    """MMPose-backed RTMPose estimator.

    The model is loaded lazily on the first frame so importing the pipeline does
    not require MMPose until Phase 2 actually runs.
    """

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda:0",
        bbox: np.ndarray | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.bbox = bbox
        self._model: Any | None = None
        self._inference_topdown: Any | None = None

    def estimate_frame(self, image: np.ndarray, frame_index: int) -> Pose2DResult:
        self._ensure_model_loaded()

        results = self._inference_topdown(self._model, image, bboxes=self.bbox, bbox_format="xyxy")
        if not results:
            raise RuntimeError(f"RTMPose returned no pose result for frame {frame_index}.")

        pred_instances = results[0].pred_instances
        keypoints = np.asarray(pred_instances.keypoints, dtype=np.float32)
        scores = np.asarray(pred_instances.keypoint_scores, dtype=np.float32)

        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]

        return Pose2DResult(
            keypoints=keypoints,
            scores=scores,
            metadata={
                "frame_index": frame_index,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "device": self.device,
                "bbox_mode": "full_frame" if self.bbox is None else "provided_xyxy",
            },
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.config_path.exists():
            raise FileNotFoundError(f"RTMPose config does not exist: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"RTMPose checkpoint does not exist: {self.checkpoint_path}")

        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MMPose is required for RTMPose inference. Install MMPose/MMCV/MMEngine "
                "in this environment before running Phase 2."
            ) from exc

        register_all_modules()
        self._model = init_model(str(self.config_path), str(self.checkpoint_path), device=self.device)
        self._inference_topdown = inference_topdown
