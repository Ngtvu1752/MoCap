from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class KeypointFormat(str, Enum):
    WHOLE_BODY133 = "whole_body133"
    RTMPOSE_RAW = "whole_body133"
    HUMAN36M_17 = "human36m_17"


class CocoBody17:
    """COCO body joint indices used by RTMPose whole-body outputs.

    For COCO-WholeBody, the first 17 joints are the standard COCO body joints.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Human36M17:
    """Human3.6M 17-joint order used by many 2D-to-3D pose lifters.

    Order:
    0 pelvis/root, 1 right hip, 2 right knee, 3 right ankle,
    4 left hip, 5 left knee, 6 left ankle, 7 spine, 8 thorax,
    9 neck, 10 head, 11 left shoulder, 12 left elbow, 13 left wrist,
    14 right shoulder, 15 right elbow, 16 right wrist.
    """

    PELVIS = 0
    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_ANKLE = 3
    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


@dataclass(frozen=True)
class Pose2DFormatConverter:
    source_format: KeypointFormat
    target_format: KeypointFormat

    def convert(self, pose2d: np.ndarray) -> np.ndarray:
        if self.source_format == self.target_format:
            return pose2d

        if (
            self.source_format == KeypointFormat.WHOLE_BODY133
            and self.target_format == KeypointFormat.HUMAN36M_17
        ):
            return self._whole_body133_to_human36m_17(pose2d)

        raise ValueError(f"Unsupported conversion: {self.source_format} -> {self.target_format}")

    def _whole_body133_to_human36m_17(self, pose2d: np.ndarray) -> np.ndarray:
        pose2d = np.asarray(pose2d, dtype=np.float32)
        self._validate_whole_body133(pose2d)

        coco = pose2d[:, :17, :]
        h36m = np.zeros((pose2d.shape[0], 17, 2), dtype=np.float32)

        left_hip = coco[:, CocoBody17.LEFT_HIP]
        right_hip = coco[:, CocoBody17.RIGHT_HIP]
        left_shoulder = coco[:, CocoBody17.LEFT_SHOULDER]
        right_shoulder = coco[:, CocoBody17.RIGHT_SHOULDER]

        pelvis = midpoint(left_hip, right_hip)
        thorax = midpoint(left_shoulder, right_shoulder)
        spine = midpoint(pelvis, thorax)

        h36m[:, Human36M17.PELVIS] = pelvis
        h36m[:, Human36M17.RIGHT_HIP] = right_hip
        h36m[:, Human36M17.RIGHT_KNEE] = coco[:, CocoBody17.RIGHT_KNEE]
        h36m[:, Human36M17.RIGHT_ANKLE] = coco[:, CocoBody17.RIGHT_ANKLE]
        h36m[:, Human36M17.LEFT_HIP] = left_hip
        h36m[:, Human36M17.LEFT_KNEE] = coco[:, CocoBody17.LEFT_KNEE]
        h36m[:, Human36M17.LEFT_ANKLE] = coco[:, CocoBody17.LEFT_ANKLE]
        h36m[:, Human36M17.SPINE] = spine
        head = coco[:, CocoBody17.NOSE]
        neck = midpoint(thorax, head)

        h36m[:, Human36M17.THORAX] = thorax
        h36m[:, Human36M17.NECK] = neck
        h36m[:, Human36M17.HEAD] = head
        h36m[:, Human36M17.LEFT_SHOULDER] = left_shoulder
        h36m[:, Human36M17.LEFT_ELBOW] = coco[:, CocoBody17.LEFT_ELBOW]
        h36m[:, Human36M17.LEFT_WRIST] = coco[:, CocoBody17.LEFT_WRIST]
        h36m[:, Human36M17.RIGHT_SHOULDER] = right_shoulder
        h36m[:, Human36M17.RIGHT_ELBOW] = coco[:, CocoBody17.RIGHT_ELBOW]
        h36m[:, Human36M17.RIGHT_WRIST] = coco[:, CocoBody17.RIGHT_WRIST]

        return h36m

    def _validate_whole_body133(self, pose2d: np.ndarray) -> None:
        if pose2d.ndim != 3:
            raise ValueError(f"Expected pose2d shape (T, K, 2), got {pose2d.shape}")
        if pose2d.shape[1] < 133:
            raise ValueError(f"Expected at least 133 WholeBody133 joints, got {pose2d.shape[1]}")
        if pose2d.shape[2] != 2:
            raise ValueError(f"Expected xy coordinates in last dimension, got {pose2d.shape}")


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5
