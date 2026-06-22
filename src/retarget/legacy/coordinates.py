from __future__ import annotations

import numpy as np


SOURCE_COORDINATE_SYSTEM = "motionbert_smpl"
BVH_COORDINATE_SYSTEM = "bvh_zup"

# Match the existing MotionBERT visualizer convention: (x, y, z) -> (-x, -z, -y).
MOTIONBERT_SMPL_TO_BVH_ZUP = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def motionbert_points_to_bvh(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points @ MOTIONBERT_SMPL_TO_BVH_ZUP.T


def motionbert_rotations_to_bvh(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    transform = MOTIONBERT_SMPL_TO_BVH_ZUP.astype(np.float32)
    return transform @ rotations @ transform.T
