from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.retarget.legacy.coordinates import MOTIONBERT_SMPL_TO_BVH_ZUP, motionbert_points_to_bvh, motionbert_rotations_to_bvh


class CoordinateConversionTests(unittest.TestCase):
    def test_motionbert_to_bvh_transform_is_proper_rotation(self) -> None:
        self.assertTrue(np.isclose(np.linalg.det(MOTIONBERT_SMPL_TO_BVH_ZUP), 1.0))

    def test_motionbert_to_bvh_points_match_renderer_convention(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)

        converted = motionbert_points_to_bvh(points)

        np.testing.assert_allclose(
            converted,
            np.array([[-1.0, -3.0, -2.0], [4.0, 6.0, -5.0]], dtype=np.float32),
        )

    def test_rotation_conversion_preserves_transformed_action(self) -> None:
        rotation = Rotation.from_euler("xyz", [25.0, -15.0, 70.0], degrees=True).as_matrix().astype(np.float32)
        point = np.array([0.25, -2.0, 1.5], dtype=np.float32)

        converted_rotation = motionbert_rotations_to_bvh(rotation)
        converted_after_source_rotation = motionbert_points_to_bvh(rotation @ point)
        converted_rotation_after_point = converted_rotation @ motionbert_points_to_bvh(point)

        np.testing.assert_allclose(converted_after_source_rotation, converted_rotation_after_point, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
