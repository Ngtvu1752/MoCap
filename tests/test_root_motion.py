from __future__ import annotations

import unittest

import numpy as np

from src.retarget.root_motion import (
    RootMotionConfig,
    build_pose3d_root_motion,
    estimate_pose3d_scale_mm,
    optimize_root_trajectory,
    source_to_world_zup,
)


BASE_SKELETON = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0], [-0.1, 0.45, 0.0], [-0.1, 0.9, 0.0],
        [0.1, 0.0, 0.0], [0.1, 0.45, 0.0], [0.1, 0.9, 0.0],
        [0.0, -0.25, 0.0], [0.0, -0.5, 0.0], [0.0, -0.65, 0.0], [0.0, -0.8, 0.0],
        [0.2, -0.5, 0.0], [0.45, -0.5, 0.0], [0.7, -0.5, 0.0],
        [-0.2, -0.5, 0.0], [-0.45, -0.5, 0.0], [-0.7, -0.5, 0.0],
    ],
    dtype=np.float64,
)


def repeated_skeleton(frame_count: int, scale: float = 1.0) -> np.ndarray:
    return np.repeat((BASE_SKELETON * scale)[None], frame_count, axis=0)


class RootMotionTests(unittest.TestCase):
    def test_estimate_scale_from_matching_skeletons(self) -> None:
        pose3d = repeated_skeleton(20)
        smpl = repeated_skeleton(20, scale=950.0)

        scale = estimate_pose3d_scale_mm(pose3d, smpl)

        self.assertAlmostEqual(scale, 950.0, places=5)

    def test_source_coordinate_conversion_matches_renderer(self) -> None:
        converted = source_to_world_zup(np.array([[1.0, 2.0, 3.0]]))

        np.testing.assert_allclose(converted, [[-1.0, -3.0, -2.0]])

    def test_constant_velocity_keeps_direction_and_starts_at_origin(self) -> None:
        frame_count = 90
        pose3d = repeated_skeleton(frame_count)
        pose3d[:, :, 0] += np.linspace(0.0, 1.0, frame_count)[:, None]
        smpl = repeated_skeleton(frame_count, scale=1000.0)

        result = build_pose3d_root_motion(
            pose3d,
            smpl,
            30.0,
            scale_mm_per_pose_unit=1000.0,
        )

        np.testing.assert_allclose(result.translation_source_mm[0], 0.0, atol=1e-5)
        self.assertGreater(result.translation_source_mm[-1, 0], 950.0)
        self.assertLess(float(np.max(np.abs(result.translation_source_mm[:, 1:]))), 1e-3)

    def test_contact_optimization_reduces_world_foot_sliding(self) -> None:
        frame_count = 40
        raw_root = np.zeros((frame_count, 3), dtype=np.float64)
        left_local = np.zeros_like(raw_root)
        right_local = np.zeros_like(raw_root)
        left_local[:, 0] = np.linspace(0.0, 0.2, frame_count)
        contact = np.ones(frame_count, dtype=bool)
        no_contact = np.zeros(frame_count, dtype=bool)
        config = RootMotionConfig(smoothness_weight=0.0, contact_weight=50.0)

        cleaned = optimize_root_trajectory(
            raw_root,
            left_local,
            right_local,
            contact,
            no_contact,
            floor_height=0.0,
            config=config,
        )

        raw_slide = np.ptp((raw_root + left_local)[:, 0])
        cleaned_slide = np.ptp((cleaned + left_local)[:, 0])
        self.assertLess(cleaned_slide, raw_slide * 0.1)

    def test_airborne_jump_is_not_flattened(self) -> None:
        frame_count = 61
        phase = np.linspace(0.0, np.pi, frame_count)
        raw_root = np.zeros((frame_count, 3), dtype=np.float64)
        raw_root[:, 2] = np.sin(phase) * 0.6
        raw_root[:4, 2] = 0.0
        raw_root[-4:, 2] = 0.0
        feet = np.zeros_like(raw_root)
        contact = np.zeros(frame_count, dtype=bool)
        contact[:4] = True
        contact[-4:] = True

        cleaned = optimize_root_trajectory(
            raw_root,
            feet,
            feet,
            contact,
            contact,
            floor_height=0.0,
            config=RootMotionConfig(),
        )

        self.assertGreater(float(cleaned[:, 2].max()), 0.5)
        self.assertLess(float(np.max(np.abs(cleaned[:4, 2]))), 0.02)
        self.assertLess(float(np.max(np.abs(cleaned[-4:, 2]))), 0.02)

    def test_rejects_frame_mismatch_and_invalid_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "Frame count mismatch"):
            build_pose3d_root_motion(repeated_skeleton(5), repeated_skeleton(4), 30.0)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            build_pose3d_root_motion(
                repeated_skeleton(5),
                repeated_skeleton(5),
                30.0,
                scale_mm_per_pose_unit=0.0,
            )


if __name__ == "__main__":
    unittest.main()
