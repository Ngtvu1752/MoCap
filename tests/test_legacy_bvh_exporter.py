from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.retarget.legacy.bvh_exporter import (
    SourceMotionForBVH,
    _rotation_degrees_for_bvh,
    validate_source_motion_is_fresh,
)


def _source_motion_from_root_rotvec(root_rotvec: np.ndarray) -> SourceMotionForBVH:
    frame_count = root_rotvec.shape[0]
    pose_axis_angle = np.zeros((frame_count, 24, 3), dtype=np.float32)
    pose_axis_angle[:, 0, :] = root_rotvec
    pose_euler_xyz = Rotation.from_rotvec(pose_axis_angle.reshape(-1, 3)).as_euler("xyz").reshape(frame_count, 24, 3)
    return SourceMotionForBVH(
        pose_axis_angle=pose_axis_angle,
        pose_quat=None,
        pose_euler_xyz=pose_euler_xyz.astype(np.float32),
        root_translation=np.zeros((frame_count, 3), dtype=np.float32),
        fps=30.0,
        frame_count=frame_count,
        joint_count=24,
    )


class BVHExporterTests(unittest.TestCase):
    def test_source_yaw_360_becomes_bvh_z_yaw_without_roll_flip(self) -> None:
        # Source up is negative Y in the renderer/BVH mapping, so a source-up yaw
        # should land on BVH Z rotation after coordinate conversion.
        angles = np.linspace(0.0, 2.0 * np.pi, 20, dtype=np.float32)
        root_rotvec = np.column_stack([np.zeros_like(angles), -angles, np.zeros_like(angles)])
        motion = _source_motion_from_root_rotvec(root_rotvec)

        euler_degrees = _rotation_degrees_for_bvh(motion, root_rotation="source")
        root_euler = euler_degrees[:, 0, :]

        self.assertLess(float(np.max(np.abs(root_euler[:, :2]))), 1e-3)
        self.assertGreater(float(root_euler[-1, 2]), 359.0)
        self.assertLess(float(np.max(np.abs(np.diff(root_euler[:, 2])))), 30.0)

    def test_relative_yaw_keeps_only_bvh_z_heading(self) -> None:
        root_rot = Rotation.from_euler("xyz", [[30.0, 0.0, 0.0], [30.0, 0.0, 90.0]], degrees=True)
        motion = _source_motion_from_root_rotvec(root_rot.as_rotvec().astype(np.float32))

        euler_degrees = _rotation_degrees_for_bvh(motion, root_rotation="relative_yaw")
        root_euler = euler_degrees[:, 0, :]

        np.testing.assert_allclose(root_euler[:, :2], 0.0, atol=1e-4)

    def test_validate_source_motion_is_fresh_rejects_stale_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "output"
            retarget_dir = input_dir / "retarget"
            retarget_dir.mkdir(parents=True)
            source_motion = retarget_dir / "source_motion.npz"
            source_motion.write_bytes(b"old")

            for name in ["smpl_theta.npy", "smpl_joints3d.npy", "metadata.json"]:
                (input_dir / name).write_bytes(b"new")

            old_time = 1_700_000_000
            new_time = old_time + 10
            os.utime(source_motion, (old_time, old_time))
            for name in ["smpl_theta.npy", "smpl_joints3d.npy", "metadata.json"]:
                os.utime(input_dir / name, (new_time, new_time))

            with self.assertRaisesRegex(RuntimeError, "--build-source-motion"):
                validate_source_motion_is_fresh(input_dir, source_motion)


if __name__ == "__main__":
    unittest.main()
