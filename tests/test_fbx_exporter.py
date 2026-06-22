from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retarget.fbx_exporter import _validate_inputs, build_blender_command


class PipelineTriggerTests(unittest.TestCase):
    def test_validate_inputs_returns_metadata_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            theta = root / "smpl_theta.npy"
            joints = root / "smpl_joints3d.npy"
            metadata = root / "metadata.json"
            np.save(theta, np.zeros((12, 82), dtype=np.float32))
            np.save(joints, np.zeros((12, 17, 3), dtype=np.float32))
            metadata.write_text(json.dumps({"video": {"fps": 30.0, "frame_count": 12}}), encoding="utf-8")

            self.assertEqual(_validate_inputs(theta, joints, metadata), 30.0)

    def test_validate_inputs_rejects_frame_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            theta = root / "smpl_theta.npy"
            joints = root / "smpl_joints3d.npy"
            metadata = root / "metadata.json"
            np.save(theta, np.zeros((12, 82), dtype=np.float32))
            np.save(joints, np.zeros((11, 17, 3), dtype=np.float32))
            metadata.write_text(json.dumps({"video": {"fps": 30.0, "frame_count": 12}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Frame count mismatch"):
                _validate_inputs(theta, joints, metadata)

    def test_build_blender_command_contains_headless_arguments(self) -> None:
        command = build_blender_command(
            "/usr/bin/blender",
            Path("script.py"),
            Path("base.fbx"),
            Path("theta.npy"),
            Path("joints.npy"),
            Path("animated.fbx"),
            fps=30.0,
            root_scale=0.001,
        )

        self.assertEqual(command[:4], ["/usr/bin/blender", "--background", "--python", "script.py"])
        self.assertIn("--fps", command)
        self.assertIn("30", command)
        self.assertIn("--root-scale", command)
        self.assertIn("0.001", command)


if __name__ == "__main__":
    unittest.main()
