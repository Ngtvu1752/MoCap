from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retarget.fbx_exporter import (
    BlenderExportConfig,
    _validate_inputs,
    build_blender_command,
    export_animated_smpl_fbx,
    parse_args,
)


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

    def test_validate_inputs_rejects_pose3d_frame_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            theta = root / "smpl_theta.npy"
            joints = root / "smpl_joints3d.npy"
            pose3d = root / "pose3d.npy"
            metadata = root / "metadata.json"
            np.save(theta, np.zeros((12, 82), dtype=np.float32))
            np.save(joints, np.zeros((12, 17, 3), dtype=np.float32))
            np.save(pose3d, np.zeros((11, 17, 3), dtype=np.float32))
            metadata.write_text(json.dumps({"video": {"fps": 30.0, "frame_count": 12}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pose3d=11"):
                _validate_inputs(theta, joints, metadata, pose3d)

    def test_default_pose3d_mode_does_not_fallback_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "smpl_theta.npy", np.zeros((2, 82), dtype=np.float32))
            np.save(root / "smpl_joints3d.npy", np.zeros((2, 17, 3), dtype=np.float32))
            (root / "metadata.json").write_text(
                json.dumps({"video": {"fps": 30.0, "frame_count": 2}}),
                encoding="utf-8",
            )
            base_fbx = root / "base.fbx"
            blender_script = root / "bake.py"
            base_fbx.write_bytes(b"test")
            blender_script.write_text("# test", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "pose3d.npy"):
                export_animated_smpl_fbx(
                    BlenderExportConfig(
                        input_dir=root,
                        base_fbx=base_fbx,
                        blender_script=blender_script,
                    )
                )

    def test_build_blender_command_contains_headless_arguments(self) -> None:
        command = build_blender_command(
            "/usr/bin/blender",
            Path("script.py"),
            Path("base.fbx"),
            Path("theta.npy"),
            Path("joints.npy"),
            Path("animated.fbx"),
            fps=30.0,
            root_scale=0.1,
        )

        self.assertEqual(command[:4], ["/usr/bin/blender", "--background", "--python", "script.py"])
        self.assertIn("--fps", command)
        self.assertIn("30", command)
        self.assertIn("--root-scale", command)
        self.assertIn("0.1", command)
        self.assertIn("--root-translation", command)

    def test_pose3d_is_the_default_root_trajectory(self) -> None:
        args = parse_args(["--input-dir", "output/test"])

        self.assertEqual(args.root_trajectory, "pose3d")
        self.assertIsNone(args.pose3d_scale_mm)


if __name__ == "__main__":
    unittest.main()
