from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from service.config import ServiceConfig
from service.core.cleanup import cleanup_expired_jobs
from service.core.database import JobStore
from service.core.models import JobRequest, validate_status
from service.core.pipeline_runner import process_mocap_job
from service.core.report import build_report, write_report
from service.core.storage import ServiceStorage
from service.core.validation import VideoProbe, validate_upload_name


class ServiceCoreTests(unittest.TestCase):
    def test_validate_status_rejects_unknown_status(self) -> None:
        self.assertEqual(validate_status("queued"), "queued")
        with self.assertRaisesRegex(ValueError, "Unknown job status"):
            validate_status("sleeping")

    def test_storage_rejects_invalid_extension_and_empty_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ServiceStorage(Path(temp_dir))
            job_id = "abc123"
            with self.assertRaisesRegex(ValueError, "Unsupported video extension"):
                storage.input_path_for_name(job_id, "source.txt")
            with self.assertRaisesRegex(ValueError, "empty"):
                storage.save_stream(job_id, "source.mp4", _BytesReader(b""), max_bytes=10)

    def test_job_store_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "jobs.sqlite3"
            store = JobStore(db)
            store.init()
            request = JobRequest(mesh_clip_len=243, mesh_clip_stride=121)
            record = store.create_job("job1", Path(temp_dir) / "source.mp4", request)

            self.assertEqual(record.status, "queued")
            updated = store.update_job("job1", status="validating", progress=0.1)
            self.assertEqual(updated.status, "validating")
            self.assertAlmostEqual(updated.progress, 0.1)
            self.assertEqual(store.get_job("job1").request["mesh_clip_stride"], 121)

    def test_report_includes_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "animated_smpl.fbx"
            artifact.write_bytes(b"fbx")
            store = JobStore(root / "jobs.sqlite3")
            store.init()
            job = store.create_job("job1", root / "source.mp4", JobRequest())
            config = ServiceConfig(storage_root=root, database_path=root / "jobs.sqlite3")
            report = build_report(job, config, artifacts={"fbx": artifact})
            report_path = write_report(root / "report.json", report)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["artifacts"]["fbx"]["size_bytes"], 3)
            self.assertEqual(len(payload["artifacts"]["fbx"]["sha256"]), 64)

    def test_mock_pipeline_job_produces_done_report_and_fbx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ServiceConfig(
                repo_root=root,
                storage_root=root / "service_data",
                database_path=root / "service_data" / "jobs.sqlite3",
                device="cpu",
            )
            storage = ServiceStorage(config.storage_root)
            paths = storage.prepare_job("job1")
            input_path = paths.input_dir / "source.mp4"
            input_path.write_bytes(b"video")
            store = JobStore(config.database_path)
            store.init()
            store.create_job("job1", input_path, JobRequest())

            def fake_pipeline(input_path: Path, work_dir: Path, job, config: ServiceConfig) -> Path:
                output = work_dir / input_path.stem
                output.mkdir(parents=True, exist_ok=True)
                return output

            def fake_export(pipeline_output: Path, target: Path, job, config: ServiceConfig):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fbx")
                trajectory = target.parent / "root_trajectory.npy"
                np.save(trajectory, np.array([[0, 0, 0], [1000, 0, 0]], dtype=np.float32))
                return target, trajectory

            def fake_preview(fbx_path: Path, target: Path, config: ServiceConfig, input_metadata):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fbx-preview")
                return target

            fake_probe = VideoProbe(
                width=1280,
                height=720,
                fps=30.0,
                duration_seconds=2.0,
                frame_count=60,
                codec_name="h264",
                pixel_format="yuv420p",
            )
            with patch("service.core.pipeline_runner.missing_asset_paths", return_value=[]), patch(
                "service.core.pipeline_runner.validate_video_for_service", return_value=fake_probe
            ), patch("service.core.pipeline_runner._run_pipeline", side_effect=fake_pipeline), patch(
                "service.core.pipeline_runner._export_fbx", side_effect=fake_export
            ), patch("service.core.pipeline_runner._render_fbx_preview", side_effect=fake_preview):
                report = process_mocap_job("job1", config)

            job = store.get_job("job1")
            self.assertEqual(job.status, "done")
            self.assertTrue(Path(job.output_fbx_path).exists())
            self.assertTrue(Path(job.preview_path).exists())
            self.assertTrue(Path(job.report_path).exists())
            self.assertEqual(report["root_trajectory"]["span_xyz_m"], [1.0, 0.0, 0.0])

    def test_cleanup_expires_old_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ServiceConfig(
                storage_root=root / "service_data",
                database_path=root / "service_data" / "jobs.sqlite3",
                retention_days=7,
            )
            storage = ServiceStorage(config.storage_root)
            paths = storage.prepare_job("job1")
            (paths.artifacts_dir / "animated_smpl.fbx").write_bytes(b"fbx")
            store = JobStore(config.database_path)
            store.init()
            store.create_job("job1", paths.input_dir / "source.mp4", JobRequest())
            old = (datetime.now(timezone.utc) - timedelta(days=8)).replace(microsecond=0).isoformat()
            store.update_job("job1", status="done", created_at=old)

            expired = cleanup_expired_jobs(config)

            self.assertEqual(expired, ["job1"])
            self.assertFalse(paths.root.exists())
            self.assertEqual(store.get_job("job1").status, "expired")


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


if __name__ == "__main__":
    unittest.main()

