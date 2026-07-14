from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}


@dataclass(frozen=True)
class JobPaths:
    root: Path
    input_dir: Path
    work_dir: Path
    artifacts_dir: Path
    logs_dir: Path

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / "report.json"

    @property
    def fbx_path(self) -> Path:
        return self.artifacts_dir / "animated_smpl.fbx"

    @property
    def preview_path(self) -> Path:
        return self.artifacts_dir / "fbx_preview.mp4"



class ServiceStorage:
    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def paths(self, job_id: str) -> JobPaths:
        if not job_id or any(char in job_id for char in "/\\.."):
            raise ValueError(f"Invalid job id {job_id!r}.")
        root = self.storage_root / "jobs" / job_id
        return JobPaths(
            root=root,
            input_dir=root / "input",
            work_dir=root / "work",
            artifacts_dir=root / "artifacts",
            logs_dir=root / "logs",
        )

    def prepare_job(self, job_id: str) -> JobPaths:
        paths = self.paths(job_id)
        for directory in (paths.input_dir, paths.work_dir, paths.artifacts_dir, paths.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    def input_path_for_name(self, job_id: str, filename: str) -> Path:
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension {extension!r}; expected one of "
                f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
            )
        return self.paths(job_id).input_dir / f"source{extension}"

    def save_stream(self, job_id: str, filename: str, source: BinaryIO, max_bytes: int) -> Path:
        paths = self.prepare_job(job_id)
        target = self.input_path_for_name(job_id, filename)
        written = 0
        with target.open("wb") as handle:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    target.unlink(missing_ok=True)
                    raise ValueError(f"Upload exceeds max size of {max_bytes} bytes.")
                handle.write(chunk)
        if written == 0:
            target.unlink(missing_ok=True)
            raise ValueError("Uploaded video is empty.")
        for directory in (paths.work_dir, paths.artifacts_dir, paths.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return target

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self.paths(job_id).root, ignore_errors=True)

