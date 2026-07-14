from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from service.core.models import JobRecord, JobRequest, utc_now_iso, validate_status


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def init(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    input_path TEXT,
                    output_fbx_path TEXT,
                    report_path TEXT,
                    preview_path TEXT,
                    error TEXT,
                    rq_job_id TEXT,
                    request_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_job(self, job_id: str, input_path: Path, request: JobRequest) -> JobRecord:
        now = utc_now_iso()
        record = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            input_path=str(input_path),
            request=request.to_dict(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, updated_at, progress, input_path,
                    output_fbx_path, report_path, preview_path, error, rq_job_id, request_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.progress,
                    record.input_path,
                    record.output_fbx_path,
                    record.report_path,
                    record.preview_path,
                    record.error,
                    record.rq_job_id,
                    json.dumps(record.request, sort_keys=True),
                ),
            )
            conn.commit()
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def update_job(self, job_id: str, **fields: Any) -> JobRecord:
        if "status" in fields:
            validate_status(str(fields["status"]))
        if "request" in fields:
            fields["request_json"] = json.dumps(fields.pop("request"), sort_keys=True)
        fields["updated_at"] = utc_now_iso()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [self._serialize(value) for value in fields.values()]
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", values)
            conn.commit()
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(f"Job not found: {job_id}")
        return record

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._record_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _record_from_row(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            progress=float(row["progress"]),
            input_path=row["input_path"],
            output_fbx_path=row["output_fbx_path"],
            report_path=row["report_path"],
            preview_path=row["preview_path"],
            error=row["error"],
            rq_job_id=row["rq_job_id"],
            request=json.loads(row["request_json"]),
        )

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        return value

