from __future__ import annotations

from datetime import datetime, timedelta, timezone

from service.config import ServiceConfig
from service.core.database import JobStore
from service.core.models import TERMINAL_STATUSES
from service.core.storage import ServiceStorage


def cleanup_expired_jobs(config: ServiceConfig) -> list[str]:
    store = JobStore(config.database_path)
    store.init()
    storage = ServiceStorage(config.storage_root)
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.retention_days)
    expired: list[str] = []
    for job in store.list_jobs():
        created_at = datetime.fromisoformat(job.created_at)
        if job.status in TERMINAL_STATUSES and created_at < cutoff:
            storage.delete_job(job.job_id)
            store.update_job(job.job_id, status="expired", progress=1.0)
            expired.append(job.job_id)
    return expired

