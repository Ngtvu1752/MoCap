from __future__ import annotations

from service.config import ServiceConfig
from service.core.pipeline_runner import process_mocap_job


def process_job(job_id: str) -> dict:
    return process_mocap_job(job_id, ServiceConfig.from_env())

