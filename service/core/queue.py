from __future__ import annotations

from service.config import ServiceConfig


def enqueue_mocap_job(job_id: str, config: ServiceConfig) -> str:
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        raise RuntimeError("Install service dependencies from requirements-service.txt before using Redis RQ.") from exc

    connection = Redis.from_url(config.redis_url)
    queue = Queue(config.queue_name, connection=connection)
    rq_job = queue.enqueue(
        "service.worker.jobs.process_job",
        job_id,
        job_timeout=config.job_timeout_seconds,
        failure_ttl=config.retention_days * 24 * 3600,
        result_ttl=config.retention_days * 24 * 3600,
    )
    return str(rq_job.id)

