from __future__ import annotations

import sys

from service.config import ServiceConfig
from service.core.database import JobStore


def main() -> None:
    try:
        from redis import Redis
        from rq import Worker
    except ImportError as exc:
        raise SystemExit("Install service dependencies from requirements-service.txt before running worker.") from exc

    config = ServiceConfig.from_env()
    JobStore(config.database_path).init()
    connection = Redis.from_url(config.redis_url)
    worker = Worker([config.queue_name], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()

