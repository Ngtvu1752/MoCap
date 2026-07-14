from __future__ import annotations

import sys

from service.config import ServiceConfig
from service.core.environment import check_runtime_environment
from service.core.database import JobStore


def main() -> None:
    config = ServiceConfig.from_env()
    config.storage_root.mkdir(parents=True, exist_ok=True)
    JobStore(config.database_path).init()
    errors = check_runtime_environment(config)
    if errors:
        print("MoCap service doctor failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("MoCap service doctor passed.")
    print(f"Storage: {config.storage_root.resolve()}")
    print(f"Database: {config.database_path.resolve()}")
    print(f"Queue: {config.queue_name} @ {config.redis_url}")


if __name__ == "__main__":
    main()

