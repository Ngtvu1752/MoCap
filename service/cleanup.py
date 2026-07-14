from __future__ import annotations

from service.config import ServiceConfig
from service.core.cleanup import cleanup_expired_jobs


def main() -> None:
    config = ServiceConfig.from_env()
    expired = cleanup_expired_jobs(config)
    print(f"Expired jobs cleaned: {len(expired)}")
    for job_id in expired:
        print(job_id)


if __name__ == "__main__":
    main()

