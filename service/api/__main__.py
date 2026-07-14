from __future__ import annotations

import socket

from service.config import ServiceConfig


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install service dependencies from requirements-service.txt before running API.") from exc

    config = ServiceConfig.from_env()
    host = config.api_host
    port = _select_port(host, config.api_port)
    if port != config.api_port:
        print(f"Port {config.api_port} is already in use; using {port} instead.")
    print(f"Starting MoCap API server on {host}:{port}...")
    print(f"Open http://localhost:{port}/ in your browser.")
    uvicorn.run("service.api.app:app", host=host, port=port, reload=False)


def _select_port(host: str, requested_port: int) -> int:
    for port in range(requested_port, requested_port + 20):
        if _port_is_free(host, port):
            return port
    raise SystemExit(
        f"No free API port found from {requested_port} to {requested_port + 19}. "
        "Set MOCAP_API_PORT to a free port."
    )


def _port_is_free(host: str, port: int) -> bool:
    bind_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


if __name__ == "__main__":
    main()

