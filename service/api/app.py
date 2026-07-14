from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError as exc:  # pragma: no cover - exercised by doctor.
    raise RuntimeError("Install service dependencies from requirements-service.txt before running API.") from exc

from service.config import ServiceConfig
from service.core.database import JobStore
from service.core.models import JobRequest
from service.core.queue import enqueue_mocap_job
from service.core.storage import ServiceStorage
from service.core.validation import validate_upload_name
from src.pipeline_defaults import DEFAULT_MESH_CLIP_LEN, DEFAULT_MESH_CLIP_STRIDE, DEFAULT_POSE2D_FORMAT, SUPPORTED_POSE2D_FORMATS


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    config = config or ServiceConfig.from_env()
    store = JobStore(config.database_path)
    storage = ServiceStorage(config.storage_root)
    app = FastAPI(title="Offline AI MoCap FBX Generation Service", version="0.1.0")

    def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        if not x_api_key or x_api_key != config.api_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")

    @app.on_event("startup")
    def startup() -> None:
        store.init()
        config.storage_root.mkdir(parents=True, exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html", media_type="text/html")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}


    @app.get("/ready")
    def ready(_: None = Depends(require_api_key)) -> Dict[str, object]:
        try:
            store.init()
            return {"ready": True, "queue": config.queue_name}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/jobs")
    def create_job(
        video: UploadFile = File(...),
        profile: str = Form("unity_humanoid_fbx"),
        pose2d_format: str = Form(DEFAULT_POSE2D_FORMAT),
        mesh_clip_len: int = Form(DEFAULT_MESH_CLIP_LEN),
        mesh_clip_stride: int = Form(DEFAULT_MESH_CLIP_STRIDE),
        root_trajectory: str = Form("pose3d"),
        _: None = Depends(require_api_key),
    ) -> Dict[str, object]:
        if profile != "unity_humanoid_fbx":
            raise HTTPException(status_code=400, detail="Only profile='unity_humanoid_fbx' is supported.")
        if pose2d_format not in SUPPORTED_POSE2D_FORMATS:
            raise HTTPException(status_code=400, detail=f"pose2d_format must be one of {SUPPORTED_POSE2D_FORMATS}.")
        if root_trajectory not in {"pose3d", "smpl", "zero"}:
            raise HTTPException(status_code=400, detail="root_trajectory must be pose3d, smpl, or zero.")
        filename = video.filename or "source.mp4"
        try:
            validate_upload_name(filename)
            job_id = storage.new_job_id()
            input_path = storage.save_stream(job_id, filename, video.file, config.max_upload_bytes)
            request = JobRequest(
                profile=profile,
                pose2d_format=pose2d_format,
                mesh_clip_len=mesh_clip_len,
                mesh_clip_stride=mesh_clip_stride,
                root_trajectory=root_trajectory,
            )
            record = store.create_job(job_id, input_path, request)
            rq_job_id = enqueue_mocap_job(job_id, config)
            record = store.update_job(job_id, rq_job_id=rq_job_id)
            return record.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(require_api_key)) -> Dict[str, object]:
        record = store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return record.to_dict()

    @app.get("/v1/jobs/{job_id}/download/fbx")
    def download_fbx(job_id: str, _: None = Depends(require_api_key)) -> FileResponse:
        record = _done_job_or_404(store, job_id)
        return _file_response(record.output_fbx_path, "animated_smpl.fbx")


    @app.get("/v1/jobs/{job_id}/download/source")
    def download_source(job_id: str, _: None = Depends(require_api_key)) -> FileResponse:
        record = store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        suffix = Path(record.input_path or "source.mp4").suffix or ".mp4"
        return _file_response(record.input_path, f"source{suffix}")

    @app.get("/v1/jobs/{job_id}/report")
    def download_report(job_id: str, _: None = Depends(require_api_key)) -> FileResponse:
        record = store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _file_response(record.report_path, "report.json")

    @app.get("/v1/jobs/{job_id}/download/preview")
    def download_preview(job_id: str, _: None = Depends(require_api_key)) -> FileResponse:
        record = _done_job_or_404(store, job_id)
        return _file_response(record.preview_path, "preview.mp4")

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str, _: None = Depends(require_api_key)) -> Dict[str, str]:
        record = store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        storage.delete_job(job_id)
        store.update_job(job_id, status="canceled", progress=1.0)
        return {"job_id": job_id, "status": "canceled"}

    return app


def _done_job_or_404(store: JobStore, job_id: str):
    record = store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.status != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done: {record.status}.")
    return record


def _file_response(path: Optional[str], filename: str) -> FileResponse:
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(path, filename=filename)


app = create_app()

