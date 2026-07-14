from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from service.config import ServiceConfig
from service.core.database import JobStore
from service.core.environment import missing_asset_paths
from service.core.models import JobRecord
from service.core.report import build_report, write_report
from service.core.storage import ServiceStorage
from service.core.validation import validate_video_for_service


def process_mocap_job(job_id: str, config: ServiceConfig | None = None) -> dict[str, Any]:
    config = config or ServiceConfig.from_env()
    store = JobStore(config.database_path)
    store.init()
    storage = ServiceStorage(config.storage_root)
    paths = storage.prepare_job(job_id)
    job = _require_job(store, job_id)
    stage_timings: dict[str, float] = {}
    input_metadata: dict[str, Any] = {}
    artifacts: dict[str, Path] = {}
    root_trajectory: dict[str, Any] = {}

    try:
        store.update_job(job_id, status="validating", progress=0.05)
        started = time.perf_counter()
        missing = missing_asset_paths(config)
        if missing:
            raise FileNotFoundError("Missing required asset(s): " + ", ".join(str(path) for path in missing))
        if not job.input_path:
            raise ValueError("Job does not have an input video path.")
        input_path = Path(job.input_path)
        probe = validate_video_for_service(input_path, config)
        input_metadata = probe.to_dict()
        stage_timings["validation"] = time.perf_counter() - started
        _write_progress_report(store, job_id, config, input_metadata, stage_timings, artifacts, root_trajectory)

        store.update_job(job_id, status="pose2d", progress=0.15)
        started = time.perf_counter()
        pipeline_output = _run_pipeline(input_path, paths.work_dir, job, config)
        stage_timings["mocap_pipeline"] = time.perf_counter() - started
        store.update_job(job_id, status="fbx_export", progress=0.80)
        started = time.perf_counter()
        fbx_path, trajectory_path = _export_fbx(pipeline_output, paths.fbx_path, job, config)
        stage_timings["fbx_export"] = time.perf_counter() - started
        artifacts["fbx"] = fbx_path
        if trajectory_path.exists():
            artifacts["root_trajectory"] = trajectory_path
            root_trajectory = _root_trajectory_summary(trajectory_path)

        store.update_job(job_id, status="fbx_preview", progress=0.90)
        started = time.perf_counter()
        preview_path = _render_fbx_preview(fbx_path, paths.preview_path, config, input_metadata)
        stage_timings["fbx_preview"] = time.perf_counter() - started
        if preview_path is not None:
            artifacts["fbx_preview"] = preview_path
            store.update_job(job_id, preview_path=str(preview_path))

        store.update_job(job_id, status="packaging", progress=0.95)
        job = store.update_job(
            job_id,
            status="done",
            progress=1.0,
            output_fbx_path=str(fbx_path),
            report_path=str(paths.report_path),
        )
        report = build_report(
            job,
            config,
            input_metadata=input_metadata,
            stage_timings=stage_timings,
            artifacts=artifacts,
            root_trajectory=root_trajectory,
        )
        write_report(paths.report_path, report)
        return report
    except Exception as exc:
        error = str(exc)
        log_tail = traceback.format_exc()
        job = store.update_job(
            job_id,
            status="failed",
            progress=1.0,
            error=error,
            report_path=str(paths.report_path),
        )
        report = build_report(
            job,
            config,
            input_metadata=input_metadata,
            stage_timings=stage_timings,
            artifacts=artifacts,
            root_trajectory=root_trajectory,
            error=error,
            log_tail=log_tail[-8000:],
        )
        write_report(paths.report_path, report)
        raise


def _require_job(store: JobStore, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(f"Job not found: {job_id}")
    return job


def _write_progress_report(
    store: JobStore,
    job_id: str,
    config: ServiceConfig,
    input_metadata: dict[str, Any],
    stage_timings: dict[str, float],
    artifacts: dict[str, Path],
    root_trajectory: dict[str, Any],
) -> None:
    job = _require_job(store, job_id)
    report_path = ServiceStorage(config.storage_root).paths(job_id).report_path
    write_report(
        report_path,
        build_report(
            job,
            config,
            input_metadata=input_metadata,
            stage_timings=stage_timings,
            artifacts=artifacts,
            root_trajectory=root_trajectory,
        ),
    )
    store.update_job(job_id, report_path=str(report_path))


def _run_pipeline(input_path: Path, work_dir: Path, job: JobRecord, config: ServiceConfig) -> Path:
    from src.pipeline import PipelineConfig
    from src.pipeline_factory import create_mocap_pipeline

    pipeline = create_mocap_pipeline(
        PipelineConfig(
            video_path=input_path,
            output_dir=work_dir,
            human_mesh=True,
            render_human_mesh=False,
            pose2d_format=str(job.request.get("pose2d_format", config.pose2d_format)),
            pose2d_config_path=config.resolve(config.pose2d_config),
            pose2d_checkpoint_path=config.resolve(config.pose2d_checkpoint),
            motionbert_repo=config.resolve(config.motionbert_repo),
            pose3d_config_path=config.resolve(config.pose3d_config),
            pose3d_checkpoint_path=config.resolve(config.pose3d_checkpoint),
            mesh_config_path=config.resolve(config.mesh_config),
            mesh_checkpoint_path=config.resolve(config.mesh_checkpoint),
            smpl_data_root=config.resolve(config.smpl_data_root),
            mesh_clip_len=int(job.request.get("mesh_clip_len", config.mesh_clip_len)),
            mesh_clip_stride=int(job.request.get("mesh_clip_stride", config.mesh_clip_stride)),
            device=config.device,
        )
    )
    result = pipeline.run()
    output_dir = result.metadata_path.parent if result.metadata_path is not None else work_dir / input_path.stem
    return output_dir



def _export_fbx(pipeline_output: Path, target: Path, job: JobRecord, config: ServiceConfig) -> tuple[Path, Path]:
    from src.retarget.fbx_exporter import BlenderExportConfig, export_animated_smpl_fbx

    export_animated_smpl_fbx(
        BlenderExportConfig(
            input_dir=pipeline_output,
            base_fbx=config.resolve(config.base_fbx),
            output=target,
            blender_executable=config.blender_executable,
            root_scale=config.root_scale,
            root_trajectory=str(job.request.get("root_trajectory", config.root_trajectory)),
        )
    )
    return target, target.parent / "root_trajectory.npy"


def _render_fbx_preview(fbx_path: Path, target: Path, config: ServiceConfig, input_metadata: dict[str, Any]) -> Path | None:
    from service.core.fbx_preview import DEFAULT_PREVIEW_SCRIPT, render_fbx_preview

    try:
        fps_value = input_metadata.get("fps")
        fps = float(fps_value) if fps_value is not None else None
        return render_fbx_preview(
            blender_executable=config.blender_executable,
            fbx_path=fbx_path,
            output_path=target,
            script_path=config.resolve(DEFAULT_PREVIEW_SCRIPT),
            fps=fps,
        )
    except Exception as exc:
        print(f"Warning: failed to render FBX preview: {exc}")
        return None


def _root_trajectory_summary(path: Path) -> dict[str, Any]:
    values = np.load(path)
    if values.ndim != 2 or values.shape[1] != 3:
        return {"path": str(path), "shape": list(values.shape)}
    span_m = np.ptp(values, axis=0) * 0.001
    return {
        "path": str(path),
        "shape": list(values.shape),
        "span_xyz_m": [float(value) for value in span_m],
    }
