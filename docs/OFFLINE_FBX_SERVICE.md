# Offline AI MoCap FBX Generation Service

MVP service nội bộ cho luồng:

```text
upload video -> queued job -> RTMPose -> MotionBERT -> SMPL -> Blender FBX -> download animated_smpl.fbx
```

## Install

Chuẩn bị môi trường MoCap theo `INSTALL.md`, sau đó cài service deps:

```bash
python -m pip install -r requirements-service.txt
cp .env.example .env
```

Sửa `.env` để trỏ đúng `MOCAP_SERVICE_API_KEY`, `MOCAP_BLENDER`, checkpoints, `MotionBERT`, `checkpoints/Mesh` và storage.

Kiểm tra môi trường:

```bash
python -m service.doctor
```

## Run Locally

Terminal 1:

```bash
redis-server
```

Terminal 2:

```bash
python -m service.api
```

Neu port dang ban, API se tu chon port trong tiep theo va in URL that. Co the dat co dinh bang `MOCAP_API_PORT=8001` trong `.env`.

Terminal 3:

```bash
python -m service.worker
```

## API

Upload job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "X-API-Key: $MOCAP_SERVICE_API_KEY" \
  -F "video=@input/dance2.mp4" \
  -F "profile=unity_humanoid_fbx" \
  -F "pose2d_format=halpe26" \
  -F "mesh_clip_len=243" \
  -F "mesh_clip_stride=121" \
  -F "root_trajectory=pose3d"
```

Check status:

```bash
curl -H "X-API-Key: $MOCAP_SERVICE_API_KEY" \
  http://localhost:8000/v1/jobs/<job_id>
```

Download source video:

```bash
curl -L -H "X-API-Key: $MOCAP_SERVICE_API_KEY" \
  http://localhost:8000/v1/jobs/<job_id>/download/source \
  -o source.mp4
```

Download SMPL FBX:

```bash
curl -L -H "X-API-Key: $MOCAP_SERVICE_API_KEY" \
  http://localhost:8000/v1/jobs/<job_id>/download/fbx \
  -o animated_smpl.fbx
```

Download report:

```bash
curl -L -H "X-API-Key: $MOCAP_SERVICE_API_KEY" \
  http://localhost:8000/v1/jobs/<job_id>/report \
  -o report.json
```

## Defaults And Limits

- Một GPU worker xử lý tuần tự.
- Video tối đa 5 phút, 1920x1080.
- Input/output được giữ 7 ngày.
- Output chính: `animated_smpl.fbx`, `report.json`, optional `fbx_preview.mp4`.
- Web UI hiển thị `source video` và preview render từ `animated_smpl.fbx` nếu render preview thành công.
- Service dùng `pose3d` làm root trajectory mặc định.
- Service không retarget trực tiếp sang avatar Mixamo/Ch36; target-avatar retargeting nên được xử lý ở Unity/Blender hoặc một phase riêng theo rig thật.

Cleanup:

```bash
python -m service.cleanup
```

## Docker Compose

```bash
cp .env.example .env
docker compose -f deployment/docker-compose.service.yml up --build
```

Không bake `SMPL_NEUTRAL.pkl` vào image. Mount `checkpoints/`, `MotionBERT/`, `assets/`, `service_data/` từ host.

