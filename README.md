# MoCap Demo: Video to 3D Skeleton Pipeline

Demo pipeline chuyen video mot nguoi thanh chuoi 2D pose, 3D pose va video render skeleton/tube mesh.

Pipeline hien tai:

```text
input video (.mp4)
-> VideoReader
-> RTMPose 2D pose
-> pose2d.npy
-> keypoint adapter to MotionBERT/H36M 17 joints
-> MotionBERT 2D-to-3D lifting
-> pose3d.npy
-> 3D renderer
-> skeleton3d.mp4 + tube_mesh.mp4
```

## Output

Mac dinh output nam trong `output/`:

```text
output/pose2d.npy        # raw RTMPose output, shape (T,K,2)
output/pose3d.npy        # MotionBERT output, shape (T,17,3)
output/skeleton3d.mp4    # 3D skeleton render
output/tube_mesh.mp4     # tube/stick mesh render from 17 joints
```

Luu y: `tube_mesh.mp4` khong phai full body surface mesh/SMPL mesh. No la mesh dang ong duoc ve tu 17 joints va bones.

## Supported 2D Keypoint Formats

Pipeline co the nhan cac output RTMPose sau:

```text
whole_body133  # RTMPose COCO-WholeBody 133 keypoints
coco_body17    # RTMPose/COCO body 17 keypoints
halpe26        # RTMPose body 26 Halpe keypoints
```

Tat ca deu duoc convert sang MotionBERT/Human3.6M 17 joints:

```text
0 root, 1 RHip, 2 RKnee, 3 RAnkle,
4 LHip, 5 LKnee, 6 LAnkle, 7 torso,
8 neck, 9 nose, 10 head,
11 LShoulder, 12 LElbow, 13 LWrist,
14 RShoulder, 15 RElbow, 16 RWrist
```

## Checkpoints Expected

RTMPose WholeBody 133 default:

```text
checkpoints/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
checkpoints/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
```

RTMPose Body Halpe26:

```text
checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py
checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth
```

MotionBERT:

```text
MotionBERT/
checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

## Install Notes

Project dang dung env `mocap` co MMPose/MMCV/MMDetection cho RTMPose va MotionBERT deps toi thieu.

Cai dependency co ban:

```bash
pip install -r requirements.txt
```

Neu chay MotionBERT bi thieu lib, cai toi thieu:

```bash
pip install tensorboardX easydict prettytable imageio-ffmpeg roma ipdb timm einops
```

## CLI Usage

### 1. Check video metadata only

```bash
python main.py input/dance.mp4 --metadata-only
```

### 2. Run Phase 2 only: RTMPose -> pose2d.npy

WholeBody 133:

```bash
python main.py input/dance.mp4 \
  --pose2d-only \
  --pose2d-format whole_body133 \
  --device cuda:0
```

Halpe26:

```bash
python main.py input/dance.mp4 \
  --pose2d-only \
  --pose2d-format halpe26 \
  --pose2d-config checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py \
  --pose2d-checkpoint checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth \
  --device cuda:0
```

Check shape:

```bash
python - <<'CHECK_SHAPE'
import numpy as np
x = np.load('output/pose2d.npy')
print(x.shape, x.dtype)
CHECK_SHAPE
```

Expected examples:

```text
WholeBody: (T,133,2)
Halpe26:   (T,26,2)
```

### 3. Run Phase 3 only: pose2d.npy -> pose3d.npy

For WholeBody 133 pose2d:

```bash
python lift_pose3d.py \
  --input output/pose2d.npy \
  --input-format whole_body133 \
  --output output/pose3d.npy \
  --device cuda:0
```

For Halpe26 pose2d:

```bash
python lift_pose3d.py \
  --input output/pose2d.npy \
  --input-format halpe26 \
  --output output/pose3d.npy \
  --device cuda:0
```

Expected:

```text
output/pose3d.npy shape: (T,17,3)
```

### 4. Run Phase 4 only: render pose3d.npy

```bash
python render_pose3d.py \
  --input output/pose3d.npy \
  --skeleton-output output/skeleton3d.mp4 \
  --mesh-output output/tube_mesh.mp4 \
  --fps 60 \
  --mode both
```

Render only skeleton:

```bash
python render_pose3d.py --input output/pose3d.npy --mode skeleton
```

Render only tube mesh:

```bash
python render_pose3d.py --input output/pose3d.npy --mode mesh
```

### 5. Run full pipeline

WholeBody 133 default:

```bash
python main.py input/dance.mp4 \
  --pose2d-format whole_body133 \
  --device cuda:0
```

Halpe26:

```bash
python main.py input/dance.mp4 \
  --pose2d-format halpe26 \
  --pose2d-config checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py \
  --pose2d-checkpoint checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth \
  --device cuda:0
```

Run up to Phase 3 only, without rendering:

```bash
python main.py input/dance.mp4 --pose3d-only --device cuda:0
```

## Device

`--device cuda:0` dung GPU NVIDIA so 0 cho RTMPose va MotionBERT.

Dung CPU:

```bash
python main.py input/dance.mp4 --device cpu
```

Render video hien tai dung Matplotlib/OpenCV nen chay CPU, vi vay GPU chi hoat dong ro trong Phase 2 va Phase 3.

## Project Structure

```text
main.py                       # full pipeline CLI
lift_pose3d.py                # Phase 3 CLI
render_pose3d.py              # Phase 4 CLI
src/io/video_reader.py        # video metadata/frame reader
src/pose2d/rtmpose_estimator.py
src/pose3d/adapters.py        # keypoint format conversion
src/pose3d/motionbert_estimator.py
src/renderer/mesh_renderer.py # skeleton/tube mesh renderer
```
