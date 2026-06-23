# MoCap Demo: Video to 3D Skeleton Pipeline

Demo pipeline chuyen video mot nguoi thanh chuoi 2D pose, 3D pose va video render skeleton and optional SMPL human mesh.

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
-> skeleton3d.mp4
```

## Output

Mac dinh moi video se co mot folder con trong `output/`, vi du `output/dance/`:

```text
output/dance/metadata.json       # video, pipeline, model config/checkpoint/device metadata
output/dance/pose2d.npy          # raw RTMPose output, shape (T,K,2)
output/dance/pose2d_scores.npy   # RTMPose keypoint confidence scores, shape (T,K)
output/dance/pose3d.npy          # MotionBERT output, shape (T,17,3)
output/dance/skeleton3d.mp4      # 3D skeleton render
output/dance/smpl_vertices.npy   # optional HMR output, shape (T,6890,3)
output/dance/smpl_joints3d.npy   # optional HMR regressed H36M joints, shape (T,17,3)
output/dance/smpl_theta.npy      # optional HMR SMPL params, shape (T,82)
output/dance/human_mesh.mp4      # optional HMR rendered SMPL mesh video
```

Luu y: full body mesh/SMPL render chi duoc tao khi bat `--human-mesh`, output la `human_mesh.mp4`.

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

## Checkpoints And Model Assets Expected

Dat cac checkpoint/model asset theo dung duong dan ben duoi de chay pipeline khong can truyen path tuy bien.

### RTMPose WholeBody 133

Dung cho `--pose2d-format whole_body133` mac dinh:

```text
checkpoints/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
checkpoints/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
```

### RTMPose Body Halpe26

Khuyen dung khi chay Human Mesh Recovery vi MotionBERT mesh branch duoc thiet ke quanh Halpe26 -> H36M17 conversion:

```text
checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py
checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth
```

### MotionBERT Pose3D

Dung cho buoc 2D-to-3D skeleton lifting, tao `pose3d.npy`:

```text
MotionBERT/
MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml
checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

### MotionBERT Human Mesh Recovery

Dung khi bat `--human-mesh`, tao `smpl_vertices.npy`, `smpl_theta.npy`, `human_mesh.mp4`:

```text
MotionBERT/configs/mesh/MB_ft_pw3d.yaml
checkpoints/MotionBERT/FT_MB_release_MB_ft_pw3d/best_epoch.bin
```

### SMPL Mesh Assets

MotionBERT HMR can day du cac file nay trong cung mot thu muc `--smpl-data-root`, mac dinh la `checkpoints/Mesh`:

```text
checkpoints/Mesh/SMPL_NEUTRAL.pkl
checkpoints/Mesh/J_regressor_h36m_correct.npy
checkpoints/Mesh/J_regressor_extra.npy
checkpoints/Mesh/smpl_mean_params.npz
```

`SMPL_NEUTRAL.pkl` la model SMPL neutral body. Ba file con lai la asset phu tro ma MotionBERT mesh head can de khoi tao mean pose/shape va regress H36M joints tu SMPL vertices.

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
x = np.load('output/dance/pose2d.npy')
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
  --input output/dance/pose2d.npy \
  --input-format whole_body133 \
  --output output/dance/pose3d.npy \
  --device cuda:0
```

For Halpe26 pose2d:

```bash
python lift_pose3d.py \
  --input output/dance/pose2d.npy \
  --input-format halpe26 \
  --output output/dance/pose3d.npy \
  --device cuda:0
```

Expected:

```text
output/dance/pose3d.npy shape: (T,17,3)
```

### 4. Run Phase 4 only: render pose3d.npy

```bash
python render_pose3d.py \
  --input output/dance/pose3d.npy \
  --skeleton-output output/dance/skeleton3d.mp4 \
  --fps 60
```

Render skeleton:

```bash
python render_pose3d.py --input output/dance/pose3d.npy --skeleton-output output/dance/skeleton3d.mp4
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

Run full pipeline with Human Mesh Recovery, Halpe26 recommended:

```bash
python main.py input/dance.mp4 \
  --pose2d-format halpe26 \
  --pose2d-config checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py \
  --pose2d-checkpoint checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth \
  --human-mesh \
  --mesh-config MotionBERT/configs/mesh/MB_ft_pw3d.yaml \
  --mesh-checkpoint checkpoints/MotionBERT/FT_MB_release_MB_ft_pw3d/best_epoch.bin \
  --smpl-data-root checkpoints/Mesh \
  --mesh-clip-len 243 \
  --mesh-clip-stride 121 \
  --device cuda:0
```

`--mesh-clip-stride` la so frame overlap giua hai mesh clip lien tiep. Vi du `--mesh-clip-len 243 --mesh-clip-stride 121` se cat clip cach nhau 122 frame va blend prediction o vung chong lan de giam giat o bien clip.

### Export animated SMPL FBX for Unity

Pipeline retarget dung rotation tu `smpl_theta.npy`, root trajectory tu `pose3d.npy`, va `smpl_joints3d.npy` de auto-scale/foot contact truoc khi Blender headless bake len SMPL T-pose rig:

```bash
python export_smpl_fbx.py \
  --input-dir output/dance2 \
  --base-fbx assets/retarget/smpl_base_tpose.fbx
```

Mac dinh `--root-trajectory pose3d`. Dung `--root-trajectory smpl` de debug hanh vi cu, hoac `zero` de xuat animation in-place. Co the override auto-scale bang `--pose3d-scale-mm FLOAT`.

Output:

```text
output/dance2/retarget/root_trajectory.npy
output/dance2/retarget/animated_smpl.fbx
```

Trong Unity, tat `Bake Into Pose` cho Root Transform Position (XZ) va bat `Apply Root Motion` tren Animator.

Neu Blender khong nam trong `PATH`, truyen executable tuyet doi:

```bash
python export_smpl_fbx.py \
  --input-dir output/dance2 \
  --blender /path/to/blender
```

BVH exporter van duoc giu lai de debug/inspect motion cu, nhung khong con la duong chinh cho Unity Mecanim.

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
src/renderer/mesh_renderer.py # skeleton and optional SMPL human mesh renderer
```
