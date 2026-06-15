# MoCap Environment Setup

Huong dan nay tao moi Conda environment `mocap` de chay pipeline:

```text
Video -> RTMPose -> pose2d.npy -> MotionBERT -> pose3d.npy -> renderer
```

Stack duoc khuyen dung:

```text
Python 3.9
NumPy 1.26.4
OpenCV 4.10.0.84
PyTorch 2.1.2 + CUDA 11.8
MMCV 2.1.0
MMEngine 0.10.4
MMDetection 3.2.0
MMPose 1.3.2
```

Luu y quan trong: khong dung `numpy==2.x` voi stack nay. `torch/mmcv` co the loi ABI:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
_ARRAY_API not found
```

## 1. Create Conda Environment

```bash
conda create -n mocap python=3.9 -y
conda activate mocap
cd /workspace/MoCap
```

## 2. Pin NumPy And OpenCV

Tao file constraint de `pip/mim` khong tu nang NumPy len 2.x:

```bash
printf "numpy==1.26.4\nopencv-python==4.10.0.84\n" > constraints-mocap.txt
```

Cai NumPy va OpenCV:

```bash
python -m pip install --upgrade pip
python -m pip install -c constraints-mocap.txt numpy==1.26.4 opencv-python==4.10.0.84
```

## 3. Install PyTorch CUDA 11.8

```bash
python -m pip install \
  torch==2.1.2+cu118 \
  torchvision==0.16.2+cu118 \
  torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install OpenMMLab Stack

```bash
python -m pip install openmim==0.3.9

PIP_CONSTRAINT=constraints-mocap.txt mim install mmengine==0.10.4
PIP_CONSTRAINT=constraints-mocap.txt mim install mmcv==2.1.0
PIP_CONSTRAINT=constraints-mocap.txt mim install mmdet==3.2.0
PIP_CONSTRAINT=constraints-mocap.txt mim install mmpose==1.3.2
```

Dung `mmdet==3.2.0`, khong dung `mmdet==3.3.0`, vi `mmpose==1.3.2` yeu cau `mmdet>=3.0.0,<3.3.0`.

Neu sau khi chay `mim install`, NumPy bi keo len `2.0.2`, ha lai:

```bash
python -m pip install --force-reinstall -c constraints-mocap.txt \
  numpy==1.26.4 \
  opencv-python==4.10.0.84
```

## 5. Install MotionBERT And Render Dependencies

```bash
python -m pip install -c constraints-mocap.txt \
  matplotlib==3.9.4 \
  easydict==1.13 \
  prettytable==3.16.0 \
  tensorboardX==2.6.5 \
  imageio==2.37.2 \
  imageio-ffmpeg==0.6.0 \
  roma==1.5.6 \
  timm==1.0.27 \
  einops==0.8.2 \
  ipdb==0.13.13 \
  scipy==1.13.1 \
  pandas==2.3.3 \
  tqdm==4.65.2 \
  pyyaml==6.0.3 \
  pycocotools==2.0.11 \
  xtcocotools==1.14.3 \
  json-tricks==3.17.3 \
  chumpy==0.70 \
  smplx==0.1.28 \
  trimesh==4.12.2 \
  pyrender==0.1.45
```

## 6. Verify Environment

```bash
python - <<'PY'
import numpy as np
import cv2
import torch
import mmcv
import mmengine
import mmdet
import mmpose

print("numpy", np.__version__)
print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmdet", mmdet.__version__)
print("mmpose", mmpose.__version__)
PY
```

Expected:

```text
numpy 1.26.4
cv2 4.10.0
torch 2.1.2+cu118
mmcv 2.1.0
mmengine 0.10.4
mmdet 3.2.0
mmpose 1.3.2
```

## 7. Check Required Artifacts

Repo can co cac file sau:

```text
input/dance.mp4

checkpoints/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
checkpoints/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth

checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py
checkpoints/body_2d_keypoint/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth

MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml
checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

## 8. Run Pipeline

Check video metadata:

```bash
python main.py input/dance.mp4 --metadata-only
```

Run RTMPose only:

```bash
python main.py input/dance.mp4 \
  --pose2d-only \
  --pose2d-format whole_body133 \
  --device cuda:0
```

Lift 2D pose to 3D:

```bash
python lift_pose3d.py \
  --input output/dance/pose2d.npy \
  --input-format whole_body133 \
  --output output/dance/pose3d.npy \
  --device cuda:0
```

Render 3D pose:

```bash
python render_pose3d.py \
  --input output/dance/pose3d.npy \
  --skeleton-output output/dance/skeleton3d.mp4 \
  --mesh-output output/dance/tube_mesh.mp4 \
  --fps 60 \
  --mode both
```

Run full pipeline:

```bash
python main.py input/dance.mp4 \
  --pose2d-format whole_body133 \
  --device cuda:0
```

## Notes

- `tube_mesh.mp4` la tube/stick mesh tu 17 joints, khong phai full SMPL body mesh.
- `PoseMambaEstimator` hien chi la placeholder. Pipeline thuc te dang dung MotionBERT.
- Neu may khong co GPU hoac CUDA khong san sang, co the thu `--device cpu`, nhung Phase 2/3 se cham hon nhieu.
