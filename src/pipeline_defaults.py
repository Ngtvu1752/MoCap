from __future__ import annotations

from pathlib import Path


SUPPORTED_POSE2D_FORMATS = ("whole_body133", "coco_body17", "halpe26")

DEFAULT_POSE2D_FORMAT = "halpe26"
DEFAULT_POSE2D_CONFIG = Path("checkpoints/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py")
DEFAULT_POSE2D_CHECKPOINT = Path(
    "checkpoints/body_2d_keypoint/"
    "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth"
)

DEFAULT_WHOLEBODY_POSE2D_CONFIG = Path("checkpoints/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py")
DEFAULT_WHOLEBODY_POSE2D_CHECKPOINT = Path(
    "checkpoints/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
)

DEFAULT_MOTIONBERT_REPO = Path("MotionBERT")
DEFAULT_POSE3D_CONFIG = Path("MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml")
DEFAULT_POSE3D_CHECKPOINT = Path("checkpoints/MotionBERT/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")

DEFAULT_MESH_CONFIG = Path("MotionBERT/configs/mesh/MB_ft_pw3d.yaml")
DEFAULT_MESH_CHECKPOINT = Path("checkpoints/MotionBERT/FT_MB_release_MB_ft_pw3d/best_epoch.bin")
DEFAULT_SMPL_DATA_ROOT = Path("checkpoints/Mesh")

DEFAULT_DEVICE = "cuda:0"
DEFAULT_MESH_CLIP_LEN = 243
DEFAULT_MESH_CLIP_STRIDE = 121
