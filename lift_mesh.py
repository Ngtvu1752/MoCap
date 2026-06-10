from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.pose3d.adapters import KeypointFormat, Pose2DFormatConverter
from src.pose3d.motionbert_mesh_estimator import MotionBERTMeshEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover SMPL mesh from RTMPose 2D keypoints with MotionBERT mesh")
    parser.add_argument("--input", type=Path, default=Path("output/pose2d.npy"), help="Input pose2d .npy")
    parser.add_argument("--input-format", choices=["rtmpose_raw", "whole_body133", "coco_body17", "halpe26", "human36m_17"], default="halpe26")
    parser.add_argument("--output-dir", type=Path, default=Path("output/mesh_recovery"))
    parser.add_argument("--motionbert-repo", type=Path, default=Path("MotionBERT"))
    parser.add_argument("--config", type=Path, default=Path("MotionBERT/configs/mesh/MB_ft_pw3d.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-len", type=int, default=243)
    parser.add_argument("--render", action="store_true", help="Also render a mesh video. This is slower.")
    parser.add_argument("--render-stride", type=int, default=1, help="Render every Nth frame for quick previews")
    parser.add_argument("--video", type=Path, default=None, help="Optional source video path. When set, render FPS is read from this video.")
    parser.add_argument("--fps", type=float, default=None, help="Render FPS. If omitted with --video, source video FPS is used; otherwise 30 FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose2d = np.load(args.input)
    if args.input_format != "human36m_17":
        source_format = KeypointFormat.RTMPOSE_RAW if args.input_format == "rtmpose_raw" else KeypointFormat(args.input_format)
        pose2d = Pose2DFormatConverter(source_format, KeypointFormat.HUMAN36M_17).convert(pose2d)

    estimator = MotionBERTMeshEstimator(
        repo_path=args.motionbert_repo,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        clip_len=args.clip_len,
    )
    result = estimator.recover(pose2d)

    vertices_path = args.output_dir / "vertices.npy"
    kp3d_path = args.output_dir / "mesh_kp3d.npy"
    theta_path = args.output_dir / "theta.npy"
    faces_path = args.output_dir / "faces.npy"
    np.save(vertices_path, result.vertices)
    np.save(kp3d_path, result.keypoints3d)
    np.save(theta_path, result.theta)
    np.save(faces_path, result.faces)

    print(f"Saved vertices {result.vertices.shape} to {vertices_path}")
    print(f"Saved mesh keypoints {result.keypoints3d.shape} to {kp3d_path}")
    print(f"Saved theta {result.theta.shape} to {theta_path}")
    print(f"Saved faces {result.faces.shape} to {faces_path}")

    if args.render:
        video_path = args.output_dir / "mesh_smpl.mp4"
        fps = resolve_render_fps(args.video, args.fps)
        render_mesh_video(result.vertices[::args.render_stride], result.faces, video_path, fps=fps / args.render_stride)
        print(f"Saved SMPL mesh video to {video_path} at {fps / args.render_stride:.4f} FPS")


def resolve_render_fps(video_path: Path | None, fps: float | None) -> float:
    if fps is not None:
        return fps
    if video_path is None:
        return 30.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video to read FPS: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if video_fps <= 0:
        raise ValueError(f"Cannot read a valid FPS from source video: {video_path}")
    return video_fps


def render_mesh_video(vertices: np.ndarray, faces: np.ndarray, output_path: Path, fps: float) -> None:
    width = height = 900
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")

    verts = vertices.astype(np.float32)
    center = verts.reshape(-1, 3).mean(axis=0)
    radius = np.max(np.ptp(verts.reshape(-1, 3), axis=0)) * 0.65
    if radius <= 0:
        radius = 1.0
    limits = [(center[i] - radius, center[i] + radius) for i in range(3)]
    face_subset = faces[::6]

    try:
        for frame in verts:
            fig = plt.figure(figsize=(9, 9), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_zlim(*limits[2])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=-90, azim=-90)
            ax.axis("off")
            mesh = Poly3DCollection(frame[face_subset], alpha=0.88, linewidths=0.05)
            mesh.set_facecolor((0.55, 0.66, 0.82, 0.88))
            mesh.set_edgecolor((0.25, 0.30, 0.38, 0.10))
            ax.add_collection3d(mesh)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(image)
            plt.close(fig)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
