from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.renderer.base import PoseRenderer, RenderResult


H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_BONES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}
MID_COLOR = "#1f77b4"
LEFT_COLOR = "#2ca02c"
RIGHT_COLOR = "#d62728"
MESH_COLOR = "#2f6f9f"


@dataclass(frozen=True)
class RenderStyle:
    width: int = 900
    height: int = 900
    dpi: int = 100
    skeleton_line_width: float = 3.0
    mesh_line_width: float = 10.0
    marker_size: float = 24.0
    elev: float = 12.0
    azim: float = 80.0
    margin: float = 0.15


class SkeletonRenderer(PoseRenderer):
    """Render Human3.6M 17-joint 3D motion as skeleton and tube-like mesh videos."""

    def __init__(self, style: RenderStyle | None = None) -> None:
        self.style = style or RenderStyle()

    def render(self, pose3d: np.ndarray, output_path: Path, fps: float) -> RenderResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pose3d = self._validate_pose3d(pose3d)
        skeleton_path = output_path.with_name("skeleton3d.mp4")
        mesh_path = output_path.with_name("tube_mesh.mp4")

        self.render_skeleton(pose3d, skeleton_path, fps=fps)
        self.render_mesh(pose3d, mesh_path, fps=fps)
        return RenderResult(skeleton_video_path=skeleton_path, mesh_video_path=mesh_path)

    def render_skeleton(self, pose3d: np.ndarray, output_path: Path, fps: float = 30.0) -> Path:
        return self._render_video(pose3d, output_path, fps=fps, mode="skeleton")

    def render_mesh(self, pose3d: np.ndarray, output_path: Path, fps: float = 30.0) -> Path:
        return self._render_video(pose3d, output_path, fps=fps, mode="mesh")

    def _render_video(self, pose3d: np.ndarray, output_path: Path, fps: float, mode: str) -> Path:
        pose3d = self._validate_pose3d(pose3d)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        limits = self._axis_limits(pose3d)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps) if fps > 0 else 30.0,
            (self.style.width, self.style.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for {output_path}")

        try:
            for frame in pose3d:
                image = self._draw_frame(frame, limits=limits, mode=mode)
                writer.write(image)
        finally:
            writer.release()

        return output_path

    def _draw_frame(self, joints: np.ndarray, limits: tuple[tuple[float, float], ...], mode: str) -> np.ndarray:
        fig = plt.figure(
            figsize=(self.style.width / self.style.dpi, self.style.height / self.style.dpi),
            dpi=self.style.dpi,
        )
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        self._setup_axes(ax, limits)
        if mode == "skeleton":
            self._draw_skeleton(ax, joints)
        elif mode == "mesh":
            self._draw_tube_mesh(ax, joints)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return rgb

    def _setup_axes(self, ax, limits: tuple[tuple[float, float], ...]) -> None:
        xlim, ylim, zlim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=self.style.elev, azim=self.style.azim)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True, alpha=0.28)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def _draw_skeleton(self, ax, joints: np.ndarray) -> None:
        for bone in H36M_BONES:
            a, b = bone
            color = self._bone_color(bone)
            xs, ys, zs = self._vis_points(joints[[a, b]])
            ax.plot(xs, ys, zs, color=color, lw=self.style.skeleton_line_width)
        xs, ys, zs = self._vis_points(joints)
        ax.scatter(xs, ys, zs, s=self.style.marker_size, c="#ffffff", edgecolors="#1f2933", linewidths=1.2)

    def _draw_tube_mesh(self, ax, joints: np.ndarray) -> None:
        segments = []
        colors = []
        for bone in H36M_BONES:
            a, b = bone
            p = np.column_stack(self._vis_points(joints[[a, b]]))
            segments.append(p)
            colors.append(MESH_COLOR)
        collection = Line3DCollection(
            segments,
            colors=colors,
            linewidths=self.style.mesh_line_width,
            alpha=0.86,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_collection3d(collection)
        xs, ys, zs = self._vis_points(joints)
        ax.scatter(xs, ys, zs, s=self.style.marker_size * 1.35, c="#f8fbff", edgecolors=MESH_COLOR, linewidths=1.3)

    def _bone_color(self, bone: tuple[int, int]) -> str:
        if bone in LEFT_BONES:
            return LEFT_COLOR
        if bone in RIGHT_BONES:
            return RIGHT_COLOR
        return MID_COLOR

    def _axis_limits(self, pose3d: np.ndarray) -> tuple[tuple[float, float], ...]:
        points = np.column_stack(self._vis_points(pose3d.reshape(-1, 3)))
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(maxs - mins) * (0.5 + self.style.margin))
        if radius <= 0:
            radius = 1.0
        return tuple((float(c - radius), float(c + radius)) for c in center)

    def _vis_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Match MotionBERT visualization convention: plot (-x, -z, -y).
        return -points[..., 0], -points[..., 2], -points[..., 1]

    def _validate_pose3d(self, pose3d: np.ndarray) -> np.ndarray:
        pose3d = np.asarray(pose3d, dtype=np.float32)
        if pose3d.ndim != 3:
            raise ValueError(f"Expected pose3d shape (T, 17, 3), got {pose3d.shape}")
        if pose3d.shape[1:] != (17, 3):
            raise ValueError(f"Expected pose3d shape (T, 17, 3), got {pose3d.shape}")
        if pose3d.shape[0] == 0:
            raise ValueError("Expected at least one pose3d frame")
        if not np.isfinite(pose3d).all():
            raise ValueError("pose3d contains NaN or infinite values")
        return pose3d
