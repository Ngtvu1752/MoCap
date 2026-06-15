from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class SMPLMeshRenderStyle:
    width: int = 900
    height: int = 900
    dpi: int = 100
    elev: float = 12.0
    azim: float = 80.0
    margin: float = 0.18
    face_stride: int = 4
    mesh_color: str = "#7aa6c2"
    edge_color: str = "#33566b"
    alpha: float = 0.92


class SMPLMeshRenderer:
    """Render SMPL vertices and faces to a simple rotating-free 3D video."""

    def __init__(self, style: SMPLMeshRenderStyle | None = None) -> None:
        self.style = style or SMPLMeshRenderStyle()

    def render(self, vertices: np.ndarray, faces: np.ndarray, output_path: Path, fps: float = 30.0) -> Path:
        vertices = self._validate_vertices(vertices)
        faces = self._validate_faces(faces)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        limits = self._axis_limits(vertices)
        faces_to_draw = faces[:: max(1, self.style.face_stride)]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps) if fps > 0 else 30.0,
            (self.style.width, self.style.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for {output_path}")

        try:
            for frame_vertices in vertices:
                writer.write(self._draw_frame(frame_vertices, faces_to_draw, limits))
        finally:
            writer.release()
        return output_path

    def _draw_frame(self, vertices: np.ndarray, faces: np.ndarray, limits: tuple[tuple[float, float], ...]) -> np.ndarray:
        fig = plt.figure(
            figsize=(self.style.width / self.style.dpi, self.style.height / self.style.dpi),
            dpi=self.style.dpi,
        )
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        self._setup_axes(ax, limits)

        points = np.column_stack(self._vis_points(vertices))
        mesh = Poly3DCollection(
            points[faces],
            facecolors=self.style.mesh_color,
            edgecolors=self.style.edge_color,
            linewidths=0.05,
            alpha=self.style.alpha,
        )
        ax.add_collection3d(mesh)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return bgr

    def _setup_axes(self, ax, limits: tuple[tuple[float, float], ...]) -> None:
        xlim, ylim, zlim = limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=self.style.elev, azim=self.style.azim)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def _axis_limits(self, vertices: np.ndarray) -> tuple[tuple[float, float], ...]:
        points = np.column_stack(self._vis_points(vertices.reshape(-1, 3)))
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(maxs - mins) * (0.5 + self.style.margin))
        if radius <= 0:
            radius = 1.0
        return tuple((float(c - radius), float(c + radius)) for c in center)

    def _vis_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return -points[..., 0], -points[..., 2], -points[..., 1]

    def _validate_vertices(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}")
        if vertices.shape[0] == 0:
            raise ValueError("Expected at least one mesh frame")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain NaN or infinite values")
        return vertices

    def _validate_faces(self, faces: np.ndarray) -> np.ndarray:
        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError(f"Expected faces shape (F, 3), got {faces.shape}")
        return faces
