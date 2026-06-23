from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.signal import butter, sosfiltfilt
from scipy.sparse.linalg import lsqr


H36M_BONES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)
LEFT_ANKLE = 6
RIGHT_ANKLE = 3
PELVIS = 0

# Source MotionBERT coordinates to the Z-up world convention used by the
# renderer and FBX validation: (x, y, z) -> (-x, -z, -y).
SOURCE_TO_WORLD_ZUP = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class RootMotionConfig:
    cutoff_hz: float = 6.0
    prior_weight: float = 1.0
    smoothness_weight: float = 10.0
    contact_weight: float = 50.0
    floor_height_fraction: float = 0.05
    contact_speed_fraction: float = 0.15
    min_contact_frames: int = 2


@dataclass(frozen=True)
class RootMotionResult:
    translation_source_mm: np.ndarray
    scale_mm_per_pose_unit: float
    raw_span_m: np.ndarray
    cleaned_span_m: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray


def build_pose3d_root_motion(
    pose3d: np.ndarray,
    smpl_joints3d: np.ndarray,
    fps: float,
    *,
    scale_mm_per_pose_unit: float | None = None,
    config: RootMotionConfig | None = None,
) -> RootMotionResult:
    pose3d = _validate_joints("pose3d", pose3d)
    smpl_joints3d = _validate_joints("smpl_joints3d", smpl_joints3d)
    if pose3d.shape[0] != smpl_joints3d.shape[0]:
        raise ValueError(
            f"Frame count mismatch: pose3d={pose3d.shape[0]}, "
            f"smpl_joints3d={smpl_joints3d.shape[0]}."
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Expected positive finite FPS, got {fps}.")

    settings = config or RootMotionConfig()
    _validate_config(settings)
    scale = (
        estimate_pose3d_scale_mm(pose3d, smpl_joints3d)
        if scale_mm_per_pose_unit is None
        else _validate_scale(scale_mm_per_pose_unit)
    )

    raw_source_mm = (pose3d[:, PELVIS] - pose3d[0, PELVIS]) * scale
    raw_world_m = source_to_world_zup(raw_source_mm) * 0.001
    filtered_world_m = _low_pass_translation(raw_world_m, fps, settings.cutoff_hz)

    local_source_mm = smpl_joints3d - smpl_joints3d[:, PELVIS : PELVIS + 1]
    local_world_m = source_to_world_zup(local_source_mm) * 0.001
    left_foot = filtered_world_m + local_world_m[:, LEFT_ANKLE]
    right_foot = filtered_world_m + local_world_m[:, RIGHT_ANKLE]
    body_height = _body_height_m(local_world_m)
    floor_height = float(np.percentile(np.minimum(left_foot[:, 2], right_foot[:, 2]), 5.0))
    left_contact = detect_foot_contact(left_foot, fps, floor_height, body_height, settings)
    right_contact = detect_foot_contact(right_foot, fps, floor_height, body_height, settings)

    cleaned_world_m = optimize_root_trajectory(
        filtered_world_m,
        local_world_m[:, LEFT_ANKLE],
        local_world_m[:, RIGHT_ANKLE],
        left_contact,
        right_contact,
        floor_height,
        settings,
    )
    cleaned_world_m -= cleaned_world_m[0]
    translation_source_mm = world_zup_to_source(cleaned_world_m) * 1000.0

    return RootMotionResult(
        translation_source_mm=translation_source_mm.astype(np.float32),
        scale_mm_per_pose_unit=float(scale),
        raw_span_m=np.ptp(raw_world_m, axis=0).astype(np.float32),
        cleaned_span_m=np.ptp(cleaned_world_m, axis=0).astype(np.float32),
        left_contact=left_contact,
        right_contact=right_contact,
    )


def estimate_pose3d_scale_mm(pose3d: np.ndarray, smpl_joints3d: np.ndarray) -> float:
    pose3d = _validate_joints("pose3d", pose3d)
    smpl_joints3d = _validate_joints("smpl_joints3d", smpl_joints3d)
    if pose3d.shape[0] != smpl_joints3d.shape[0]:
        raise ValueError(
            f"Frame count mismatch: pose3d={pose3d.shape[0]}, "
            f"smpl_joints3d={smpl_joints3d.shape[0]}."
        )

    pose_lengths = _bone_lengths(pose3d).sum(axis=1)
    smpl_lengths = _bone_lengths(smpl_joints3d).sum(axis=1)
    valid = (pose_lengths > 1e-8) & (smpl_lengths > 1e-8)
    if not np.any(valid):
        raise ValueError("Cannot estimate Pose3D scale from zero-length skeletons.")
    return _validate_scale(float(np.median(smpl_lengths[valid] / pose_lengths[valid])))


def source_to_world_zup(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in 3 coordinates, got {points.shape}.")
    return points @ SOURCE_TO_WORLD_ZUP.T


def world_zup_to_source(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in 3 coordinates, got {points.shape}.")
    return points @ SOURCE_TO_WORLD_ZUP


def detect_foot_contact(
    foot_world_m: np.ndarray,
    fps: float,
    floor_height: float,
    body_height: float,
    config: RootMotionConfig,
) -> np.ndarray:
    foot_world_m = np.asarray(foot_world_m, dtype=np.float64)
    if foot_world_m.ndim != 2 or foot_world_m.shape[1] != 3:
        raise ValueError(f"Expected foot trajectory shape (T,3), got {foot_world_m.shape}.")
    if foot_world_m.shape[0] == 0:
        raise ValueError("Expected at least one foot trajectory frame.")

    velocity = np.zeros(foot_world_m.shape[0], dtype=np.float64)
    if foot_world_m.shape[0] > 1:
        velocity[1:] = np.linalg.norm(np.diff(foot_world_m, axis=0), axis=1) * fps
        velocity[0] = velocity[1]
    height_limit = floor_height + config.floor_height_fraction * body_height
    speed_limit = config.contact_speed_fraction * body_height
    contact = (foot_world_m[:, 2] <= height_limit) & (velocity <= speed_limit)
    return _remove_short_runs(contact, config.min_contact_frames)


def optimize_root_trajectory(
    raw_root_world_m: np.ndarray,
    left_foot_local_m: np.ndarray,
    right_foot_local_m: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    floor_height: float,
    config: RootMotionConfig,
) -> np.ndarray:
    raw_root = np.asarray(raw_root_world_m, dtype=np.float64)
    left_local = np.asarray(left_foot_local_m, dtype=np.float64)
    right_local = np.asarray(right_foot_local_m, dtype=np.float64)
    frame_count = raw_root.shape[0]
    expected = (frame_count, 3)
    if raw_root.shape != expected or left_local.shape != expected or right_local.shape != expected:
        raise ValueError("Root and local foot trajectories must all have shape (T,3).")
    left_contact = np.asarray(left_contact, dtype=bool)
    right_contact = np.asarray(right_contact, dtype=bool)
    if left_contact.shape != (frame_count,) or right_contact.shape != (frame_count,):
        raise ValueError("Foot contact masks must have shape (T,).")

    prior = sparse.eye(frame_count, format="csr") * np.sqrt(config.prior_weight)
    blocks = [prior]
    if frame_count >= 3 and config.smoothness_weight > 0:
        second_difference = sparse.diags(
            (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
            (0, 1, 2),
            shape=(frame_count - 2, frame_count),
            format="csr",
        )
        blocks.append(second_difference * np.sqrt(config.smoothness_weight))

    contact_rows: list[tuple[int, np.ndarray]] = []
    for local, contact in ((left_local, left_contact), (right_local, right_contact)):
        for start, end in _true_runs(contact):
            anchor = raw_root[start] + local[start]
            contact_rows.extend((frame, anchor - local[frame]) for frame in range(start, end))

    contact_matrix = None
    if contact_rows:
        rows = np.arange(len(contact_rows))
        cols = np.array([frame for frame, _ in contact_rows], dtype=np.int64)
        values = np.full(len(contact_rows), np.sqrt(config.contact_weight))
        contact_matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(contact_rows), frame_count),
        )
        blocks.append(contact_matrix)

    matrix = sparse.vstack(blocks, format="csr")
    result = np.empty_like(raw_root)
    for axis in range(3):
        targets = [raw_root[:, axis] * np.sqrt(config.prior_weight)]
        if frame_count >= 3 and config.smoothness_weight > 0:
            targets.append(np.zeros(frame_count - 2, dtype=np.float64))
        if contact_matrix is not None:
            contact_targets = np.array(
                [target[axis] for _, target in contact_rows],
                dtype=np.float64,
            ) * np.sqrt(config.contact_weight)
            targets.append(contact_targets)

        axis_matrix = matrix
        if axis == 2:
            floor_rows: list[tuple[int, float]] = []
            for local, contact in ((left_local, left_contact), (right_local, right_contact)):
                floor_rows.extend((frame, floor_height - local[frame, 2]) for frame in np.flatnonzero(contact))
            if floor_rows:
                rows = np.arange(len(floor_rows))
                cols = np.array([frame for frame, _ in floor_rows])
                values = np.full(len(floor_rows), np.sqrt(config.contact_weight))
                floor_matrix = sparse.csr_matrix(
                    (values, (rows, cols)),
                    shape=(len(floor_rows), frame_count),
                )
                axis_matrix = sparse.vstack((matrix, floor_matrix), format="csr")
                targets.append(
                    np.array([target for _, target in floor_rows]) * np.sqrt(config.contact_weight)
                )

        result[:, axis] = lsqr(axis_matrix, np.concatenate(targets), atol=1e-10, btol=1e-10)[0]
    return result


def _low_pass_translation(translation: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape[0] < 16 or cutoff_hz <= 0:
        return translation.copy()
    effective_cutoff = min(cutoff_hz, fps * 0.45)
    if effective_cutoff <= 0:
        return translation.copy()
    sos = butter(4, effective_cutoff, btype="lowpass", fs=fps, output="sos")
    filtered = sosfiltfilt(sos, translation, axis=0)
    filtered -= filtered[0]
    return filtered


def _bone_lengths(joints: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.linalg.norm(joints[:, child] - joints[:, parent], axis=1) for parent, child in H36M_BONES],
        axis=1,
    )


def _body_height_m(local_world_m: np.ndarray) -> float:
    frame_heights = np.ptp(local_world_m[..., 2], axis=1)
    height = float(np.median(frame_heights))
    if not np.isfinite(height) or height <= 1e-6:
        raise ValueError("Cannot estimate positive body height for foot contact detection.")
    return height


def _true_runs(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    start = 0
    while start < mask.size:
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < mask.size and mask[end]:
            end += 1
        yield start, end
        start = end


def _remove_short_runs(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    start = 0
    while start < result.size:
        if not result[start]:
            start += 1
            continue
        end = start + 1
        while end < result.size and result[end]:
            end += 1
        if end - start < minimum_length:
            result[start:end] = False
        start = end
    return result


def _validate_joints(name: str, joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (17, 3):
        raise ValueError(f"Expected {name} shape (T,17,3), got {joints.shape}.")
    if joints.shape[0] == 0:
        raise ValueError(f"Expected at least one frame in {name}.")
    if not np.isfinite(joints).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return joints


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Expected positive finite Pose3D scale in mm, got {scale}.")
    return scale


def _validate_config(config: RootMotionConfig) -> None:
    weights = (config.prior_weight, config.smoothness_weight, config.contact_weight)
    if config.prior_weight <= 0 or config.smoothness_weight < 0 or config.contact_weight < 0:
        raise ValueError(f"Invalid root-motion optimization weights: {weights}.")
    if config.floor_height_fraction < 0 or config.contact_speed_fraction < 0:
        raise ValueError("Foot-contact thresholds must be non-negative.")
    if config.min_contact_frames <= 0:
        raise ValueError("min_contact_frames must be positive.")
