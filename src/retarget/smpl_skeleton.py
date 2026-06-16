from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SMPL_SKELETON_PATH = Path("configs/skeletons/smpl_skeleton.json")


@dataclass(frozen=True)
class SMPLSkeleton:
    name: str
    joints: list[str]
    parents: list[int]
    offsets: np.ndarray
    rotation_format: str
    euler_order: str
    coordinate_system: str
    units: str
    metadata: dict[str, Any]

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def root_index(self) -> int:
        return self.parents.index(-1)

    @property
    def root_joint(self) -> str:
        return self.joints[self.root_index]

    def children(self, joint_index: int) -> list[int]:
        return [index for index, parent in enumerate(self.parents) if parent == joint_index]


def load_smpl_skeleton(path: Path | str = DEFAULT_SMPL_SKELETON_PATH) -> SMPLSkeleton:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SMPL skeleton file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = _string_list(payload, "joints")
    parents = _int_list(payload, "parents")
    offsets = np.asarray(payload.get("offsets"), dtype=np.float32)

    skeleton = SMPLSkeleton(
        name=str(payload.get("name", "SMPL_24")),
        joints=joints,
        parents=parents,
        offsets=offsets,
        rotation_format=str(payload.get("rotation_format", "axis_angle")),
        euler_order=str(payload.get("euler_order", "xyz")),
        coordinate_system=str(payload.get("coordinate_system", "motionbert_smpl")),
        units=str(payload.get("units", "millimeters")),
        metadata=payload,
    )
    validate_smpl_skeleton(skeleton)
    return skeleton


def validate_smpl_skeleton(skeleton: SMPLSkeleton) -> None:
    if skeleton.joint_count != 24:
        raise ValueError(f"SMPL skeleton must have 24 joints, got {skeleton.joint_count}.")
    if len(set(skeleton.joints)) != skeleton.joint_count:
        raise ValueError("SMPL skeleton joint names must be unique.")
    if len(skeleton.parents) != skeleton.joint_count:
        raise ValueError(f"Expected 24 parent indices, got {len(skeleton.parents)}.")
    if skeleton.offsets.shape != (skeleton.joint_count, 3):
        raise ValueError(f"Expected offsets shape (24,3), got {skeleton.offsets.shape}.")
    if skeleton.parents.count(-1) != 1:
        raise ValueError("SMPL skeleton must contain exactly one root parent index (-1).")
    if skeleton.root_index != 0:
        raise ValueError(f"Expected pelvis root at index 0, got root index {skeleton.root_index}.")
    for index, parent in enumerate(skeleton.parents):
        if parent == -1:
            continue
        if parent < 0 or parent >= skeleton.joint_count:
            raise ValueError(f"Joint {index} has invalid parent index {parent}.")
        if parent >= index:
            raise ValueError(f"Joint {index} parent index {parent} must come before the child.")
    if not np.isfinite(skeleton.offsets).all():
        raise ValueError("SMPL skeleton offsets contain NaN or Inf values.")


def describe_skeleton(skeleton: SMPLSkeleton) -> str:
    lines = [
        f"Name: {skeleton.name}",
        f"Joints: {skeleton.joint_count}",
        f"Root: {skeleton.root_joint} ({skeleton.root_index})",
        f"Rotation format: {skeleton.rotation_format}",
        f"Euler order: {skeleton.euler_order}",
        f"Coordinate system: {skeleton.coordinate_system}",
        f"Units: {skeleton.units}",
    ]
    for index, joint in enumerate(skeleton.joints):
        parent = skeleton.parents[index]
        parent_name = "none" if parent == -1 else skeleton.joints[parent]
        offset = skeleton.offsets[index]
        lines.append(
            f"{index:02d} {joint} parent={parent_name} "
            f"offset=({offset[0]:.3g}, {offset[1]:.3g}, {offset[2]:.3g})"
        )
    return "\n".join(lines)


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected {key} to be a list.")
    return [str(item) for item in value]


def _int_list(payload: dict[str, Any], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected {key} to be a list.")
    return [int(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect the canonical SMPL skeleton config.")
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=DEFAULT_SMPL_SKELETON_PATH,
        help="Path to smpl_skeleton.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skeleton = load_smpl_skeleton(args.skeleton)
    print(describe_skeleton(skeleton))


if __name__ == "__main__":
    main()
