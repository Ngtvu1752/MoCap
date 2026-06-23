from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


BONE_PREFIX = "m_avg_"
ROOT_BONE_NAME = "m_avg_root"
SMPL_BONE_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
)

# The MotionBERT renderer maps source points as (-x, -z, -y). The imported
# SMPL armature already maps its local coordinates to Blender world as
# (x, -z, y), so source motion needs this local 180-degree Z correction.
SOURCE_TO_RIG_LOCAL = np.diag((-1.0, -1.0, 1.0)).astype(np.float32)


def _script_args() -> list[str]:
    try:
        return sys.argv[sys.argv.index("--") + 1 :]
    except ValueError:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake MotionBERT SMPL motion onto the base SMPL FBX armature.")
    parser.add_argument("--base-fbx", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--root-translation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--root-scale", type=float, default=0.1)
    return parser.parse_args(_script_args())


def _load_motion(theta_path: Path, root_translation_path: Path) -> tuple[np.ndarray, np.ndarray]:
    theta = np.load(theta_path).astype(np.float32)
    root_translation = np.load(root_translation_path).astype(np.float32)
    if theta.ndim != 2 or theta.shape[1] != 82:
        raise ValueError(f"Expected theta shape (T,82), got {theta.shape}.")
    if root_translation.ndim != 2 or root_translation.shape[1] != 3:
        raise ValueError(f"Expected root translation shape (T,3), got {root_translation.shape}.")
    if theta.shape[0] != root_translation.shape[0] or theta.shape[0] == 0:
        raise ValueError(
            f"Invalid frame counts: theta={theta.shape[0]}, root_translation={root_translation.shape[0]}."
        )
    if not np.isfinite(theta).all() or not np.isfinite(root_translation).all():
        raise ValueError("Motion arrays contain NaN or Inf values.")
    return theta[:, :72].reshape(-1, 24, 3), root_translation


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _patch_fbx_light_importer() -> None:
    # Blender 5.1.2's bundled FBX importer writes a removed Cycles light
    # property. Keep the already-created light and continue importing the rig.
    from io_scene_fbx import import_fbx

    original = import_fbx.blen_read_light
    if getattr(original, "_mocap_safe_light_import", False):
        return

    def safe_blen_read_light(fbx_tmpl, fbx_obj, settings):
        existing = set(bpy.data.lights)
        try:
            return original(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" not in str(exc):
                raise
            created = [light for light in bpy.data.lights if light not in existing]
            if len(created) != 1:
                raise RuntimeError("Could not recover the FBX light created before the Blender importer error.") from exc
            print("Warning: skipped unsupported CyclesLightSettings.cast_shadow while importing FBX.")
            return created[0]

    safe_blen_read_light._mocap_safe_light_import = True
    import_fbx.blen_read_light = safe_blen_read_light


def _import_armature(base_fbx: Path):
    _patch_fbx_light_importer()
    bpy.ops.import_scene.fbx(filepath=str(base_fbx.resolve()), use_anim=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected exactly one armature in {base_fbx}, found {len(armatures)}.")
    armature = armatures[0]
    required = [ROOT_BONE_NAME, *(BONE_PREFIX + name for name in SMPL_BONE_NAMES)]
    missing = [name for name in required if armature.pose.bones.get(name) is None]
    if missing:
        raise RuntimeError("Base FBX is missing required bones: " + ", ".join(missing))
    return armature


def _axis_angle_quaternion(axis_angle: np.ndarray) -> Quaternion:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    return Quaternion(Vector((axis_angle / angle).tolist()), angle)


def _continuous_quaternion(quaternion: Quaternion, previous: Quaternion | None) -> Quaternion:
    quaternion.normalize()
    if previous is not None and quaternion.dot(previous) < 0.0:
        quaternion.negate()
    return quaternion


def _source_root_to_rig_local(quaternion: Quaternion) -> Quaternion:
    correction = Quaternion(Vector((0.0, 0.0, 1.0)), math.pi)
    return correction @ quaternion


def _source_translation_to_rig_local(translation: np.ndarray) -> Vector:
    converted = SOURCE_TO_RIG_LOCAL @ np.asarray(translation, dtype=np.float32)
    return Vector(converted.tolist())


def _set_scene_fps(fps: float) -> None:
    if fps <= 0:
        raise ValueError(f"Expected positive FPS, got {fps}.")
    # FBX stores standard integer frame rates. Using fps_base for a fractional
    # source rate makes Blender resample and can silently drop end frames.
    nominal = max(1, int(round(fps)))
    bpy.context.scene.render.fps = nominal
    bpy.context.scene.render.fps_base = 1.0


def _bake_motion(
    armature,
    pose_axis_angle: np.ndarray,
    root_translation: np.ndarray,
    fps: float,
    root_scale: float,
) -> None:
    frame_count = pose_axis_angle.shape[0]
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count
    _set_scene_fps(fps)

    armature.animation_data_clear()
    armature.animation_data_create()
    action = bpy.data.actions.new(name="SMPL_Animation")
    armature.animation_data.action = action
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    root_bone = armature.pose.bones[ROOT_BONE_NAME]
    previous_quaternions: list[Quaternion | None] = [None] * len(SMPL_BONE_NAMES)

    for frame_index in range(frame_count):
        frame = frame_index + 1
        scene.frame_set(frame)

        root_bone.location = _source_translation_to_rig_local(root_translation[frame_index]) * root_scale
        root_bone.keyframe_insert(data_path="location", frame=frame, group=ROOT_BONE_NAME)

        for joint_index, bone_name in enumerate(SMPL_BONE_NAMES):
            bone = armature.pose.bones[BONE_PREFIX + bone_name]
            bone.rotation_mode = "QUATERNION"
            quaternion = _axis_angle_quaternion(pose_axis_angle[frame_index, joint_index])
            if joint_index == 0:
                quaternion = _source_root_to_rig_local(quaternion)
            quaternion = _continuous_quaternion(quaternion, previous_quaternions[joint_index])
            previous_quaternions[joint_index] = quaternion.copy()
            bone.rotation_quaternion = quaternion
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)

    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def _set_export_rest_pose(armature) -> None:
    # FBX Model transforms are taken from the current evaluated pose. Insert
    # identity keys at frame 0, outside the exported frame range 1..T, so the
    # skeleton keeps its T-pose rest while animation sampling remains unchanged.
    scene = bpy.context.scene
    scene.frame_set(0)
    root_bone = armature.pose.bones[ROOT_BONE_NAME]
    root_bone.location = Vector((0.0, 0.0, 0.0))
    root_bone.keyframe_insert(data_path="location", frame=0, group=ROOT_BONE_NAME)
    for bone_name in SMPL_BONE_NAMES:
        bone = armature.pose.bones[BONE_PREFIX + bone_name]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        bone.keyframe_insert(data_path="rotation_quaternion", frame=0, group=bone.name)
    scene.frame_set(0)
    bpy.context.view_layer.update()


def _export_fbx(armature, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_export_rest_pose(armature)
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=True,
        object_types={"ARMATURE"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FBX export failed: {output_path}")


def main() -> None:
    args = parse_args()
    pose_axis_angle, root_translation = _load_motion(args.theta, args.root_translation)
    _clear_scene()
    armature = _import_armature(args.base_fbx)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    _bake_motion(armature, pose_axis_angle, root_translation, args.fps, args.root_scale)
    _export_fbx(armature, args.output)
    print(f"Saved animated SMPL FBX: {args.output.resolve()}")


if __name__ == "__main__":
    main()
