from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _script_args() -> list[str]:
    try:
        return sys.argv[sys.argv.index("--") + 1 :]
    except ValueError:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a lightweight MP4 preview from an animated FBX armature.")
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(_script_args())


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _patch_fbx_light_importer() -> None:
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
                raise RuntimeError("Could not recover FBX light after importer error.") from exc
            print("Warning: skipped unsupported CyclesLightSettings.cast_shadow while importing FBX.")
            return created[0]

    safe_blen_read_light._mocap_safe_light_import = True
    import_fbx.blen_read_light = safe_blen_read_light


def _import_fbx(fbx_path: Path):
    _patch_fbx_light_importer()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path.resolve()), use_anim=True)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"No armature found in {fbx_path}")
    armature = armatures[0]
    armature.show_in_front = True
    armature.data.display_type = "STICK"
    return armature


def _scene_frame_range(armature) -> tuple[int, int]:
    scene = bpy.context.scene
    start = int(scene.frame_start or 1)
    end = int(scene.frame_end or start)
    action = armature.animation_data.action if armature.animation_data else None
    if action is not None:
        action_start, action_end = action.frame_range
        start = max(1, int(math.floor(action_start)))
        end = max(start, int(math.ceil(action_end)))
    return start, end


def _bone_segments(armature) -> list[tuple[str, str]]:
    names = {bone.name for bone in armature.pose.bones}
    segments: list[tuple[str, str]] = []
    for bone in armature.pose.bones:
        if bone.parent is not None and bone.parent.name in names:
            segments.append((bone.parent.name, bone.name))
    if not segments:
        segments = [(bone.name, bone.name) for bone in armature.pose.bones]
    return segments


def _joint_position(armature, bone_name: str) -> Vector:
    bone = armature.pose.bones[bone_name]
    return armature.matrix_world @ bone.head


def _make_material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _create_sphere(name: str, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.045)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _create_cylinder(name: str, material):
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.025, depth=1.0)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _orient_cylinder(obj, start: Vector, end: Vector) -> None:
    vector = end - start
    length = max(vector.length, 1e-6)
    obj.location = start + vector * 0.5
    obj.scale = (1.0, 1.0, length)
    obj.rotation_euler = vector.to_track_quat("Z", "Y").to_euler()


def _build_preview_rig(armature, frame_start: int, frame_end: int):
    material_joint = _make_material("mocap_preview_joint", (0.0, 0.65, 0.6, 1.0))
    material_bone = _make_material("mocap_preview_bone", (0.06, 0.22, 0.28, 1.0))
    segments = _bone_segments(armature)
    joint_names = sorted({name for pair in segments for name in pair})
    joint_objects = {name: _create_sphere(f"joint_{name}", material_joint) for name in joint_names}
    bone_objects = [_create_cylinder(f"bone_{parent}_to_{child}", material_bone) for parent, child in segments]

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        positions = {name: _joint_position(armature, name) for name in joint_names}
        for name, obj in joint_objects.items():
            obj.location = positions[name]
            obj.keyframe_insert(data_path="location", frame=frame)
        for obj, (parent, child) in zip(bone_objects, segments):
            _orient_cylinder(obj, positions[parent], positions[child])
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    armature.hide_viewport = True
    armature.hide_render = True
    return [*joint_objects.values(), *bone_objects]


def _bounds(objects) -> tuple[Vector, Vector]:
    points: list[Vector] = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return Vector((-1.0, -1.0, 0.0)), Vector((1.0, 1.0, 2.0))
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return minimum, maximum


def _setup_camera_and_light(objects) -> None:
    minimum, maximum = _bounds(objects)
    center = (minimum + maximum) * 0.5
    size = max((maximum - minimum).length, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - size * 1.4, center.z + size * 1.6))
    light = bpy.context.object
    light.name = "mocap_preview_key_light"
    light.data.energy = 500.0
    light.data.size = max(size, 2.0)

    bpy.ops.object.camera_add(location=(center.x, center.y - size * 2.2, center.z + size * 0.55), rotation=(math.radians(78), 0.0, 0.0))
    camera = bpy.context.object
    camera.name = "mocap_preview_camera"
    camera.data.lens = 35
    bpy.context.scene.camera = camera


def _configure_render(output_path: Path, fps: float | None, width: int, height: int, frame_start: int, frame_end: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    if fps is not None and fps > 0:
        scene.render.fps = max(1, int(round(fps)))
        scene.render.fps_base = 1.0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(output_path.resolve())
    scene.world.color = (0.96, 0.97, 0.98)


def main() -> None:
    args = parse_args()
    _clear_scene()
    armature = _import_fbx(args.fbx)
    frame_start, frame_end = _scene_frame_range(armature)
    preview_objects = _build_preview_rig(armature, frame_start, frame_end)
    _setup_camera_and_light(preview_objects)
    _configure_render(args.output, args.fps, args.width, args.height, frame_start, frame_end)
    bpy.ops.render.render(animation=True)
    if not args.output.exists() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Failed to render FBX preview: {args.output}")
    print(f"Saved FBX preview: {args.output.resolve()}")


if __name__ == "__main__":
    main()
