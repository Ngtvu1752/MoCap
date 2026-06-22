> **Legacy:** Tài liệu này chỉ còn dùng để debug BVH. Pipeline chính: UNITY_MECANIM_PIPELINE.md.

# Avatar Retargeting And Unity/Unreal Export Plan

Muc tieu phase nay la bien output Human Mesh Recovery hien tai thanh animation co the gan len avatar rigged, sau do export sang Blender, Unity hoac Unreal de dung trong metaverse/virtual production/game prototype.

Pipeline hien tai da tao duoc source motion dang SMPL:

```text
output/<video>/smpl_theta.npy
output/<video>/smpl_joints3d.npy
output/<video>/smpl_vertices.npy
output/<video>/metadata.json
```

Trong do file quan trong nhat cho retargeting la:

```text
smpl_theta.npy    # (T,82), gom 72 SMPL pose params + 10 shape betas
smpl_joints3d.npy # (T,17,3), dung de debug root motion/scale/contact
metadata.json     # fps, frame count, pipeline config
```

Tu day tro di khong can chay MotionBERT nua. MotionBERT chi tao source SMPL motion. Avatar Retargeting la bai toan animation/rig/export.

## Target Outcome

Output mong muon:

```text
output/<video>/retarget/source_smpl.bvh
output/<video>/retarget/target_avatar_motion.bvh
output/<video>/retarget/retargeted_avatar.fbx
output/<video>/retarget/retarget_metadata.json
```

Hoac output trung gian:

```text
output/<video>/retarget/source_motion.npz
output/<video>/retarget/target_local_rotations.npy
output/<video>/retarget/target_root_translation.npy
output/<video>/retarget/target_bone_names.json
```

## Phase 1: Normalize Source SMPL Motion

Input:

```text
output/<video>/smpl_theta.npy
output/<video>/smpl_joints3d.npy
output/<video>/metadata.json
```

Tasks:

```text
smpl_theta.npy
-> split pose axis-angle: (T,24,3)
-> split shape betas: (T,10)
-> get root translation from smpl_joints3d[:,0,:]
-> convert axis-angle to quaternion/Euler
-> normalize coordinate system
-> preserve fps/frame count from metadata.json
```

Output:

```text
output/<video>/retarget/source_motion.npz
```

Suggested module:

```text
src/retarget/legacy/smpl_motion.py
```

## Phase 2: Define SMPL Skeleton

Create a canonical SMPL skeleton definition:

```text
configs/legacy/smpl_skeleton.json
```

It should contain:

```text
24 SMPL joint names
parent indices
rest pose offsets
rotation convention
coordinate convention
```

Example structure:

```json
{
  "name": "SMPL_24",
  "rotation_format": "axis_angle",
  "coordinate_system": "motionbert_smpl",
  "joints": ["pelvis", "left_hip", "right_hip"],
  "parents": [-1, 0, 0],
  "offsets": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
}
```

Output:

```text
configs/legacy/smpl_skeleton.json
```

Suggested module:

```text
src/retarget/legacy/smpl_skeleton.py
```

## Phase 3: Export Source SMPL Motion To BVH

BVH is the recommended first export target because it is easy to inspect in Blender.

Input:

```text
source_motion.npz
configs/legacy/smpl_skeleton.json
```

Tasks:

```text
build BVH HIERARCHY from SMPL skeleton
write MOTION frames from root translation + local rotations
export source_smpl.bvh
```

Output:

```text
output/<video>/retarget/source_smpl.bvh
```

Success criteria:

```text
BVH imports into Blender
motion duration matches video
root motion is visible
limbs move in plausible directions
no major left/right inversion
```

Suggested files:

```text
src/retarget/legacy/bvh_exporter.py
export_smpl_bvh.py
```

## Phase 4: Prepare Target Avatar

Target avatar should be a rigged character, not just a static mesh.

Recommended target asset formats:

```text
FBX
GLB / glTF
VRM
BLEND
```

Required target data:

```text
avatar mesh
armature/skeleton
skinning weights
bone names
rest pose, usually T-pose or A-pose
bone hierarchy
```

Recommended project structure:

```text
avatars/<avatar_name>/avatar.fbx
avatars/<avatar_name>/skeleton.json
avatars/<avatar_name>/rest_pose.json
avatars/<avatar_name>/retarget_map.json
```

The target skeleton file should contain:

```text
bone names
parent hierarchy
rest local transforms
rest world transforms
coordinate system
unit scale
```

Suggested module:

```text
src/retarget/target_skeleton.py
```

## Phase 5: Create Retarget Map

Create a mapping from SMPL joints to target avatar bones.

Example:

```json
{
  "pelvis": "Hips",
  "left_hip": "LeftUpperLeg",
  "left_knee": "LeftLowerLeg",
  "left_ankle": "LeftFoot",
  "right_hip": "RightUpperLeg",
  "right_knee": "RightLowerLeg",
  "right_ankle": "RightFoot",
  "spine": "Spine",
  "neck": "Neck",
  "head": "Head",
  "left_shoulder": "LeftUpperArm",
  "left_elbow": "LeftLowerArm",
  "left_wrist": "LeftHand",
  "right_shoulder": "RightUpperArm",
  "right_elbow": "RightLowerArm",
  "right_wrist": "RightHand"
}
```

Output:

```text
avatars/<avatar_name>/retarget_map.json
```

Suggested module:

```text
src/retarget/retarget_map.py
```

## Phase 6: Build Retarget Solver

The solver transfers motion from SMPL skeleton to target skeleton.

Core algorithm:

```text
for each frame:
  read source local rotations
  apply source rest-pose correction
  apply target rest-pose correction
  map source bones to target bones
  compute target local rotations
  apply root translation scale/axis conversion
```

Handle:

```text
source vs target rest pose difference
A-pose/T-pose offsets
bone direction mismatch
coordinate system conversion
scale ratio
root translation
left/right consistency
```

Output:

```text
output/<video>/retarget/target_local_rotations.npy
output/<video>/retarget/target_root_translation.npy
output/<video>/retarget/target_motion_metadata.json
```

Suggested module:

```text
src/retarget/retarget_solver.py
```

## Phase 7: Export Target Animation

Two export paths are recommended.

### Path A: BVH First

```text
target_local_rotations.npy
+ target skeleton
-> target_avatar_motion.bvh
```

Pros:

```text
easy to debug
simple format
imports into Blender
can be converted to FBX from Blender
```

Cons:

```text
not ideal as final game-engine asset
limited material/mesh support
```

Output:

```text
output/<video>/retarget/target_avatar_motion.bvh
```

### Path B: FBX Through Blender

```text
avatar.fbx
+ target animation
-> Blender Python script
-> retargeted_avatar.fbx
```

Pros:

```text
works well for Unity/Unreal
keeps avatar mesh + rig + animation together
common game-engine workflow
```

Cons:

```text
needs Blender installed
FBX export behavior can vary by Blender version
```

Output:

```text
output/<video>/retarget/retargeted_avatar.fbx
```

Suggested files:

```text
src/retarget/blender_fbx_export.py
scripts/export_retargeted_fbx.py
```

## Phase 8: Cleanup And Motion Quality

After basic retargeting works, add quality passes:

```text
temporal smoothing
root height stabilization
foot contact detection
foot locking
hand jitter smoothing
frame resampling
scale correction
hip/spine stabilization
```

Recommended output:

```text
output/<video>/retarget/retargeted_avatar_clean.fbx
output/<video>/retarget/cleanup_report.json
```

## Recommended Tools

### Python

Use Python for source motion decoding, retarget math, exporters, and automation.

Needed libraries:

```text
numpy
scipy
scipy.spatial.transform.Rotation
```

Optional libraries:

```text
trimesh
pygltflib
bvhio or custom BVH writer
```

Use for:

```text
SMPL theta decoding
axis-angle/quaternion/Euler conversion
BVH generation
retarget map processing
batch automation
```

### Blender

Blender is the most practical bridge tool for this project.

Use Blender for:

```text
import BVH
inspect source SMPL motion
import avatar FBX/GLB/VRM
inspect target rig
apply animation to armature
export FBX for Unity/Unreal
visual debug of retarget results
```

Recommended usage:

```text
start with manual import/export for validation
then automate with Blender Python
```

Important Blender features:

```text
Armature editor
Dope Sheet / Graph Editor
BVH import
FBX import/export
Python scripting
```

### Unity

Unity is useful if the target is real-time avatar playback, mobile, WebGL, VR, AR, or metaverse-style interactive scenes.

Use Unity for:

```text
Humanoid avatar import
Mecanim retargeting
Animator Controller
AnimationClip preview
runtime avatar playback
VRChat/VRM-style workflows
```

Recommended Unity assets/formats:

```text
FBX character with Humanoid rig
FBX animation clip
VRM avatar if targeting VRM ecosystem
```

Unity checks:

```text
Rig tab -> Animation Type = Humanoid
Avatar definition configured
bone mapping valid
animation clip loops/plays correctly
root motion setting correct
```

### Unreal Engine

Unreal is useful if the target is high-fidelity metaverse, virtual production, MetaHuman, cinematic rendering, or game-quality environments.

Use Unreal for:

```text
Skeletal Mesh import
Animation Sequence import
IK Retargeter
Control Rig
MetaHuman retargeting
Sequencer preview
```

Recommended Unreal assets/formats:

```text
FBX skeletal mesh
FBX animation
IK Rig / IK Retargeter assets
```

Unreal checks:

```text
skeleton imported correctly
retarget pose configured
IK Rig chains mapped
Animation Sequence plays correctly
root motion setting correct
```

### Mixamo

Mixamo can be useful for quick avatar/rig testing.

Use for:

```text
getting a simple rigged humanoid FBX
testing retarget pipeline with standard bone names
quick visual validation
```

Limitations:

```text
not ideal for production-quality metaverse avatars
bone naming/conventions differ from SMPL
```

### VRM Tools

If the metaverse target uses VRM avatars, prepare VRM tooling.

Useful tools:

```text
UniVRM for Unity
Blender VRM add-on
```

Use for:

```text
import/export VRM avatars
inspect humanoid bone mapping
prepare anime/metaverse avatars
```

### Optional: Autodesk FBX SDK

Use only if you need direct FBX export from Python without Blender.

Pros:

```text
direct FBX generation
production-oriented format support
```

Cons:

```text
more setup friction
Python bindings can be inconvenient
Blender route is usually faster for this project
```

## Recommended Toolchain For This Project

Start with this minimal toolchain:

```text
Python        -> decode SMPL theta, write BVH, run retarget math
Blender       -> inspect BVH/avatar and export FBX
Unity         -> validate Humanoid retarget/playback
```

Add Unreal later if needed:

```text
Unreal Engine -> validate FBX animation, IK Retargeter, MetaHuman/real-time scene
```

Recommended first milestone:

```text
smpl_theta.npy
-> source_smpl.bvh
-> import into Blender successfully
```

Recommended second milestone:

```text
source_smpl.bvh
+ simple Mixamo humanoid avatar
-> retargeted_avatar.fbx
-> import into Unity as Humanoid animation
```

Recommended third milestone:

```text
retargeted_avatar.fbx
-> import into Unreal
-> play as Animation Sequence or retarget to MetaHuman via IK Retargeter
```

## Proposed Code Structure

```text
src/retarget/
  __init__.py
  smpl_skeleton.py
  smpl_motion.py
  bvh_exporter.py
  target_skeleton.py
  retarget_map.py
  retarget_solver.py
  blender_fbx_export.py

configs/skeletons/
  smpl_skeleton.json

avatars/<avatar_name>/
  avatar.fbx
  skeleton.json
  rest_pose.json
  retarget_map.json

export_smpl_bvh.py
retarget_avatar.py
```

## Work Order

1. Build `smpl_skeleton.json`.
2. Build `smpl_motion.py` to decode `smpl_theta.npy`.
3. Build `bvh_exporter.py`.
4. Export `source_smpl.bvh`.
5. Validate `source_smpl.bvh` in Blender.
6. Prepare one simple rigged avatar, preferably Mixamo first.
7. Create `retarget_map.json`.
8. Build basic `retarget_solver.py`.
9. Export target BVH.
10. Use Blender Python to bind/export `retargeted_avatar.fbx`.
11. Validate in Unity.
12. Validate in Unreal if needed.
13. Add cleanup: smoothing, foot locking, root stabilization.

## Current Readiness

Already available from current pipeline:

```text
source SMPL pose: output/<video>/smpl_theta.npy
source SMPL joints: output/<video>/smpl_joints3d.npy
source mesh preview: output/<video>/human_mesh.mp4
fps/frame_count metadata: output/<video>/metadata.json
```

Still needed:

```text
SMPL skeleton definition
BVH exporter
Target avatar rig
Target skeleton extraction
Retarget map
Retarget solver
FBX/Unity/Unreal export path
```
