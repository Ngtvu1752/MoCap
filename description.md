# Demo Requirement - Video to 3D Mesh Pipeline

## 1. Objective

Xây dựng một bản demo có khả năng chuyển đổi một video người thật đang thực hiện động tác (ví dụ: nhảy, đi bộ, tạo dáng...) thành một biểu diễn 3D dưới dạng skeleton/mesh.

Đây là bước đầu tiên trong hệ thống Motion Capture và sẽ là nền tảng để phát triển Avatar Retargeting trong tương lai.

---

# 2. Input

Input của hệ thống:

```
Video (.mp4)
```

Ví dụ:

```
input/
    dance.mp4
```

Yêu cầu:

* Video chỉ chứa một người
* Camera cố định hoặc ít rung
* Toàn thân xuất hiện trong khung hình
* FPS khoảng 24~60

---

# 3. Output

Output mong muốn:

```
output/
    pose2d.npy
    pose3d.npy
    mesh.mp4
```

Trong đó:

## pose2d.npy

Chứa chuỗi keypoints 2D:

```
(T, K, 2)
```

Ví dụ:

```
300 frames

↓

300 × 133 × 2
```

---

## pose3d.npy

Chứa chuỗi keypoints 3D:

```
(T, K, 3)
```

Ví dụ:

```
300 × 133 × 3
```

---

## mesh.mp4

Video hiển thị nhân vật dạng mesh hoặc skeleton 3D thực hiện đúng chuyển động của người trong video.

---

# 4. Pipeline

```
Video

↓

VideoReader

↓

RTMPose

↓

2D Keypoints

↓

PoseMamba

↓

3D Skeleton

↓

Mesh Renderer

↓

Mesh Video
```

---

# 5. Module Design

## Module 1

Video Reader

Nhiệm vụ:

* đọc video
* lấy metadata
* trả về từng frame

Input:

```
video.mp4
```

Output:

```
frame
```

---

## Module 2

RTMPose

Nhiệm vụ:

* phát hiện pose 2D

Input:

```
frame
```

Output:

```
keypoints2D
```

---

## Module 3

PoseMamba

Nhiệm vụ:

* nâng từ 2D pose thành 3D pose

Input:

```
(T,K,2)
```

Output:

```
(T,K,3)
```

---

## Module 4

Mesh Renderer

Nhiệm vụ:

* dựng skeleton hoặc mesh
* render thành video

Input:

```
(T,K,3)
```

Output:

```
mesh.mp4
```

---

# 6. Project Structure

```
project/

│
├── input/
│
├── output/
│
├── checkpoints/
│
├── src/
│
│    io/
│        video_reader.py
│
│    pose2d/
│        rtmpose_estimator.py
│
│    pose3d/
│        posemamba_estimator.py
│
│    renderer/
│        mesh_renderer.py
│
│    pipeline.py
│
└── main.py
```

---

# 7. Development Roadmap

## Phase 1

```
Video

↓

VideoReader
```

Deliverable:

* đọc được toàn bộ video
* lấy metadata
* iterator theo frame

---

## Phase 2

```
Video

↓

RTMPose

↓

2D Keypoints
```

Deliverable:

* xuất pose2d.npy

---

## Phase 3

```
2D Keypoints

↓

PoseMamba

↓

3D Skeleton
```

Deliverable:

* xuất pose3d.npy

---

## Phase 4

```
3D Skeleton

↓

Renderer

↓

Mesh Video
```

Deliverable:

* xuất mesh.mp4

---

# 8. Future Extensions

Bản demo này chỉ là nền tảng.

Các chức năng sẽ phát triển sau:

* Avatar Retargeting
* IK Solver
* Bone Mapping
* Motion Editing
* Multi-character Support
* Real-time Inference
* Unity/Unreal Export
* FBX/GLB/USD Export

---

# 9. Success Criteria

Một bản demo được coi là thành công khi:

* Đọc được video đầu vào
* Trích xuất được chuỗi 2D pose
* Sinh được chuỗi 3D skeleton
* Render thành một video mesh/skeleton thể hiện đúng chuyển động của người trong video
* Kiến trúc module đủ linh hoạt để tích hợp Avatar Retargeting trong các giai đoạn tiếp theo.
