# Unity Mecanim Retargeting Pipeline

Kế hoạch Ánh xạ Chuyển động Tự động cho Unity (Mecanim Pipeline)

Mục tiêu của Phase này là biến dữ liệu đầu ra từ MotionBERT (.npy) thành một chuỗi chuyển động hoàn chỉnh có thể gắn lên bất kỳ nhân vật 3D (Avatar) nào trong Unity (Metaverse, Game, VR) một cách tự động và ổn định nhất.

Thay vì tự xây dựng công cụ tính toán ma trận xoay bằng Python và sử dụng định dạng cũ .bvh, quy trình mới sẽ sử dụng Blender Python API (chạy ngầm) làm công cụ nướng chuyển động (Baking) và xuất trực tiếp ra định dạng chuẩn công nghiệp .fbx cho Unity.

1. Dữ Liệu Nguồn (Từ MotionBERT)

Quy trình chỉ cần 2 file chính từ kết quả của MotionBERT:

smpl_theta.npy: (Kích thước T x 82) - Chứa 72 tham số xoay Axis-Angle của 24 khớp xương.

smpl_joints3d.npy: (Kích thước T x 17 x 3) - Tọa độ khớp 3D trong không gian crop của HMR. Pelvis (index 0) chỉ cung cấp chuyển động root tương đối còn giữ lại trong crop, không phải trajectory toàn cục trong cảnh.

MotionBERT HMR chuẩn hóa crop theo từng frame và không dự đoán camera/global translation. Vì vậy FBX có thể giữ body motion và một phần pelvis motion, nhưng không thể tái tạo chính xác việc nhân vật đi xuyên qua khung hình chỉ từ `smpl_theta.npy` và `smpl_joints3d.npy`. Global locomotion cần một bước camera/trajectory recovery riêng.

2. Mục Tiêu Đầu Ra (Target Outcome)

Hệ thống chỉ sinh ra MỘT file duy nhất sau quy trình tự động:

output/<video>/retarget/animated_smpl.fbx


Lưu ý: File này chỉ chứa khung xương SMPL đang chuyển động, không chứa Mesh (da thịt) để tối ưu dung lượng.

3. Cấu Trúc Hệ Thống Mới (Đã triển khai)

src/retarget/
  ├── fbx_exporter.py                 # Validate input và gọi Blender headless
  └── blender/
      └── bake_smpl_fbx.py            # Script chạy bên trong Blender

assets/retarget/
  └── smpl_base_tpose.fbx             # SMPL T-pose source rig

export_smpl_fbx.py                     # CLI chính

output/<video>/retarget/
  └── animated_smpl.fbx               # Skeleton animation cho Unity Mecanim


4. Lộ Trình Triển Khai (Work Order)

Quy trình thực hiện được chia làm 3 Giai đoạn chính (Phát triển -> Tự động hóa -> Tích hợp Unity):

Giai đoạn 1: Chuẩn bị Môi trường (Setup)

Tạo File Base FBX: Mở Blender, tạo một khung xương SMPL với 24 khớp chuẩn hóa ở tư thế T-Pose. Lưu thành assets/retarget/smpl_base_tpose.fbx.

Cài đặt lõi script Blender: triển khai src/retarget/blender/bake_smpl_fbx.py. File này đảm nhiệm việc đọc .npy, chuyển Axis-Angle sang Quaternion để chống Gimbal Lock, và nạp vào khung xương Base.

Giai đoạn 2: Tự Động Hóa Backend (Automation)

Xây dựng Python Trigger: Viết một hàm Python trên server sử dụng thư viện subprocess để kích hoạt giao diện dòng lệnh của Blender.

Lệnh chuẩn: python export_smpl_fbx.py --input-dir output/dance2

Kiểm thử quy trình ngầm (Headless Test): Chạy thử quy trình từ đầu đến cuối mà không cần mở giao diện đồ họa. Đảm bảo file FBX được sinh ra có dung lượng hợp lý và chứa Keyframe.

Giai đoạn 3: Tích Hợp Unity (Unity Mecanim Retargeting)

(Các bước này do Game Developer hoặc Unity Editor Script thực hiện)
5.  Cấu hình Humanoid cho Source: Nhập file animated_smpl.fbx vào thư mục Unity. Chọn file, mở bảng Inspector -> Rig -> Animation Type: Humanoid -> Apply.
6.  Cấu hình Humanoid cho Target Avatar: Đảm bảo Avatar đích của bạn (Mixamo, ReadyPlayerMe, v.v.) cũng được set Rig thành Humanoid.
7.  Áp dụng chuyển động: Mở Animator Controller của Avatar đích, kéo đoạn Animation Clip từ file animated_smpl.fbx vào. Unity sẽ tự động nội suy và ánh xạ (Retarget) chuyển động từ khung xương SMPL sang Avatar đích một cách mượt mà.
8.  Xử lý IK (Nếu cần): Bật tính năng Foot IK trong Animator của Unity để khóa gót chân chạm đất, xử lý lỗi trượt băng (Ice-skating) nếu có.

5. Những Hạng Mục Đã Bị Xóa Bỏ (Deprecated)

So với kế hoạch cũ, các hạng mục sau đây đã bị loại bỏ vì Unity Mecanim đã tự động giải quyết:

⚠️ BVH exporter được chuyển vào src/retarget/legacy/ và chỉ dùng để debug.

❌ Xóa retarget_solver.py (Không cần tự code thuật toán ánh xạ).

❌ Xóa retarget_map.json (Unity Humanoid tự động nhận diện tên xương chuẩn).

❌ Xóa các bước tính toán bù trừ hình học, T-Pose vs A-Pose (Unity Avatar Configuration có sẵn tính năng Enforce T-Pose).