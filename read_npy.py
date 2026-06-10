import numpy as np
import os

# Đường dẫn tới file
file_path = "/workspace/MoCap/output/pose3d.npy"

# Đọc file
data = np.load(file_path, allow_pickle=True)

print("=" * 50)
print("File:", file_path)
print("File size on disk:", os.path.getsize(file_path), "bytes")
print("=" * 50)

print("Type:", type(data))
print("dtype:", data.dtype)
print("shape:", data.shape)
print("ndim:", data.ndim)
print("size (number of elements):", data.size)
print("memory usage:", data.nbytes, "bytes")

# Nếu là dữ liệu số thì in min/max
if np.issubdtype(data.dtype, np.number):
    print("min:", data.min())
    print("max:", data.max())
    print("mean:", data.mean())

print("=" * 50)
print("First few elements:")
print(data.flat[:10])   # 10 phần tử đầu tiên
print("=" * 50)