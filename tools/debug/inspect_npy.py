from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a .npy file shape, dtype, and simple stats.")
    parser.add_argument("path", type=Path, help="Path to the .npy file to inspect")
    parser.add_argument("--limit", type=int, default=10, help="Number of flattened values to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.path, allow_pickle=True)

    print("=" * 50)
    print("File:", args.path)
    print("File size on disk:", args.path.stat().st_size, "bytes")
    print("=" * 50)
    print("Type:", type(data))
    print("dtype:", data.dtype)
    print("shape:", data.shape)
    print("ndim:", data.ndim)
    print("size (number of elements):", data.size)
    print("memory usage:", data.nbytes, "bytes")

    if np.issubdtype(data.dtype, np.number) and data.size:
        print("min:", data.min())
        print("max:", data.max())
        print("mean:", data.mean())

    print("=" * 50)
    print("First values:")
    print(data.flat[: args.limit])
    print("=" * 50)


if __name__ == "__main__":
    main()
