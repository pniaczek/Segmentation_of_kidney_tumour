#!/usr/bin/env python3
import argparse
import os
import numpy as np

def load_any(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        return np.load(file_path, allow_pickle=False)
    # Handle .nii and .nii.gz
    if ext == ".gz" or ext == ".nii":
        import nibabel as nib  # lazy import so .npy users don't need it
        img = nib.load(file_path)
        return img.get_fdata()  # float64 ndarray
    raise ValueError(f"Unsupported file type: {file_path}")

def analyze_file(file_path: str):
    data = load_any(file_path)

    # Ensure numeric array
    if not np.issubdtype(data.dtype, np.number):
        data = data.astype(np.float32)

    stats = {
        "file": file_path,
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }

    # Pretty print
    print(f"File: {stats['file']}")
    print(f"Shape: {stats['shape']}")
    print(f"DType: {stats['dtype']}")
    print(f"Min:   {stats['min']}")
    print(f"Max:   {stats['max']}")
    print(f"Mean:  {stats['mean']}")
    print(f"Std:   {stats['std']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute min/max/mean/std for a .nii/.nii.gz or .npy file"
    )
    parser.add_argument("file", help="Path to the file")
    args = parser.parse_args()
    analyze_file(args.file)
