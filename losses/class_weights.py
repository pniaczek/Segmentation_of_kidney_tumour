import torch
import os
from glob import glob
from tqdm import tqdm

def compute_class_weights_partial(data_folder, num_classes=4, skip_classes=[0], num_samples=391):
    class_counts = torch.zeros(num_classes)
    file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))[:num_samples]

    for path in tqdm(file_paths, desc="Counting voxels"):
        label = torch.load(path, map_location="cpu")["label"]
        for cls in range(num_classes):
            class_counts[cls] += (label == cls).sum()

    inv_freq = 1.0 / (class_counts + 1e-8)
    for skip_cls in skip_classes:
        inv_freq[skip_cls] = 0.0

    weights = inv_freq / inv_freq.sum()

    print(f"Voxel counts: {class_counts.tolist()}")
    print(f"Computed weights: {weights.tolist()}")
    return weights
