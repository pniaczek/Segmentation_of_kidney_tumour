import os
import numpy as np
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt

INPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_dataset"
OUTPUT_PATH = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/visualizations"
CLASSES = [1, 2, 3]

os.makedirs(OUTPUT_PATH, exist_ok=True)

sum_per_class = {c: None for c in CLASSES}
num_cases = 0

label_paths = sorted(glob(os.path.join(INPUT_DIR, "*_label.nii.gz")))
print(f"Found {len(label_paths)} label files.")

for label_path in label_paths:
    seg = nib.load(label_path).get_fdata().astype(np.uint8)
    unique_labels = np.unique(seg)
    if 3 in unique_labels:
        print(f" Class 3 found in: {os.path.basename(label_path)}")

    num_cases += 1

    for c in CLASSES:
        mask = (seg == c).astype(np.uint8)

        if sum_per_class[c] is None:
            sum_per_class[c] = mask
        else:
            if mask.shape != sum_per_class[c].shape:
                print(f"[Warning] Shape mismatch in {label_path} for class {c}: {mask.shape} vs {sum_per_class[c].shape}")
                continue
            sum_per_class[c] += mask


for c in CLASSES:
    summed = sum_per_class[c]
    if summed is None:
        continue

    proj_z = summed.sum(axis=(1, 2)) / num_cases
    proj_y = summed.sum(axis=(0, 2)) / num_cases
    proj_x = summed.sum(axis=(0, 1)) / num_cases

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(proj_z)
    axes[0].set_title(f'Class {c} - Z axis (axial)')
    axes[0].set_xlabel('Z slice')
    axes[0].set_ylabel('Avg count')

    axes[1].plot(proj_y)
    axes[1].set_title(f'Class {c} - Y axis (coronal)')
    axes[1].set_xlabel('Y row')

    axes[2].plot(proj_x)
    axes[2].set_title(f'Class {c} - X axis (sagittal)')
    axes[2].set_xlabel('X col')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_PATH, f"class_{c}_axis_distribution.png")
    plt.savefig(output_file)
    plt.close()

    print(f" Saved: {output_file}")
