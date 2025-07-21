import os
import sys
from glob import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def save_slice_png(case_path, slice_index, axis='axial', output_dir='slices_png'):
    data = torch.load(case_path)
    img = data['image'].squeeze(0).numpy()
    label = data['label'].numpy()

    if axis == 'axial':
        img_slice = img[:, :, slice_index]
        label_slice = label[:, :, slice_index]
        plane = 'axial'
    elif axis == 'sagittal':
        img_slice = img[slice_index, :, :]
        label_slice = label[slice_index, :, :]
        plane = 'sagittal'
    elif axis == 'coronal':
        img_slice = img[:, slice_index, :]
        label_slice = label[:, slice_index, :]
        plane = 'coronal'
    else:
        raise ValueError("Oś musi być jedną z: 'axial', 'sagittal', 'coronal'")

    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(
        os.path.join(output_dir, f"{os.path.basename(case_path)[:-3]}_{plane}_{slice_index}_img.png"),
        img_slice,
        cmap='gray'
    )

    plt.imsave(
        os.path.join(output_dir, f"{os.path.basename(case_path)[:-3]}_{plane}_{slice_index}_label.png"),
        label_slice,
        cmap='jet'
    )

def main(input_dir, output_dir, slice_index=100, axis='axial'):
    os.makedirs(output_dir, exist_ok=True)
    cases = sorted(glob(os.path.join(input_dir, "case_*.pt")))

    for case_path in tqdm(cases, desc=f"Saving {axis} slices at index {slice_index}"):
        save_slice_png(case_path, slice_index, axis, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m kits23.visualizations.visualizations <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    main(input_dir, output_dir, slice_index=100, axis='axial')
