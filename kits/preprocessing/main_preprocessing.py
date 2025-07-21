import os
import nibabel as nib
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

CT_MIN, CT_MAX = -54, 242
PATCH_SIZE = (256, 256, 256)

def load_nifti(path):
    return nib.load(path).get_fdata()

def rescale_ct(img):
    img = np.clip(img, CT_MIN, CT_MAX)
    img = 2 * (img - CT_MIN) / (CT_MAX - CT_MIN) - 1  # scale to [-1, 1]
    return img

def center_crop(vol, size):
    crop = []
    for s, dim in zip(size, vol.shape):
        start = max((dim - s) // 2, 0)
        crop.append(slice(start, start + s))
    return vol[tuple(crop)]

def preprocess_case(img_path, seg_path, out_path):
    img = load_nifti(img_path)
    seg = load_nifti(seg_path)

    img = rescale_ct(img)
    img = center_crop(img, PATCH_SIZE)
    seg = center_crop(seg, PATCH_SIZE).astype(np.uint8)

    # Convert to tensors
    img_tensor = torch.tensor(img[np.newaxis], dtype=torch.float32)
    seg_tensor = torch.tensor(seg, dtype=torch.long)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'image': img_tensor, 'label': seg_tensor}, out_path)

def main(data_root, output_dir):
    cases = sorted(glob(os.path.join(data_root, "case_*")))
    for case_dir in tqdm(cases, desc="Preprocessing"):
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        out_path = os.path.join(output_dir, case_id + ".pt")
        preprocess_case(img_path, seg_path, out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m kits23.data.preprocessing <input_dir> <output_dir>")
        exit(1)
    main(sys.argv[1], sys.argv[2])
