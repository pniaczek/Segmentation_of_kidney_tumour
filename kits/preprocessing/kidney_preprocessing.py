# kidney_roi_preprocessing.py

import os
import nibabel as nib
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split

INPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/dataset"
OUTPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_detection_processed"

CT_MIN, CT_MAX = -54, 242
TARGET_SHAPE = (128, 128, 128)


def rescale_ct(img):
    img = np.clip(img, CT_MIN, CT_MAX)
    return 2 * (img - CT_MIN) / (CT_MAX - CT_MIN) - 1


def resize_volume(volume, shape):
    from scipy.ndimage import zoom
    factors = [ns / float(osz) for ns, osz in zip(shape, volume.shape)]
    return zoom(volume, zoom=factors, order=1)


def preprocess_case(img_path, seg_path, out_path):
    img = nib.load(img_path).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)

    img = rescale_ct(img)
    mask = (seg > 0).astype(np.uint8)

    img_resized = resize_volume(img, TARGET_SHAPE)
    mask_resized = resize_volume(mask, TARGET_SHAPE)

    img_tensor = torch.tensor(img_resized[np.newaxis], dtype=torch.float32)
    mask_tensor = torch.tensor(mask_resized, dtype=torch.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'image': img_tensor, 'label': mask_tensor}, out_path)


def main():
    all_cases = sorted(glob(os.path.join(INPUT_DIR, "case_*")))
    train_cases, test_cases = train_test_split(all_cases, train_size=391, random_state=42)

    for subset, cases in [('train', train_cases), ('test', test_cases)]:
        for case_path in tqdm(cases, desc=f"Preprocessing {subset}"):
            case_id = os.path.basename(case_path)
            img_path = os.path.join(case_path, "imaging.nii.gz")
            seg_path = os.path.join(case_path, "segmentation.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, subset, case_id + ".pt")
            preprocess_case(img_path, seg_path, out_path)


if __name__ == "__main__":
    main()
