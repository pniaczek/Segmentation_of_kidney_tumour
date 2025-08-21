# datasets/kidney_dataset.py
import os
import re
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


def _lower_name(p: str) -> str:
    return os.path.basename(p).lower()


def _index_labels(folder: str) -> Dict[str, str]:
    # accept either "label" or "labels" in filename (case-insensitive)
    lab_paths = glob(os.path.join(folder, "*label*.nii.gz")) + glob(os.path.join(folder, "*labels*.nii.gz"))
    idx = {}
    for p in lab_paths:
        idx[_lower_name(p)] = p
    return idx


def _numeric_tokens(name: str) -> List[str]:
    # return all numeric substrings to help matching by ID
    return re.findall(r"\d+", name)


def _candidate_label_names(img_name: str) -> List[str]:
    # generate plausible label filenames from image filename
    cands = set()
    n = img_name

    # common token swaps
    swaps = [
        ("_image_", "_label_"),
        ("_images_", "_labels_"),
        ("-image-", "-label-"),
        ("-images-", "-labels-"),
        (" image ", " label "),
        (" images ", " labels "),
        ("image", "label"),
        ("images", "labels"),
    ]
    for a, b in swaps:
        if a in n:
            cands.add(n.replace(a, b))

    # also try removing 'image' completely and inserting 'label'
    cands.add(n.replace("image", "label"))
    cands.add(n.replace("images", "labels"))

    # Sometimes people use 'img'/'lbl'
    cands.add(n.replace("img", "label"))
    cands.add(n.replace("imgs", "labels"))

    # ensure .nii.gz extension remains
    cands2 = set()
    for c in cands:
        if not c.endswith(".nii.gz"):
            if c.endswith(".nii"):
                c = c + ".gz"
            else:
                c = re.sub(r"(\.nii(\.gz)?)?$", ".nii.gz", c)
        cands2.add(c)
    return list(cands2)


def _best_label_match(img_path: str, label_index: Dict[str, str]) -> str:
    img_name = _lower_name(img_path)

    # 1) exact candidates
    for cand in _candidate_label_names(img_name):
        if cand in label_index:
            return label_index[cand]

    # 2) match by numeric id (e.g., 0001) + presence of 'label'
    img_nums = set(_numeric_tokens(img_name))
    if img_nums:
        for lab_name in label_index.keys():
            if img_nums.intersection(_numeric_tokens(lab_name)):
                return label_index[lab_name]

    # 3) prefix/suffix heuristic: keep same prefix before 'image' and suffix after
    m = re.split(r"image|images", img_name, maxsplit=1)
    if len(m) == 2:
        prefix, suffix = m[0], m[1]
        for lab_name in label_index.keys():
            if "label" in lab_name or "labels" in lab_name:
                if lab_name.startswith(prefix) and lab_name.endswith(suffix):
                    return label_index[lab_name]

    # no match
    return ""


class KidneyDataset(Dataset):
    """
    NIfTI dataset (no normalization/clipping inside):
      - image files: *image*.nii.gz  (float preprocessed)
      - label files: *label*.nii.gz or *labels*.nii.gz (ints: 0=bg, 1=kidney, 2=tumor, 3=cyst)

    Output:
      image  -> FloatTensor [1, D, H, W]
      target -> FloatTensor [3, D, H, W]
                ch0: kidney ∪ tumor ∪ cyst
                ch1: tumor ∪ cyst
                ch2: tumor
    """
    def __init__(self, folder: str, return_name: bool = False):
        self.folder = folder
        self.return_name = return_name

        img_paths = sorted(glob(os.path.join(folder, "*image*.nii.gz")))
        if not img_paths:
            raise FileNotFoundError(f"No image NIfTI files (*image*.nii.gz) found in: {folder}")

        label_index = _index_labels(folder)

        pairs: List[Tuple[str, str]] = []
        unmatched: List[str] = []

        for ip in img_paths:
            lp = _best_label_match(ip, label_index)
            if lp:
                pairs.append((ip, lp))
            else:
                unmatched.append(os.path.basename(ip))

        if not pairs:
            raise FileNotFoundError(
                "Found images but could not pair any with labels in: {}\n"
                "Sample image names: {}\n"
                "Sample label names: {}".format(
                    folder,
                    ", ".join([p for p in unmatched[:5]]) if unmatched else "(none)",
                    ", ".join(list(label_index.keys())[:5]) if label_index else "(no label files found)"
                )
            )

        if unmatched:
            # Non-fatal: warn once (many datasets contain extras)
            print(f"[KidneyDataset] Warning: {len(unmatched)} images had no matching label. "
                  f"First few: {unmatched[:5]}")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        img_np = np.asanyarray(nib.load(img_path).get_fdata(), dtype=np.float32)  # already preprocessed
        lbl_np = np.asanyarray(nib.load(lbl_path).get_fdata(), dtype=np.int16)    # integer labels

        image = torch.from_numpy(img_np).unsqueeze(0)  # [1, D, H, W]
        label = torch.from_numpy(lbl_np).long()        # [D, H, W]

        kidney = (label == 1)
        tumor  = (label == 2)
        cyst   = (label == 3)

        k_and_m = (kidney | tumor | cyst)
        masses  = (tumor | cyst)
        tumor_c = tumor

        target = torch.stack(
            [k_and_m.float(), masses.float(), tumor_c.float()],
            dim=0
        )  # [3, D, H, W]

        if self.return_name:
            base = os.path.basename(img_path).replace(".nii.gz", "")
            return image, target, base
        else:
            return image, target
