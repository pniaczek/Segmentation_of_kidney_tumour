# datasets/kidney_nifti_patches.py
import os
from glob import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

WINDOW_MIN, WINDOW_MAX = -54.0, 242.0

def _window_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), WINDOW_MIN, WINDOW_MAX)
    return 2.0 * (arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN) - 1.0

def _safe_shape(path: Path):
    try:
        ni = nib.load(str(path))
        return ni.shape  # tuple
    except Exception:
        return None

def _num_voxels(shape):
    if not shape: return 0
    n = 1
    for s in shape:
        n *= max(int(s), 0)
    return n

def _has_zero_voxels(path: Path) -> bool:
    shape = _safe_shape(path)
    if shape is None:  # unreadable
        return True
    if _num_voxels(shape) == 0:
        return True
    # some corrupt files report odd shapes; try actually reading minimal header data_len
    try:
        arr = nib.load(str(path)).get_fdata(dtype=np.float32, caching='unchanged')
        return arr.size == 0
    except Exception:
        return True

def _load_image_zyx_float(img_path: Path) -> np.ndarray:
    ni = nib.load(str(img_path))
    arr = ni.get_fdata()
    if arr.ndim == 4:  # take first channel if present
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise RuntimeError(f"Image has unexpected ndim={arr.ndim} at {img_path}")
    return _window_to_unit(arr)

def _label_to_int_like(lbl_path: Path, img_shape: tuple) -> np.ndarray:
    ni = nib.load(str(lbl_path))
    arr = ni.get_fdata()
    if arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.argmax(arr, axis=-1)
    elif arr.ndim == 1:
        vox = int(np.prod(img_shape))
        if arr.size == vox:
            arr = arr.reshape(img_shape)
        else:
            ok = False
            for C in range(2, 9):
                if arr.size == vox * C:
                    arr = np.argmax(arr.reshape((*img_shape, C)), axis=-1)
                    ok = True
                    break
            if not ok:
                raise RuntimeError(f"Cannot reshape 1-D label of size {arr.size} to {img_shape} at {lbl_path}")
    else:
        raise RuntimeError(f"Label has unexpected ndim={arr.ndim} at {lbl_path}")

    arr = np.rint(arr).astype(np.int16, copy=False)
    if arr.shape != img_shape:
        raise RuntimeError(f"Label shape {arr.shape} != image shape {img_shape} at {lbl_path}")
    return arr

class KidneyNiftiPatchesDataset(Dataset):
    """
    Pairs *_image.nii.gz with *_label.nii.gz.

    Returns:
      image  [1,Z,Y,X] float32 in [-1,1]
      target [3,Z,Y,X] float32 (ch0: kidney|tumor|cyst, ch1: tumor|cyst, ch2: tumor)
    """
    def __init__(self, folder: str, return_name: bool = False):
        self.folder = Path(folder)
        cand_images = sorted(glob(os.path.join(folder, "*_image.nii.gz")))
        if not cand_images:
            raise FileNotFoundError(f"No *_image.nii.gz found in: {folder}")
        self.return_name = return_name

        self.images, self.labels, skipped = [], [], []
        for img in cand_images:
            img_p = Path(img)
            lbl_p = img_p.with_name(img_p.name.replace("_image.nii.gz", "_label.nii.gz"))
            if not lbl_p.exists():
                skipped.append((img_p.name, "missing_label"))
                continue

            # Hard prefilter: any zero-voxel or unreadable label/image -> skip
            if _has_zero_voxels(lbl_p):
                skipped.append((img_p.name, "label_zero_or_unreadable"))
                continue
            if _has_zero_voxels(img_p):
                skipped.append((img_p.name, "image_zero_or_unreadable"))
                continue

            # Basic dims
            ishape = _safe_shape(img_p)
            if ishape is None or len(ishape) not in (3, 4):
                skipped.append((img_p.name, f"bad_img_shape={ishape}"))
                continue

            self.images.append(str(img_p))
            self.labels.append(str(lbl_p))

        if skipped:
            print(f"[KidneyNiftiPatchesDataset] Using {len(self.images)} pairs, skipped {len(skipped)} broken pairs.")
            for name, reason in skipped[:20]:
                print(f"  - skipped: {name}  ({reason})")
        if not self.images:
            raise RuntimeError("No valid pairs after filtering.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = Path(self.images[idx])
        lbl_path = Path(self.labels[idx])

        img_zyx = _load_image_zyx_float(img_path)             # (Z,Y,X)
        lbl_zyx = _label_to_int_like(lbl_path, img_zyx.shape) # (Z,Y,X)

        kidney = (lbl_zyx == 1)
        tumor  = (lbl_zyx == 2)
        cyst   = (lbl_zyx == 3)
        ch0 = (kidney | tumor | cyst)
        ch1 = (tumor  | cyst)
        ch2 = tumor

        image = torch.from_numpy(img_zyx).unsqueeze(0).contiguous()
        target = torch.stack([
            torch.from_numpy(ch0.astype(np.float32, copy=False)),
            torch.from_numpy(ch1.astype(np.float32, copy=False)),
            torch.from_numpy(ch2.astype(np.float32, copy=False)),
        ], dim=0).contiguous()

        if self.return_name:
            return image, target, img_path.stem
        return image, target
