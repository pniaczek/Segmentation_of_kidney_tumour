#!/usr/bin/env python3
import os, sys
from pathlib import Path
import nibabel as nib
import numpy as np
from glob import glob
from collections import Counter, defaultdict

def safe_load(path):
    try:
        ni = nib.load(str(path))
        arr = ni.get_fdata(dtype=np.float32, caching='unchanged')
        return arr, ni
    except Exception as e:
        return None, e

def main(root_dir, target_shape=(256,256,336)):
    root = Path(root_dir)
    imgs = sorted(glob(os.path.join(root_dir, "*_cropped_image.nii.gz")))
    if not imgs:
        print(f"No *_cropped_image.nii.gz in {root_dir}")
        sys.exit(1)

    bad = []
    cls_counter = Counter()
    per_case = []

    for ip in imgs:
        ip = Path(ip)
        lp = ip.with_name(ip.name.replace("_cropped_image.nii.gz","_cropped_label.nii.gz"))
        if not lp.exists():
            bad.append((ip.name, "missing_label"))
            continue

        img, e1 = safe_load(ip)
        if img is None:
            bad.append((ip.name, f"unreadable_image:{e1}"))
            continue

        lbl, e2 = safe_load(lp)
        if lbl is None:
            bad.append((ip.name, f"unreadable_label:{e2}"))
            continue

        # shape checks
        if img.ndim != 3:
            bad.append((ip.name, f"img_ndim={img.ndim}"))
            continue
        if lbl.ndim == 4 and lbl.shape[-1] == 1:
            lbl = lbl[..., 0]
        elif lbl.ndim == 4 and lbl.shape[-1] >= 2:
            # one-hot saved by mistake -> argmax
            lbl = np.argmax(lbl, axis=-1)
        elif lbl.ndim != 3:
            bad.append((ip.name, f"lbl_ndim={lbl.ndim}"))
            continue

        if img.shape != lbl.shape:
            bad.append((ip.name, f"shape_mismatch img{img.shape} lbl{lbl.shape}"))
            continue

        # optional: expect your target crop size (pad_to_shape used)
        if target_shape and tuple(img.shape) != tuple(target_shape):
            bad.append((ip.name, f"unexpected_shape {img.shape}"))
            # keep going; not fatal for foreground check

        # label stats
        uvals = np.unique(lbl.astype(np.int16, copy=False))
        fgvox = int((lbl > 0).sum())
        for uv in uvals:
            cls_counter[int(uv)] += (lbl == uv).sum()

        per_case.append((ip.name, img.shape, uvals.tolist(), fgvox))
        if fgvox == 0:
            bad.append((ip.name, "empty_label_foreground"))

    # Summary
    print(f"\n==== SUMMARY for {root} ====")
    print(f"Total pairs found: {len(imgs)}")
    print(f"Problematic pairs: {len(bad)}")
    if bad:
        print("First 20 problems:")
        for name, reason in bad[:20]:
            print(f"  - {name}: {reason}")

    print("\nClass voxel counts across all labels:")
    for k in sorted(cls_counter.keys()):
        print(f"  class {k}: {int(cls_counter[k])}")

    # Save detailed report
    rep = root / "cropped_dataset_report.txt"
    with open(rep, "w") as f:
        f.write("name\tshape\tunique_labels\tfgvox\n")
        for name, shape, u, fg in per_case:
            f.write(f"{name}\t{shape}\t{u}\t{fg}\n")
        f.write("\n[PROBLEMS]\n")
        for name, reason in bad:
            f.write(f"{name}\t{reason}\n")
    print(f"\nWrote report: {rep}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/check_cropped_dataset.py /path/to/cropped_dataset")
        sys.exit(1)
    main(sys.argv[1])
