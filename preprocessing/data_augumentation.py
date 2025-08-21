#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
import numpy as np

from monai.utils import set_determinism
from monai.transforms import (
    LoadImaged, EnsureTyped, EnsureChannelFirstd,
    RandRotate90d, RandFlipd, RandShiftIntensityd,
    RandAffined, RandZoomd, RandGaussianNoised,
    SaveImaged, Compose
)


def get_transform_list(aug_config, output_dir, postfix):
    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ]

    for aug_name, params in aug_config.items():
        if not params.get("enable", False):
            continue
        prob = params.get("prob", 1.0)
        if aug_name == "rotate90":
            t.append(RandRotate90d(
                keys=["image", "label"],
                prob=prob,
                spatial_axes=params.get("axes", [0, 1, 2]),
            ))
        elif aug_name == "flip":
            t.append(RandFlipd(
                keys=["image", "label"],
                prob=prob,
                spatial_axis=params.get("axis", 0),
            ))
        elif aug_name == "intensity":
            t.append(RandShiftIntensityd(
                keys=["image"],
                offsets=params.get("offsets", (-0.1, 0.1)),
                prob=prob,
            ))
        elif aug_name == "affine":
            t.append(RandAffined(
                keys=["image", "label"],
                prob=prob,
                rotate_range=params.get("rotate_range", (0.1, 0.1, 0.1)),
                scale_range=params.get("scale_range", (0.1, 0.1, 0.1)),
            ))
        elif aug_name == "zoom":
            t.append(RandZoomd(
                keys=["image", "label"],
                prob=prob,
                min_zoom=params.get("min_zoom", 0.9),
                max_zoom=params.get("max_zoom", 1.1),
            ))
        elif aug_name == "noise":
            t.append(RandGaussianNoised(
                keys=["image"],
                prob=prob,
                mean=params.get("mean", 0.0),
                std=params.get("std", 0.01),
            ))

    t.append(SaveImaged(
        keys=["image", "label"],
        output_dir=output_dir,
        output_postfix=postfix,
        output_ext=".nii.gz",
        resample=False,
        mode="nearest",
        separate_folder=False,
        print_log=False,
    ))

    return Compose(t)


def augment_dataset(src_dir, dst_dir, aug_config, repeats=1, seed=42):
    set_determinism(seed)
    os.makedirs(dst_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(src_dir, "case_*_cropped_image.nii.gz")))
    for img_path in tqdm(image_paths, desc="Cases"):
        basename = os.path.basename(img_path).replace("_cropped_image.nii.gz", "")
        lbl_path = os.path.join(src_dir, f"{basename}_cropped_label.nii.gz")
        if not os.path.isfile(lbl_path):
            tqdm.write(f"Warning: brak labelu dla {basename}")
            continue

        for r in range(repeats):
            postfix = f"aug{r:02d}"
            transform = get_transform_list(aug_config, dst_dir, postfix)
            transform({"image": img_path, "label": lbl_path})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MONAI 3D Augmentation (kompatybilny z wersjami ≥ 0.8)")
    parser.add_argument("--src",   type=str, required=False,
                        default="./cropped_dataset",
                        help="Wejściowy folder z ~_cropped_image.nii.gz i ~_cropped_label.nii.gz")
    parser.add_argument("--dst",   type=str, required=False,
                        default="/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augumented_dataset",
                        help="Wyjściowy folder gdzie zostaną zapisane pliki .nii.gz")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Ile augmentowanych wersji na przypadek")
    parser.add_argument("--seed",  type=int, default=0, help="Deterministyczne rozsiewanie randoma")

    args = parser.parse_args()

    aug_config = {
        "rotate90":   {"enable": True,  "prob": 0.5, "axes": [0, 2]},
        "flip":       {"enable": True,  "prob": 0.5, "axis": 1},
        "intensity":  {"enable": False, "prob": 1.0, "offsets": (-0.1, 0.1)},
        "affine":     {"enable": True,  "prob": 0.3, "rotate_range": (0.1, 0.1, 0.1), "scale_range": (0.1, 0.1, 0.1)},
        "zoom":       {"enable": False, "prob": 0.3, "min_zoom": 0.9, "max_zoom": 1.1},
        "noise":      {"enable": False, "prob": 0.3, "mean": 0.0, "std": 0.01},
    }

    augment_dataset(args.src, args.dst, aug_config, args.repeats, args.seed)
