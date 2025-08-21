#!/usr/bin/env python3
import argparse, random, json
from pathlib import Path
import numpy as np
import nibabel as nib

CLIP_MIN, CLIP_MAX = -80.0, 310.0

def list_pairs(in_dir: Path):
    imgs = sorted(p for p in in_dir.glob("*.nii.gz") if "_image_" in p.name)
    pairs = []
    for ip in imgs:
        lp = in_dir / ip.name.replace("_image_", "_label_")
        if lp.exists():
            pairs.append((ip, lp))
        else:
            print(f"[WARN] missing label for {ip.name}")
    return pairs

def clip(arr: np.ndarray):
    return np.clip(arr.astype(np.float32, copy=False), CLIP_MIN, CLIP_MAX)

def compute_global_mean_std(image_paths, k=5):
    """Mean/std over first k images after clipping, computed in a streaming way."""
    k = min(k, len(image_paths))
    if k == 0: raise RuntimeError("No images found to compute mean/std.")
    nvox = 0
    s = 0.0
    ss = 0.0
    for p in image_paths[:k]:
        a = nib.load(str(p)).get_fdata()
        a = clip(a)
        a = a.astype(np.float64, copy=False)
        n = a.size
        nvox += n
        s  += a.sum()
        ss += np.square(a).sum()
    mean = s / nvox
    var  = max(1e-12, ss / nvox - mean * mean)
    std  = float(np.sqrt(var))
    return float(mean), std, {"ref_files": [str(p) for p in image_paths[:k]], "n_vox": int(nvox)}

def pad_to_min(arr, min_shape, pad_value):
    """Pad [Z,Y,X] to at least min_shape with constant pad_value (centered)."""
    z,y,x = arr.shape
    mz,my,mx = map(int, min_shape)
    pz = max(0, mz - z); py = max(0, my - y); px = max(0, mx - x)
    if pz==py==px==0: return arr
    pb = (pz//2, py//2, px//2)
    pa = (pz-pb[0], py-pb[1], px-pb[2])
    return np.pad(arr,
                  ((pb[0], pa[0]), (pb[1], pa[1]), (pb[2], pa[2])),
                  mode="constant", constant_values=pad_value)

def choose_center(fg_mask, patch, p_foreground=0.7, min_fg_voxels=64):
    """Pick (cz,cy,cx); try to bias into foreground."""
    z,y,x = fg_mask.shape
    rz,ry,rx = (s//2 for s in patch)
    if np.random.rand() < p_foreground and fg_mask.sum() >= min_fg_voxels:
        idx = np.argwhere(fg_mask)
        cz,cy,cx = idx[np.random.randint(len(idx))]
        cz = int(np.clip(cz, rz, z-rz-1))
        cy = int(np.clip(cy, ry, y-ry-1))
        cx = int(np.clip(cx, rx, x-rx-1))
    else:
        def rnd(b, r): 
            lo, hi = r, b-r
            return int(b//2) if hi <= lo else np.random.randint(lo, hi+1)
        cz, cy, cx = rnd(z,rz), rnd(y,ry), rnd(x,rx)
    return cz,cy,cx

def crop3d(arr, center, size):
    cz,cy,cx = center
    sz,sy,sx = size
    rz,ry,rx = sz//2, sy//2, sx//2
    z0,z1 = cz - rz, cz + (sz - rz)
    y0,y1 = cy - ry, cy + (sy - ry)
    x0,x1 = cx - rx, cx + (sx - rx)
    return arr[z0:z1, y0:y1, x0:x1]

def save_img(out_path: Path, data_zyx: np.ndarray, ref_img_ni: nib.Nifti1Image):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(data_zyx.astype(np.float32, copy=False))
    ni  = nib.Nifti1Image(arr, ref_img_ni.affine)  # fresh header
    ni.header.set_data_dtype(np.float32)
    ni.header.set_xyzt_units(*ref_img_ni.header.get_xyzt_units())
    nib.save(ni, str(out_path))

def save_lbl(out_path: Path, label_zyx_int: np.ndarray, ref_img_ni: nib.Nifti1Image):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lab = np.ascontiguousarray(label_zyx_int.astype(np.int16, copy=False))
    ni  = nib.Nifti1Image(lab, ref_img_ni.affine)  # use image affine
    ni.header.set_data_dtype(np.int16)
    ni.header.set_xyzt_units(*ref_img_ni.header.get_xyzt_units())
    nib.save(ni, str(out_path))

def to_class_map(lbl: np.ndarray) -> np.ndarray:
    """Accept 3D int labels or 4D one-hot; return 3D int16 class map."""
    if lbl.ndim == 4:
        if lbl.shape[-1] == 1:
            lbl = lbl[..., 0]
        else:
            lbl = np.argmax(lbl, axis=-1)
    elif lbl.ndim != 3:
        raise RuntimeError(f"Label ndim must be 3 or 4, got {lbl.ndim}")
    return np.rint(lbl).astype(np.int16, copy=False)

def main():
    ap = argparse.ArgumentParser(description="Clip+normalize (images only) and create 128^3 patches.")
    ap.add_argument("--input_dir",  required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--patch", nargs=3, type=int, default=[128,128,128], metavar=("Z","Y","X"))
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patches_per_pair_train", type=int, default=6)
    ap.add_argument("--patches_per_pair_test",  type=int, default=2)
    ap.add_argument("--p_foreground", type=float, default=0.7)
    ap.add_argument("--min_fg_voxels", type=int, default=64)
    ap.add_argument("--norm_ref_n", type=int, default=5, help="How many images to estimate mean/std (after clipping).")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    pairs   = list_pairs(in_dir)
    if not pairs:
        print("No (image,label) pairs found."); return

    # --- global mean/std from first K clipped images ---
    img_paths = [ip for ip,_ in pairs]
    g_mean, g_std, meta = compute_global_mean_std(img_paths, k=args.norm_ref_n)
    print(f"[norm] global mean={g_mean:.6f}, std={g_std:.6f}  (from {len(meta['ref_files'])} files)")
    (out_dir / "_meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "_meta" / "normalization.json", "w") as f:
        json.dump({"clip":[CLIP_MIN,CLIP_MAX],"mean":g_mean,"std":g_std, **meta}, f, indent=2)

    # --- split pairs ---
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.train_ratio)
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]
    print(f"Pairs: {len(pairs)} (train {len(train_pairs)}, test {len(test_pairs)})")

    out_train, out_test = out_dir/"train", out_dir/"test"

    def work(split_pairs, out_root, n_patches_each):
        for ip, lp in split_pairs:
            img_ni = nib.load(str(ip))
            lbl_ni = nib.load(str(lp))

            img = img_ni.get_fdata()
            if img.ndim == 4 and img.shape[-1] == 1:
                img = img[..., 0]
            assert img.ndim == 3, f"image {ip} must be 3D, got {img.shape}"

            # image: clip then z-score with global stats
            img = clip(img)
            img = (img - g_mean) / g_std

            lbl = lbl_ni.get_fdata()
            lbl = to_class_map(lbl)  # 3D int16, no normalization

            # pad to allow cropping everywhere (pad image with mean=0 after z-score, labels with 0)
            img = pad_to_min(img, args.patch, pad_value=0.0)
            lbl = pad_to_min(lbl, args.patch, pad_value=0)

            fg = (lbl > 0)

            stem = ip.stem
            if stem.endswith(".nii"): stem = stem[:-4]

            for k in range(n_patches_each):
                cz,cy,cx = choose_center(fg, args.patch, args.p_foreground, args.min_fg_voxels)
                img_p = crop3d(img, (cz,cy,cx), args.patch)
                lbl_p = crop3d(lbl, (cz,cy,cx), args.patch)

                # hard sanity
                assert img_p.shape == tuple(args.patch), (img_p.shape, args.patch)
                assert lbl_p.shape == tuple(args.patch), (lbl_p.shape, args.patch)

                out_img = out_root / f"{stem}_patch{k:03d}_image.nii.gz"
                out_lbl = out_root / f"{stem}_patch{k:03d}_label.nii.gz"
                save_img(out_img, img_p, img_ni)
                save_lbl(out_lbl, lbl_p, img_ni)

    work(train_pairs, out_train, args.patches_per_pair_train)
    work(test_pairs,  out_test,  args.patches_per_pair_test)
    print("[done] patches written to", out_dir)

if __name__ == "__main__":
    main()
