import os
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from glob import glob
from tqdm import tqdm
import sys
sys.path.append("/net/tscratch/people/plgmpniak/KITS_project/Kits_23/")

from kits23.models.kidney_segmentation import UNet3D
from kits23.evaluation.dice import dice


# Ścieżki
MODEL_PATH = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/models/kidney_roi_model.pt"
INPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset"
OUTPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CT_MIN, CT_MAX = -54, 242
DETECTION_SHAPE = (128, 128, 128)
CROP_SIZE = (256, 256, 336)

def rescale_ct(img):
    img = np.clip(img, CT_MIN, CT_MAX)
    return 2 * (img - CT_MIN) / (CT_MAX - CT_MIN) - 1

def resize(img, shape, order=1):
    factors = [ns / float(osz) for ns, osz in zip(shape, img.shape)]
    return zoom(img, zoom=factors, order=order)

def get_bounding_box(mask):
    positions = np.argwhere(mask > 0)
    if positions.shape[0] == 0:
        return None
    minz, miny, minx = positions.min(axis=0)
    maxz, maxy, maxx = positions.max(axis=0)
    return slice(minz, maxz + 1), slice(miny, maxy + 1), slice(minx, maxx + 1)

def scale_bbox(bbox_small, from_shape, to_shape):
    scaled = []
    for slc, s_from, s_to in zip(bbox_small, from_shape, to_shape):
        scale = s_to / s_from
        start = int(round(slc.start * scale))
        stop = int(round(slc.stop * scale))
        scaled.append(slice(start, stop))
    return tuple(scaled)

def center_crop_around_bbox(img, bbox, crop_size):
    zc = (bbox[0].start + bbox[0].stop) // 2
    yc = (bbox[1].start + bbox[1].stop) // 2
    xc = (bbox[2].start + bbox[2].stop) // 2

    shape = img.shape
    half = [s // 2 for s in crop_size]

    def get_slice(center, half, max_size):
        start = max(center - half, 0)
        end = min(center + half, max_size)
        # Adjust if at the border
        if end - start < 2 * half:
            if start == 0:
                end = min(2 * half, max_size)
            elif end == max_size:
                start = max(0, max_size - 2 * half)
        return slice(start, end)

    return (
        get_slice(zc, half[0], shape[0]),
        get_slice(yc, half[1], shape[1]),
        get_slice(xc, half[2], shape[2])
    )

def crop_and_save(case_id, orig_img, orig_seg, bbox, output_dir):
    z, y, x = bbox
    cropped_img = orig_img[z, y, x]
    cropped_seg = orig_seg[z, y, x]

    target_shape = (256, 256, 336)
    def pad_to_shape(arr, shape):
        pad = []
        for a, s in zip(arr.shape, shape):
            total = max(0, s - a)
            before = total // 2
            after = total - before
            pad.append((before, after))
        return np.pad(arr, pad, mode='constant', constant_values=0)


    cropped_img = pad_to_shape(cropped_img, target_shape)
    cropped_seg = pad_to_shape(cropped_seg, target_shape)

    affine = np.eye(4)
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(cropped_img, affine), os.path.join(output_dir, f"{case_id}_cropped_image.nii.gz"))
    nib.save(nib.Nifti1Image(cropped_seg.astype(np.uint8), affine), os.path.join(output_dir, f"{case_id}_cropped_label.nii.gz"))


def main():
    model = UNet3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    cases = sorted(glob(os.path.join(INPUT_DIR, "case_*")))

    for case_path in tqdm(cases, desc="Cropping ROI@128 and extract 256³ patch"):
        case_id = os.path.basename(case_path)
        img_path = os.path.join(case_path, "imaging.nii.gz")
        seg_path = os.path.join(case_path, "segmentation.nii.gz")

        orig_img = nib.load(img_path).get_fdata()
        orig_seg = nib.load(seg_path).get_fdata()

        img_rescaled = rescale_ct(orig_img)
        img_down = resize(img_rescaled, DETECTION_SHAPE)
        img_tensor = torch.tensor(img_down[np.newaxis, np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = (torch.sigmoid(pred) > 0.5).cpu().numpy().astype(np.uint8)[0, 0]

        if np.sum(pred_mask) == 0:
            print(f"[Warning] No kidney detected in {case_id}. Skipping.")
            continue

        bbox_small = get_bounding_box(pred_mask)
        bbox_orig = scale_bbox(bbox_small, DETECTION_SHAPE, orig_img.shape)
        crop_box = center_crop_around_bbox(orig_img, bbox_orig, CROP_SIZE)
        crop_and_save(case_id, orig_img, orig_seg, crop_box, OUTPUT_DIR)

if __name__ == "__main__":
    main()
    print("Sprawdzam INPUT_DIR:", INPUT_DIR)
    cases = sorted([
    os.path.join(INPUT_DIR, d)
    for d in os.listdir(INPUT_DIR)
    if d.startswith("case_") and os.path.isdir(os.path.join(INPUT_DIR, d))
])

    print("Znaleziono przypadków:", len(cases))
    print("Przykład:", cases[:3])

