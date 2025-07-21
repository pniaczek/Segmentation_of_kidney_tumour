import os
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from glob import glob
from tqdm import tqdm

from kits23.models.unet_3d import UNet3D  # lub inna ścieżka
from kits23.evaluation.dice import dice

# Ścieżki
MODEL_PATH = "/ścieżka/do/model.pt"
INPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/dataset"
OUTPUT_DIR = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_cropped"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CT_MIN, CT_MAX = -54, 242
ROI_SHAPE = (128, 128, 128)


def rescale_ct(img):
    img = np.clip(img, CT_MIN, CT_MAX)
    return 2 * (img - CT_MIN) / (CT_MAX - CT_MIN) - 1


def resize(img, shape, order=1):
    factors = [ns / float(osz) for ns, osz in zip(shape, img.shape)]
    return zoom(img, zoom=factors, order=order)


def get_bounding_box(mask):
    positions = np.argwhere(mask > 0)
    minz, miny, minx = positions.min(axis=0)
    maxz, maxy, maxx = positions.max(axis=0)
    return slice(minz, maxz + 1), slice(miny, maxy + 1), slice(minx, maxx + 1)


def crop_and_save(case_id, orig_img, orig_seg, bbox, output_dir):
    z, y, x = bbox
    cropped_img = orig_img[z, y, x]
    cropped_seg = orig_seg[z, y, x]

    affine = np.eye(4)
    os.makedirs(output_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(cropped_img, affine), os.path.join(output_dir, f"{case_id}_cropped_image.nii.gz"))
    nib.save(nib.Nifti1Image(cropped_seg.astype(np.uint8), affine), os.path.join(output_dir, f"{case_id}_cropped_label.nii.gz"))


def main():
    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    cases = sorted(glob(os.path.join(INPUT_DIR, "case_*")))

    for case_path in tqdm(cases, desc="Cropping with ROI"):
        case_id = os.path.basename(case_path)
        img_path = os.path.join(case_path, "imaging.nii.gz")
        seg_path = os.path.join(case_path, "segmentation.nii.gz")

        orig_img = nib.load(img_path).get_fdata()
        orig_seg = nib.load(seg_path).get_fdata()

        img_rescaled = rescale_ct(orig_img)
        img_down = resize(img_rescaled, ROI_SHAPE)
        img_tensor = torch.tensor(img_down[np.newaxis, np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = (torch.sigmoid(pred) > 0.5).cpu().numpy().astype(np.uint8)[0, 0]

        # Upsample predykcji do rozdzielczości oryginalnej
        pred_upsampled = resize(pred_mask, orig_img.shape, order=0)

        if np.sum(pred_upsampled) == 0:
            print(f"[Warning] No kidney detected in {case_id}. Skipping.")
            continue

        bbox = get_bounding_box(pred_upsampled)
        crop_and_save(case_id, orig_img, orig_seg, bbox, OUTPUT_DIR)


if __name__ == "__main__":
    main()
