# train/evaluate.py
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✨ use your functions
from losses.dice_loss import dice_mean_over_channels

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        img, mask = batch[0], batch[1]
        case_id = batch[2] if len(batch) > 2 else ["?"]
    else:
        img, mask = batch["image"], batch["mask"]
        case_id = batch.get("case_id", ["?"])
    if isinstance(case_id, (list, tuple)) and len(case_id) == 1:
        case_id = case_id[0]
    return img, mask, case_id


@torch.no_grad()
def evaluate_model(model, loader, device="cuda", amp_dtype=torch.bfloat16):
    """
    Uses dice_mean_over_channels(sigmoid(logits), target_onehot, channels)
    to compute:
      - mean Dice over channels (0,1,2)
      - per-class Dice by calling the same function with [0], [1], [2]
    Returns:
      {
        "mean_dice": float,
        "per_class_mean": [d0, d1, d2],
        "per_case": [{ "case_id": str, "dice": float, "per_class": [d0,d1,d2] }, ...]
      }
    """
    model.eval()

    scores = []
    per_class_all = []
    per_case = []

    from torch.amp import autocast
    for batch in loader:
        img, mask, case_id = _unpack_batch(batch)
        img  = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).float()  # one-hot expected

        with autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(img)
            # overall mean dice across channels 0,1,2 (bg+kidney+mass OR your active set)
            dice_all = dice_mean_over_channels(torch.sigmoid(logits), mask, channels=(0,1,2))

            # per-class via the same helper (single-channel)
            d0 = dice_mean_over_channels(torch.sigmoid(logits), mask, channels=[0])
            d1 = dice_mean_over_channels(torch.sigmoid(logits), mask, channels=[1])
            d2 = dice_mean_over_channels(torch.sigmoid(logits), mask, channels=[2])

        case_per_class = [float(d0.item()), float(d1.item()), float(d2.item())]
        scores.append(float(dice_all.item()))
        per_class_all.append(case_per_class)
        per_case.append({"case_id": str(case_id), "dice": float(dice_all.item()), "per_class": case_per_class})

    mean_dice = float(sum(scores) / max(1, len(scores)))

    if per_class_all:
        pc = torch.tensor(per_class_all, dtype=torch.float32)
        per_class_mean = pc.mean(dim=0).tolist()  # [d0,d1,d2]
    else:
        per_class_mean = [float("nan")] * 3

    return {
        "mean_dice": mean_dice,
        "per_class_mean": per_class_mean,
        "per_case": per_case,
    }


if __name__ == "__main__":
    # (optional CLI if you still want to run it standalone later)
    import argparse
    from models.unet3d import UNet3D
    from datasets.kidney_dataset import KidneyDataset
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_data_path", default="/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=["bf16","fp16"], default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = UNet3D(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    val_ds = KidneyDataset(args.val_data_path)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    out = evaluate_model(model, val_loader, device=device, amp_dtype=amp)
    print("Mean Dice:", out["mean_dice"])
    print("Per-class mean:", out["per_class_mean"])
