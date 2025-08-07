import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.kidney_dataset import KidneyDataset
from evaluation.dice import official_dice
from losses.dice import dice_per_class

def evaluate_model(model, loader, mode="multiclass", foreground_class=1, device="cuda"):
    model.eval()
    scores = []
    per_class_dices = []

    with torch.no_grad():
        for img, mask, case_id in loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

            if mode == "binary":
                # mask: [B, D, H, W] – zamieniamy na binarną
                mask_bin = (mask == foreground_class).float().unsqueeze(1)
                pred = torch.sigmoid(pred)
                pred_bin = (pred > 0.5).float()
                intersection = (pred_bin * mask_bin.to(device)).sum()
                union = pred_bin.sum() + mask_bin.to(device).sum()
                dice = (2 * intersection + 1e-5) / (union + 1e-5)
                scores.append(dice.item())
                print(f"Case {case_id[0]} | Binary Dice (class {foreground_class}): {dice.item():.4f}")

            else:  # multiclass
                pred_soft = torch.softmax(pred, dim=1)
                pred_np = torch.argmax(pred_soft, dim=1).cpu().numpy()
                mask_np = torch.argmax(mask, dim=1).cpu().numpy()
                score = official_dice(pred_np, mask_np)
                scores.append(score)

                per_class = dice_per_class(pred_np[0], mask_np[0])
                per_class_dices.append(per_class)

                print(f"Case {case_id[0]} | Dice: {score:.4f} | Per-class: {['%.4f' % d for d in per_class]}")

    mean_score = sum(scores) / len(scores)
    print("\n=== Evaluation Summary ===")
    print(f"Mean Dice: {mean_score:.4f}")

    if mode == "multiclass":
        mean_class_scores = torch.tensor(per_class_dices).mean(dim=0).tolist()
        for i, d in enumerate(mean_class_scores):
            print(f"Class {i}: {d:.4f}")

    return mean_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="multiclass", choices=["multiclass", "binary"])
    parser.add_argument("--foreground_class", type=int, default=1)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "multiclass":
        from models.unet3d import UNet3D
        model = UNet3D(in_channels=1, out_channels=4).to(device)
        data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/test"

    elif args.mode == "binary" and args.foreground_class == 1:
        from models.segmentation_crop import UNet3D
        model = UNet3D().to(device)
        data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_detection_processed/test"

    elif args.mode == "binary" and args.foreground_class in [2, 3]:
        from models.unet3d import UNet3D
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/test"

    else:
        raise ValueError("Nieobsługiwany tryb lub foreground_class")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    dataset = KidneyDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)

    evaluate_model(model, loader, mode=args.mode, foreground_class=args.foreground_class, device=device)
