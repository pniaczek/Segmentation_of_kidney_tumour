#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset

# so we can import from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.kidney_dataset import KidneyDataset
from trainer import Trainer


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    train_limit: int,
    val_limit: int,
):
    # datasets
    train_ds = KidneyDataset(train_path)
    val_ds   = KidneyDataset(val_path)

    # optional limits (for fitting within walltime)
    if train_limit and train_limit > 0:
        train_ds = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
    if val_limit and val_limit > 0:
        val_ds = Subset(val_ds, list(range(min(val_limit, len(val_ds)))))

    # dataloaders
    pw = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,               # keep 1 for validation stability
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=pw,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="multiclass", choices=["multiclass", "binary"],
                        help="Training mode: multiclass (4 classes) or binary (one foreground class).")
    parser.add_argument("--foreground_class", type=int, default=1,
                        help="Foreground class for binary mode (1=kidney, 2=tumor, 3=cyst).")
    parser.add_argument("--epochs", type=int, default=20)

    # performance / walltime knobs
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_limit", type=int, default=0, help="0 = full train set")
    parser.add_argument("--val_limit", type=int, default=48, help="validate on a subset")
    parser.add_argument("--val_every", type=int, default=6, help="run validation every N epochs")
    parser.add_argument("--save_every", type=int, default=6, help="save checkpoint every N epochs")
    parser.add_argument("--use_padded", action="store_true",
                        help="Use pad/crop forward (train_padded/validate_padded).")

    args = parser.parse_args()

    # Speed-ups (A100)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # enables TF32 fast matmuls

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    if args.mode == "binary":
        print(f"Foreground class: {args.foreground_class}")

    if args.mode == "multiclass":
        # paths
        train_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augmented_data/train"
        val_data_path   = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augmented_data/test"

        # model dir
        save_path = "trained_models/multiclass"
        os.makedirs(save_path, exist_ok=True)

        # fixed class weights (do NOT compute at runtime)
        weights = [0.0, 0.01643759198486805, 0.05980661138892174, 0.9237557649612427]

        # data
        train_loader, val_loader = build_dataloaders(
            train_data_path, val_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
        )

        # model/opt
        model = UNet3D(in_channels=1, out_channels=4).to(device).to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        # class schedule: epochs 0–9 -> [1,2], epochs 10.. -> [1,2,3]
        active_class_schedule = {i: [1, 2] for i in range(min(10, args.epochs))}
        for i in range(10, args.epochs + 1):
            active_class_schedule[i] = [1, 2, 3]

        print("[Trainer] Using provided class weights; dataset scan disabled.")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            active_class_schedule=active_class_schedule,
            save_path=save_path,
            # these two require tiny additions to your Trainer (store + gate usage)
            val_every=args.val_every,
            save_every=args.save_every,
        )

        if args.use_padded:
            trainer.train_padded(num_epochs=args.epochs, pad_multiple=8)
        else:
            trainer.train(num_epochs=args.epochs)

    elif args.mode == "binary" and args.foreground_class == 1:
        # Binary for class 1 on cropped data
        from models.kidney_segmentation_crop import UNet3D as KidneyUNet3D

        data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/kidney_detection_processed"
        save_path = "trained_models/binary_class_1"
        os.makedirs(save_path, exist_ok=True)

        train_loader, val_loader = build_dataloaders(
            os.path.join(data_root, "train"),
            os.path.join(data_root, "test"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
        )

        model = KidneyUNet3D().to(device).to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        trainer = Trainer(
            model, train_loader, val_loader, optimizer, device,
            save_path=save_path,
            val_every=args.val_every,
            save_every=args.save_every,
        )
        if args.use_padded:
            # binary path uses 1-channel output; your padded helpers work fine on shapes
            trainer.train_padded(num_epochs=args.epochs, pad_multiple=8)
        else:
            trainer.train_binary_dice(num_epochs=args.epochs, foreground_class=1)

    elif args.mode == "binary" and args.foreground_class in [2, 3]:
        # Binary for class 2 or 3 on the multiclass dataset
        data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset"
        save_path = f"trained_models/binary_class_{args.foreground_class}"
        os.makedirs(save_path, exist_ok=True)

        train_loader, val_loader = build_dataloaders(
            os.path.join(data_root, "train"),
            os.path.join(data_root, "test"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
        )

        model = UNet3D(in_channels=1, out_channels=1).to(device).to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        trainer = Trainer(
            model, train_loader, val_loader, optimizer, device,
            save_path=save_path,
            val_every=args.val_every,
            save_every=args.save_every,
        )
        if args.use_padded:
            trainer.train_padded(num_epochs=args.epochs, pad_multiple=8)
        else:
            trainer.train_binary_dice(num_epochs=args.epochs, foreground_class=args.foreground_class)

    else:
        raise ValueError(f"Unsupported combo: mode={args.mode}, foreground_class={args.foreground_class}")


if __name__ == "__main__":
    main()
