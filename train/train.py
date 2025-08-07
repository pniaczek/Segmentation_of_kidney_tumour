import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.kidney_dataset import KidneyDataset
from losses.class_weights import compute_class_weights_partial
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="multiclass", choices=["multiclass", "binary"],
                        help="Tryb treningu: multiclass (4 klasy) lub binary (1 wybrana klasa)")
    parser.add_argument("--foreground_class", type=int, default=1,
                        help="Numer klasy dla binarnego treningu (1=nerka, 2=guz, 3=torbiel)")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")
    if args.mode == "binary":
        print(f"Foreground class: {args.foreground_class}")

    # === MULTICLASS ===
    if args.mode == "multiclass":
        train_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augmented_data/train"
        val_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augmented_data/test"
        #train_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/train"
        #val_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/test"
        
        save_path = "trained_models/multiclass"
        os.makedirs(save_path, exist_ok=True)

        weights = compute_class_weights_partial(train_data_path, num_samples=391)
        train_dataset = KidneyDataset(train_data_path)
        val_dataset = KidneyDataset(val_data_path)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        model = UNet3D(in_channels=1, out_channels=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        active_class_schedule = {i: [1, 2] for i in range(10)}
        for i in range(10, 21):
            active_class_schedule[i] = [1, 2, 3]

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            active_class_schedule=active_class_schedule,
            save_path=save_path
        )
        trainer.train(num_epochs=args.epochs)

    # === BINARY – tylko klasa 1 → inne dane (cropowane) ===
    elif args.mode == "binary" and args.foreground_class == 1:
        from models.kidney_segmentation_crop import UNet3D as KidneyUNet3D
        data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/kidney_detection_processed"
        save_path = "trained_models/binary_class_1"
        os.makedirs(save_path, exist_ok=True)

        model = KidneyUNet3D().to(device)
        train_set = KidneyDataset(os.path.join(data_root, "train"))
        val_set = KidneyDataset(os.path.join(data_root, "test"))
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        trainer = Trainer(model, train_loader, val_loader, optimizer, device, save_path=save_path)
        trainer.train_binary_dice(num_epochs=args.epochs, foreground_class=1)

    # === BINARY – klasa 2 lub 3 → te same dane co w multiclass ===
    elif args.mode == "binary" and args.foreground_class in [2, 3]:
        data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset"
        save_path = f"trained_models/binary_class_{args.foreground_class}"
        os.makedirs(save_path, exist_ok=True)

        model = UNet3D(in_channels=1, out_channels=1).to(device)
        train_set = KidneyDataset(os.path.join(data_root, "train"))
        val_set = KidneyDataset(os.path.join(data_root, "test"))
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=2, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        trainer = Trainer(model, train_loader, val_loader, optimizer, device, save_path=save_path)
        trainer.train_binary_dice(num_epochs=args.epochs, foreground_class=args.foreground_class)

    else:
        raise ValueError(f"Nieobsługiwany wariant: mode={args.mode}, foreground_class={args.foreground_class}")


if __name__ == "__main__":
    main()
