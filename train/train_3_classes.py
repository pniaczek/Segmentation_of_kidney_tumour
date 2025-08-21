# train_3classes.py
#!/usr/bin/env python3
import os, sys, argparse, datetime
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler, WeightedRandomSampler

# repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.kidney_dataset import KidneyDataset
from trainer_3_classes import Trainer
from losses.utils_pos_weight import compute_pos_weight_for_multilabel

def init_distributed_if_needed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=18000)
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        return True, local_rank, world_size
    return False, 0, 1

def build_dataloaders(train_path, val_path, batch_size, num_workers, train_limit, val_limit, distributed):
    train_ds = KidneyDataset(train_path)
    val_ds   = KidneyDataset(val_path)

    if train_limit and train_limit > 0:
        train_ds = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
    if val_limit and val_limit > 0:
        val_ds = Subset(val_ds, list(range(min(val_limit, len(val_ds)))))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        shuffle_train = False
    else:
        # Oversample cases with tumor/cyst; unwrap Subset for flags
        base_ds = train_ds
        indices = None
        if isinstance(train_ds, Subset):
            base_ds = train_ds.dataset
            indices = list(train_ds.indices)

        flags = getattr(base_ds, "has_mass_flags", None)
        if flags is not None:
            if indices is None:
                flags_t = torch.tensor(flags, dtype=torch.float32)
            else:
                flags_t = torch.tensor([flags[i] for i in indices], dtype=torch.float32)
            mass_factor = 4.0
            weights = torch.where(flags_t > 0.5,
                                  torch.full_like(flags_t, mass_factor),
                                  torch.ones_like(flags_t))
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle_train = False
        else:
            train_sampler = None
            shuffle_train = True

        val_sampler = None

    shuffle_val = False
    use_persist = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persist,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persist,
    )
    return train_loader, val_loader, train_sampler

def main():
    parser = argparse.ArgumentParser(description="Train 3-class UNet3D on NIfTI train/test folders.")
    # data
    parser.add_argument("--train_data_path", type=str, required=False,
                        default="/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_smaller_dataset_z/train")
    parser.add_argument("--val_data_path", type=str, required=False,
                        default="/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_smaller_dataset_z/test")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)

    # runtime
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--save_path", type=str, default="trained_models/three_classes")

    args = parser.parse_args()

    # Speed-ups (A100)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # TF32 matmuls

    # DDP setup
    is_dist, local_rank, world_size = init_distributed_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (not is_dist) or (torch.distributed.get_rank() == 0)
    if is_main:
        print(f"Using device: {device}; distributed={is_dist}, world_size={world_size}")
        os.makedirs(args.save_path, exist_ok=True)

    train_loader, val_loader, train_sampler = build_dataloaders(
        args.train_data_path, args.val_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        distributed=is_dist,
    )

    # unwrap to base dataset for pos_weight estimation
    base_train_ds = train_loader.dataset
    while isinstance(base_train_ds, Subset):
        base_train_ds = base_train_ds.dataset

    pw = compute_pos_weight_for_multilabel(base_train_ds, max_cases=128).to(device)
    if is_main:
        print("pos_weight[ch0,ch1,ch2] =", pw.tolist())

    # Model / Optimizer / Scheduler
    model = UNet3D(in_channels=1, out_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # Multi‑GPU wrap
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_path=args.save_path,
        val_every=args.val_every,
        save_every=args.save_every,
        amp_dtype=args.amp_dtype,
        accum_steps=args.accum_steps,
        is_main=is_main,
        train_sampler=train_sampler,
        pos_weight=pw,
        scheduler=scheduler,
    )

    trainer.train(num_epochs=args.epochs)

    if is_dist:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
