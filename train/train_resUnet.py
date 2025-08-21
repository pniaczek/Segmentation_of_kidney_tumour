#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.resUnet3d import Residual3DUNet as ResidualUNet3D
from losses.dice_loss import bce_dice_loss_on_channels, dice_mean_over_channels
from train.trainer import RegionTrainer  # <- region-based trainer (see trainer.py below)

# --- PATHS ---
data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset"
save_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/models/resunet_regions"

# Dataset that emits 3 region channels: [k+m, m, tumor]
from datasets.kidney_dataset import KidneyDataset

# --- HYPERPARAMS ---
batch_size   = 1
num_epochs   = 20
learning_rate = 1e-4
num_workers  = 4
pad_multiple = 8   # to match pooling multiples

# --- DATA ---
train_dataset = KidneyDataset(os.path.join(data_root, "train"), return_name=False)
val_dataset   = KidneyDataset(os.path.join(data_root, "test"),  return_name=False)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
)

# --- MODEL ---
# IMPORTANT: out_channels must match dataset: 3 region channels
NUM_CHANNELS = 3
model = ResidualUNet3D(in_channels=1, out_channels=NUM_CHANNELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- OPTIMIZER & SCHED ---
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# --- TRAINER ---
trainer = RegionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    loss_fn=bce_dice_loss_on_channels,   # sigmoid BCE+Dice over channels (0,1,2)
    scheduler=scheduler,
    save_path=save_path,
    val_every=1,
    save_every=1,
    amp_dtype="bf16",  # A100-friendly
    active_channels=(0, 1, 2),           # [k+m, m, tumor]
)

if __name__ == "__main__":
    os.makedirs(save_path, exist_ok=True)
    trainer.train_padded(num_epochs=num_epochs, pad_multiple=pad_multiple)
