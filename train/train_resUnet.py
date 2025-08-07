import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.residual_unet import Residual3DUNet as ResidualUNet3D
from losses.dice_loss import weighted_multiclass_dice_loss
from trainer import Trainer
from dataset.kits23_dataset import Kits23Dataset
from torch.optim.lr_scheduler import StepLR

# === ŚCIEŻKI I PARAMETRY ===
data_root = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_detection_processed"
save_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/models/resunet"
os.makedirs(save_path, exist_ok=True)

batch_size = 2
num_epochs = 50
learning_rate = 1e-4
num_workers = 4

# === DANE ===
train_dataset = Kits23Dataset(data_root, split="train")
val_dataset = Kits23Dataset(data_root, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

# === MODEL ===
model = ResidualUNet3D(in_channels=1, out_channels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === OPTIMIZER I SCHEDULER ===
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# === WAGI DLA KLAS (opcjonalnie) ===
weights = torch.tensor([0.01, 1.0, 1.0, 1.0])

# === TRAINER ===
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    loss_fn=weighted_multiclass_dice_loss,
    weights=weights,
    scheduler=scheduler,
    active_class_schedule={0: [1], 10: [1, 2], 20: [1, 2, 3]},
    save_path=save_path,
)

# === TRENING ===
trainer.train(num_epochs=num_epochs)
