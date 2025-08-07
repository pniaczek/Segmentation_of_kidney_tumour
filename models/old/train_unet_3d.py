import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random

from unet_3d import UNet3D
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-5, include_background=False):
    """
    pred: tensor of shape [B, C, D, H, W] (logits)
    target: tensor of shape [B, D, H, W] (ints: 0-3)
    """
    num_classes = pred.shape[1]

    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    pred_soft = F.softmax(pred, dim=1)

    dims = (0, 2, 3, 4)
    intersection = torch.sum(pred_soft * target_onehot, dim=dims)
    denominator = torch.sum(pred_soft, dim=dims) + torch.sum(target_onehot, dim=dims)

    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)

    if not include_background:
        dice_per_class = dice_per_class[1:]

    return 1.0 - dice_per_class.mean()


# ----- Dataset -----
class KiTSPreprocessedDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob(os.path.join(folder, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx])
        return item['image'], item['label']

# ----- Train Loop -----
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = dice_loss(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 5 == 0:
            print(f"[Batch {i+1}] Dice Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# ----- Main -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = KiTSPreprocessedDataset("/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/preprocessed")

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    model = UNet3D(in_channels=1, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        avg_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Dice Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")

if __name__ == "__main__":
    main()
