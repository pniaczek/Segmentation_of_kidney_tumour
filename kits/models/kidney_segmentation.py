import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import importlib.util

# === Load official dice function ===
spec = importlib.util.spec_from_file_location("dice", "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/kits23/evaluation/dice.py")
dice_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dice_module)
official_dice = dice_module.dice

# === U-Net definition ===
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec1 = block(64, 32)
        self.out = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out(d1)

# === Dataset ===
class KidneyDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob(os.path.join(folder, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = torch.load(path)
        case_id = os.path.basename(path).replace(".pt", "")
        return data["image"], data["label"], case_id


# === Dice loss ===
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    if target.ndim == 4:
        target = target.unsqueeze(1)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# === Train function ===
def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for img, mask, _ in tqdm(train_loader, desc="Training"):
        img = img.float().to(device, non_blocking=True)
        mask = mask.float().to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(img)
        loss = dice_loss(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# === Evaluation ===
def evaluate(model, loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for img, mask, case_id in tqdm(loader, desc="Evaluating"):
            img = img.float().to(device, non_blocking=True)
            mask = mask.float().to(device, non_blocking=True)
            pred = model(img)
            pred = torch.sigmoid(pred)

            pred_np = (pred.cpu().numpy() > 0.5).astype(bool)
            mask_np = (mask.cpu().numpy() > 0.5).astype(bool)
            score = official_dice(pred_np, mask_np)

            print(f"Case {case_id[0]}: Dice Score = {score:.4f}")  # 🛠 wypisanie nazwy i wyniku
            dice_scores.append(score)

    return sum(dice_scores) / len(dice_scores)

# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_set = KidneyDataset("/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_detection_processed/train")
    test_set = KidneyDataset("/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/data/kidney_detection_processed/test")

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, pin_memory=True)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(1, 11):
        loss = train_model(train_loader, model, optimizer, device)
        print(f"[Epoch {epoch}] Dice Loss: {loss:.4f}")

    score = evaluate(model, test_loader, device)
    print(f"Test Dice Score: {score:.4f}")

    torch.save(model.state_dict(), "kidney_roi_model.pt")

if __name__ == "__main__":
    main()
