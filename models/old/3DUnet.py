import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import importlib.util

spec = importlib.util.spec_from_file_location("dice", "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/evaluation/dice.py")
dice_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dice_module)
official_dice = dice_module.dice

NUM_CLASSES = 4

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
        self.out = nn.Conv3d(32, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out(d1)

class KidneyDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob(os.path.join(folder, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = torch.load(path)
        image = data["image"]
        label = data["label"]
        label_onehot = F.one_hot(label.long(), NUM_CLASSES).permute(3, 0, 1, 2).float()
        return image, label_onehot, os.path.basename(path).replace(".pt", "")

def weighted_multiclass_dice_loss(pred, target, weights=None, smooth=1e-5, active_classes=[1, 2]):
    pred = F.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)

    dice = dice[:, active_classes]

    if weights is not None:
        weights = weights.to(pred.device)[active_classes]
        weights = weights.unsqueeze(0)
        loss = (1 - dice) * weights
    else:
        loss = 1 - dice

    return loss.mean()



def compute_class_weights_partial(data_folder, num_samples=391):
    class_counts = torch.zeros(NUM_CLASSES)
    file_paths = sorted(glob(os.path.join(data_folder, "*.pt")))[:num_samples]

    for path in tqdm(file_paths, desc=f"Counting voxels in {num_samples} samples"):
        label = torch.load(path, map_location="cpu")["label"]
        for cls in range(NUM_CLASSES):
            class_counts[cls] += (label == cls).sum()

    inv_freq = 1.0 / (class_counts + 1e-8)
    inv_freq[0] = 0.0
    weights = inv_freq / inv_freq.sum()
    
    print(f"[Partial] Class voxel counts: {class_counts.tolist()}")
    print(f"[Partial] Computed class weights: {weights.tolist()}")
    return weights


def dice_per_class(pred, target, num_classes=NUM_CLASSES, smooth=1e-5):
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(float)
        target_cls = (target == cls).astype(float)
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice)
    return dices


from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def train_model(train_loader, model, optimizer, device, weights, epoch):
    model.train()
    total_loss = 0

    if epoch < 10:
        active_classes = [1, 2]  # nerka, guz
    else:
        active_classes = [1, 2, 3]  # + torbiel

    for img, mask, _ in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            pred = model(img)
            loss = weighted_multiclass_dice_loss(pred, mask, weights, active_classes=active_classes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        print(f"Epoch {epoch}: training on classes {active_classes}")

    return total_loss / len(train_loader)




def evaluate(model, loader, device):
    model.eval()
    official_scores = []
    all_class_dices = []

    with torch.no_grad():
        for img, mask, case_id in tqdm(loader, desc="Evaluating"):
            img = img.to(device, non_blocking=True)
            mask = mask.float().to(device, non_blocking=True)

            pred = model(img)
            pred = F.softmax(pred, dim=1)
            pred_np = torch.argmax(pred, dim=1).cpu().numpy()
            mask_np = torch.argmax(mask, dim=1).cpu().numpy()

            official = official_dice(pred_np, mask_np)
            official_scores.append(official)

            per_class = dice_per_class(pred_np[0], mask_np[0])
            all_class_dices.append(per_class)

            per_class_str = " | ".join([f"Class {i}: {d:.4f}" for i, d in enumerate(per_class)])
            print(f"Case {case_id[0]}: Official Dice = {official:.4f} | Per-class Dice: {per_class_str}")

    mean_per_class = torch.tensor(all_class_dices).mean(dim=0).tolist()
    print("\n=== Mean Dice per Class ===")
    for i, d in enumerate(mean_per_class):
        print(f"Class {i}: {d:.4f}")

    return sum(official_scores) / len(official_scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/train"
    test_data_path = "/net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/dataset/test"

    weights = compute_class_weights_partial(train_data_path, num_samples=391)

    train_set = KidneyDataset(train_data_path)
    test_set = KidneyDataset(test_data_path)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, pin_memory=True)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(1, 21):
        loss = train_model(train_loader, model, optimizer, device, weights, epoch)
        print(f"[Epoch {epoch}] Dice Loss: {loss:.4f}")
        torch.cuda.empty_cache()

    score = evaluate(model, test_loader, device)
    print(f"Test Dice Score: {score:.4f}")
    torch.save(model.state_dict(), "trained_models/unet3d_multiclass_256x256x336.pt")


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()

