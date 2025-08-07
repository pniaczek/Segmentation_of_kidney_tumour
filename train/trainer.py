import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from losses.dice_loss import weighted_multiclass_dice_loss, dice_per_class, adjusted_binary_dice
from kits23.evaluation.dice import dice as official_dice

import time  # u góry pliku, jeśli nie masz



class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        loss_fn=weighted_multiclass_dice_loss,
        weights=None,
        scheduler=None,
        active_class_schedule=None,
        save_path=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.weights = weights
        self.scheduler = scheduler
        self.active_class_schedule = active_class_schedule or {}
        self.save_path = save_path
        self.scaler = GradScaler()
        self.epoch = 0

    def get_active_classes(self):
        return self.active_class_schedule.get(self.epoch, list(range(self.model.out.out_channels)))

    def train_one_epoch(self):
        start_time = time.time()
        self.model.train()
        total_loss = 0.0
        active_classes = self.get_active_classes()

        for img, mask, *_ in tqdm(self.train_loader, desc=f"Training epoch {self.epoch}"):
            img = img.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                pred = self.model(img)
                loss = self.loss_fn(pred, mask, weights=self.weights, active_classes=active_classes)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        # === TU ===
        torch.cuda.synchronize()
        duration = time.time() - start_time
        print(f"[GPU] Max memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"[Time] Epoch duration: {duration:.2f} seconds")

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        official_scores = []
        all_class_dices = []

        with torch.no_grad():
            for img, mask, *_ in tqdm(self.val_loader, desc="Validating"):
                img = img.to(self.device)
                mask = mask.float().to(self.device)

                pred = self.model(img)
                pred_soft = torch.softmax(pred, dim=1)
                pred_np = torch.argmax(pred_soft, dim=1).cpu().numpy()
                mask_np = torch.argmax(mask, dim=1).cpu().numpy()

                official = official_dice(pred_np, mask_np)
                per_class = dice_per_class(pred_np[0], mask_np[0])
                official_scores.append(official)
                all_class_dices.append(per_class)

        mean_official = sum(official_scores) / len(official_scores)
        mean_per_class = torch.tensor(all_class_dices).mean(dim=0).tolist()

        print(f"\nValidation Dice: {mean_official:.4f}")
        for i, d in enumerate(mean_per_class):
            print(f"Class {i}: {d:.4f}")

        return mean_official


    def validate_binary(self, foreground_class=1):
        self.model.eval()
        dice_scores = []
        adjusted_scores = []

        with torch.no_grad():
            for img, mask, *_ in tqdm(self.val_loader, desc=f"Validating (class {foreground_class})"):
                img = img.to(self.device)
                mask_bin = (mask == foreground_class).float().unsqueeze(1).to(self.device)

                pred = self.model(img)
                pred = torch.sigmoid(pred)
                pred_bin = (pred > 0.5).float()

                # Klasyczny Dice
                intersection = (pred_bin * mask_bin).sum()
                union = pred_bin.sum() + mask_bin.sum()
                dice = (2 * intersection + 1e-5) / (union + 1e-5)
                dice_scores.append(dice.item())

                # Adjusted Dice
                adj_dice = adjusted_binary_dice(pred_bin, mask_bin).mean().item()
                adjusted_scores.append(adj_dice)

        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_adjusted = sum(adjusted_scores) / len(adjusted_scores)

        print(f"Validation Binary Dice (class {foreground_class}): {mean_dice:.4f}")
        print(f"Validation Adjusted Dice (class {foreground_class}): {mean_adjusted:.4f}")

        return mean_dice, mean_adjusted




    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

            if self.val_loader:
                self.validate()

            if epoch == 9:
                print("\n[Evaluation before adding class 3 in epoch 10]")
                self.validate()

            if self.save_path:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, f"epoch_{epoch}.pt")
                )

    def train_binary_dice(self, num_epochs, foreground_class=1):
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0

            for img, mask, *_ in tqdm(self.train_loader, desc=f"Binary Training epoch {epoch}"):
                img = img.to(self.device)
                mask_bin = (mask == foreground_class).float().unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    pred = self.model(img)
                    pred = torch.sigmoid(pred)
                    loss = 1 - adjusted_binary_dice(pred, mask_bin).mean()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch}] Adjusted Binary Dice Loss (class {foreground_class}): {avg_loss:.4f}")

            if self.val_loader:
                dice, adj_dice = self.validate_binary(foreground_class=foreground_class)
                print(f"[Epoch {epoch}] Binary Dice: {dice:.4f} | Adjusted Dice: {adj_dice:.4f}")

            if self.save_path:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, f"binary_class{foreground_class}_epoch_{epoch}.pt")
                )