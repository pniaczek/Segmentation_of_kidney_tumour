# trainer_3_classes.py
import os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from evaluate import evaluate_model as run_full_eval

def _unwrap_for_saving(model):
    return model.module if hasattr(model, "module") else model

# -------- Loss pieces --------
def dice_mean_over_channels(sigmoid_logits, targets, channels=(0,1,2), eps=1e-6):
    p = sigmoid_logits[:, channels, ...]
    t = targets[:, channels, ...]
    inter = (p * t).sum(dim=(2,3,4))
    p_sum = p.sum(dim=(2,3,4))
    t_sum = t.sum(dim=(2,3,4))
    denom = p_sum + t_sum
    dice = (2*inter + eps) / (denom + eps)      # [B, C]
    present = (t_sum > 0)
    dice_sum = (dice * present).sum()
    denom_present = present.sum().clamp(min=1)
    return dice_sum / denom_present

def tversky_loss(sigmoid_logits, targets, channels=(1,2), alpha=0.7, beta=0.3, eps=1e-6):
    p = sigmoid_logits[:, channels, ...]
    t = targets[:,  channels, ...]
    tp = (p * t).sum(dim=(2,3,4))
    fp = (p * (1 - t)).sum(dim=(2,3,4))
    fn = ((1 - p) * t).sum(dim=(2,3,4))
    tversky = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return 1.0 - tversky.mean()

def composite_multilabel_loss(logits, targets, pos_weight, dice_w=1.0, tv_w=0.5, tv_alpha=0.7, tv_beta=0.3):
    """
    BCE(pos_weight) + Dice(all channels) + Tversky(channels 1–2).
    """
    # shape guard: pos_weight -> (1,C,1,1,1) for broadcast
    pw = pos_weight.view(1, -1, *([1] * (logits.ndim - 2)))
    bce = F.binary_cross_entropy_with_logits(
        logits[:, (0,1,2), ...],
        targets[:, (0,1,2), ...].float(),
        pos_weight=pw
    )
    p = torch.sigmoid(logits)
    dice_term = 1.0 - dice_mean_over_channels(p, targets, channels=(0,1,2))
    tv_term   = tversky_loss(p, targets, channels=(1,2), alpha=tv_alpha, beta=tv_beta)
    loss = bce + dice_w*dice_term + tv_w*tv_term
    # return scalar + nice logging dict
    return loss, {'bce': float(bce.detach()), 'dice': float(dice_term.detach()), 'tversky': float(tv_term.detach())}

# -------- Trainer --------
class Trainer:
    def __init__(self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        save_path,
        val_every=1,
        save_every=1,
        amp_dtype="bf16",
        accum_steps=1,
        is_main=True,
        train_sampler=None,
        pos_weight=None,
        scheduler=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.val_every = max(1, val_every)
        self.save_every = max(1, save_every)
        self.accum_steps = max(1, accum_steps)
        self.is_main = is_main
        self.train_sampler = train_sampler
        self.scheduler = scheduler

        if self.is_main:
            os.makedirs(self.save_path, exist_ok=True)
            self.metrics_dir = os.path.join(self.save_path, "metrics")
            os.makedirs(self.metrics_dir, exist_ok=True)
            self.metrics_csv = os.path.join(self.metrics_dir, "val_log.csv")
            if not os.path.exists(self.metrics_csv):
                with open(self.metrics_csv, "w", newline="") as f:
                    csv.writer(f).writerow(
                        ["epoch","train_loss","dice_fast_ch012","dice_official_mean","class0","class1","class2"]
                    )

        use_fp16 = (amp_dtype == "fp16")
        self.scaler = GradScaler(enabled=use_fp16)  # scaler only for fp16
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        self.pos_weight = (pos_weight.to(self.device)
                           if pos_weight is not None
                           else torch.tensor([1.,2.,3.], device=self.device))

        self.best_val_dice = 0.0
        self.active_channels = (0, 1, 2)
        self._last_train_loss = None

    def _to_device_3d(self, x):
        try:
            return x.to(self.device, non_blocking=True, memory_format=torch.channels_last_3d)
        except AttributeError:
            return x.to(self.device, non_blocking=True)

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        raise ValueError("Unexpected batch format. Expect (image, label).")

    def _save_checkpoint(self, epoch):
        if not self.is_main:
            return
        ckpt_path = os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pt")
        torch.save(_unwrap_for_saving(self.model).state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    def train_one_epoch(self, epoch):
        self.model.train()
        if isinstance(self.train_sampler, DistributedSampler):
            # important for proper shuffling with DDP
            self.train_sampler.set_epoch(epoch)

        running_loss, iter_count = 0.0, 0

        pbar = self.train_loader
        if self.is_main:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        self.optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            inputs, targets3 = self._unpack_batch(batch)
            inputs   = self._to_device_3d(inputs)
            targets3 = targets3.to(self.device, non_blocking=True).float()

            with autocast('cuda', dtype=self.amp_dtype):
                logits3 = self.model(inputs)
                loss, parts = composite_multilabel_loss(
                    logits3, targets3, pos_weight=self.pos_weight,
                    dice_w=1.0, tv_w=0.5, tv_alpha=0.7, tv_beta=0.3
                )

            loss = loss / self.accum_steps
            self.scaler.scale(loss).backward()

            if (iter_count + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * self.accum_steps
            iter_count += 1
            if self.is_main and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": running_loss / max(1, iter_count),
                    "bce":  parts["bce"],
                    "dice": parts["dice"],
                    "tv":   parts["tversky"],
                })

        avg_loss = running_loss / max(1, iter_count)
        if self.is_main:
            print(f"[Train] Epoch {epoch}: Loss = {avg_loss:.4f}")
        self._last_train_loss = float(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        if not self.is_main:
            return None
        self.model.eval()

        # quick dice
        running_dice = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val-fast]", leave=False)
        for img, tgt3 in pbar:
            img  = self._to_device_3d(img)
            tgt3 = tgt3.float().to(self.device, non_blocking=True)
            with autocast('cuda', dtype=self.amp_dtype):
                logits3 = self.model(img)
                dice = dice_mean_over_channels(torch.sigmoid(logits3), tgt3, channels=self.active_channels)
            running_dice += dice.item()
            pbar.set_postfix({"dice(0,1,2)": running_dice / (pbar.n + 1)})

        avg_dice_fast = running_dice / max(1, len(self.val_loader))
        print(f"[Val-fast] Epoch {epoch}: Dice(0,1,2) = {avg_dice_fast:.4f}")

        # full eval
        full = run_full_eval(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            amp_dtype=self.amp_dtype
        )
        off_mean = full["mean_dice"]
        per_class = full["per_class_mean"]

        print(f"[Val] Epoch {epoch}: Mean Dice = {off_mean:.4f} | Per-class: "
              + "[" + ", ".join(f"{d:.4f}" if d == d else "nan" for d in per_class[:3]) + "]")

        if off_mean > self.best_val_dice:
            self.best_val_dice = off_mean
            torch.save(_unwrap_for_saving(self.model).state_dict(), os.path.join(self.save_path, "best_model.pt"))
            print(f"Best model updated (mean Dice={off_mean:.4f})")

        # CSV
        with open(self.metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                self._last_train_loss if self._last_train_loss is not None else "",
                f"{avg_dice_fast:.6f}",
                f"{off_mean:.6f}",
                f"{per_class[0]:.6f}" if len(per_class) > 0 and per_class[0] == per_class[0] else "",
                f"{per_class[1]:.6f}" if len(per_class) > 1 and per_class[1] == per_class[1] else "",
                f"{per_class[2]:.6f}" if len(per_class) > 2 and per_class[2] == per_class[2] else "",
            ])
        return avg_dice_fast, off_mean, per_class

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.val_every == 0:
                self.validate(epoch)
            if self.scheduler is not None:
                # epoch-level step is fine for CosineAnnealingWarmRestarts
                self.scheduler.step(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    # compatibility shims
    def train_padded(self, num_epochs, pad_multiple=8):
        self.train(num_epochs)

    def validate_padded(self):
        return self.validate(0)
