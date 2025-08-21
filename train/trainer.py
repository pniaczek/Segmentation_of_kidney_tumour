import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from losses.dice_loss import bce_dice_loss_on_channels, dice_mean_over_channels


class RegionTrainer:
    """
    Region-based, multi-label trainer (3 channels):
      ch0: kidney_and_masses (kidney ∪ tumor ∪ cyst)
      ch1: masses            (tumor ∪ cyst)
      ch2: tumor

    Model head: out_channels=3 (logits), use sigmoid in metrics.
    Loss: BCE-with-logits + Dice over the active channels (default: 0,1,2).
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        loss_fn=bce_dice_loss_on_channels,
        scheduler=None,
        save_path=None,
        val_every: int = 1,
        save_every: int = 1,
        amp_dtype: str = "bf16",   # "bf16" (A100) or "fp16"
        active_channels=(0, 1, 2),
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_path = save_path
        self.val_every = max(1, int(val_every))
        self.save_every = max(1, int(save_every))
        self.epoch = 0

        self._amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.scaler = GradScaler('cuda', enabled=True)

        # region channels to optimize/evaluate
        self.active_channels = tuple(active_channels)

        self.pos_weight = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=torch.float32)

        # best score
        self.best_val_dice = -1.0

    # --- helpers for padding to multiples of N (e.g., 8 for 3 downsamplings) ---
    def _pad_to_multiple(self, x, multiple=8):
        _, _, D, H, W = x.shape
        pd = (multiple - D % multiple) % multiple
        ph = (multiple - H % multiple) % multiple
        pw = (multiple - W % multiple) % multiple
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x, (D, H, W)

    def _crop_to_size(self, y, size_dhw):
        D, H, W = size_dhw
        return y[..., :D, :H, :W]

    def _to_device_3d(self, x):
        try:
            return x.to(self.device, non_blocking=True, memory_format=torch.channels_last_3d)
        except AttributeError:
            return x.to(self.device, non_blocking=True)

    def _unpack_batch(self, batch):
        # Expect (image, target3) or dict-like with keys
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            return batch["image"], batch["mask"]
        else:
            raise ValueError("Unexpected batch format.")

    def train_one_epoch_padded(self, multiple=8):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch} [train]", leave=False):
            inputs, targets3 = self._unpack_batch(batch)  # targets3: [B,3,D,H,W] (multi-label regions)
            inputs   = self._to_device_3d(inputs)
            targets3 = targets3.to(self.device, non_blocking=True).float()

            inputs_pad, orig_size = self._pad_to_multiple(inputs, multiple=multiple)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=self._amp_dtype):
                logits_pad = self.model(inputs_pad)  # [B,3,D,H,W]
                logits3 = self._crop_to_size(logits_pad, orig_size)

                loss = self.loss_fn(
                    logits3, targets3,
                    channels=self.active_channels,
                    bce_w=1.0, dice_w=1.0,
                    pos_weight=self.pos_weight, 
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())

        return running_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate_padded(self, multiple=8):
        torch.cuda.empty_cache()
        self.model.eval()

        running_dice = 0.0
        n = 0

        for batch in tqdm(self.val_loader, desc="Validating (padded)", leave=False):
            inputs, targets3 = self._unpack_batch(batch)
            inputs   = self._to_device_3d(inputs)
            targets3 = targets3.to(self.device, non_blocking=True).float()

            inputs_pad, orig_size = self._pad_to_multiple(inputs, multiple=multiple)
            with autocast('cuda', dtype=self._amp_dtype):
                logits_pad = self.model(inputs_pad)
                logits3 = self._crop_to_size(logits_pad, orig_size)

                dice = dice_mean_over_channels(
                    torch.sigmoid(logits3), targets3, channels=self.active_channels
                )
            running_dice += float(dice.item())
            n += 1

        mean_dice = running_dice / max(1, n)
        print(f"\nValidation (regions) Dice mean over {self.active_channels}: {mean_dice:.4f}")
        return mean_dice

    def _save_checkpoint(self, tag):
        if not self.save_path:
            return
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f"{tag}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")

    def train_padded(self, num_epochs, pad_multiple=8):
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch_padded(multiple=pad_multiple)
            print(f"[Epoch {epoch}] Train Loss (padded): {train_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

            val_score = None
            if self.val_loader and (epoch % self.val_every == 0 or epoch == num_epochs - 1):
                val_score = self.validate_padded(multiple=pad_multiple)

            if epoch % self.save_every == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(f"epoch_{epoch}")

            if (val_score is not None) and (val_score > self.best_val_dice):
                self.best_val_dice = val_score
                self._save_checkpoint("best")
                print(f"[Epoch {epoch}] New best model saved (Dice={self.best_val_dice:.4f})")
