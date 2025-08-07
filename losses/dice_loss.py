import numpy as np
import torch
import torch.nn.functional as F

def weighted_multiclass_dice_loss(pred, target, weights=None, smooth=1e-5, active_classes=None):
    if active_classes is None:
        active_classes = list(range(pred.shape[1]))

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

def dice_per_class(pred, target, num_classes=4, smooth=1e-5):
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(float)
        target_cls = (target == cls).astype(float)
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice)
    return dices

def adjusted_binary_dice(pred, target, epsilon=1e-5):
    """
    Liczy Dice tylko jeśli jest co porównywać.
    W przypadku braku klasy w GT i predykcji => Dice = 0 zamiast 1.
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3, 4))
    pred_sum = pred.sum(dim=(2, 3, 4))
    target_sum = target.sum(dim=(2, 3, 4))
    union = pred_sum + target_sum

    dice = torch.where(union > 0,
                       (2.0 * intersection + epsilon) / (union + epsilon),
                       torch.tensor(0.0, device=pred.device))

    return dice  # [B]

