import torch
import torch.nn.functional as F


def dice_mean_over_channels(sigmoid_logits, targets, channels=(0,1,2), eps=1e-6):
    p = sigmoid_logits[:, channels, ...]
    t = targets[:, channels, ...]
    inter = (p * t).sum(dim=(2,3,4))
    p_sum = p.sum(dim=(2,3,4))
    t_sum = t.sum(dim=(2,3,4))
    denom = p_sum + t_sum
    dice = (2*inter + eps) / (denom + eps)      # [B, C]
    present = (t_sum > 0)                       # [B, C]
    dice_sum = (dice * present).sum()
    denom_present = present.sum().clamp(min=1)
    return dice_sum / denom_present



def bce_dice_loss_on_channels(
    logits,
    targets,
    channels=(1, 2),
    bce_w=1.0,
    dice_w=1.0,
    pos_weight=None,
):
    logits_sel = logits[:, channels, ...]
    targets_sel = targets[:, channels, ...].float()

    pw = None
    if pos_weight is not None:
        pw = pos_weight
        if not torch.is_tensor(pw):
            pw = torch.tensor(pw, dtype=logits.dtype, device=logits.device)
        else:
            pw = pw.to(device=logits.device, dtype=logits.dtype)

        # If given for ALL model channels, slice to selected channels
        if pw.numel() == logits.shape[1]:
            idx = torch.as_tensor(channels, device=logits.device)
            pw = pw.index_select(0, idx)

        # >>> IMPORTANT <<< reshape to (1, C, 1, 1, 1) so it broadcasts on channel dim
        view_shape = [1, pw.numel()] + [1] * (logits_sel.ndim - 2)  # e.g. (1,C,1,1,1)
        pw = pw.view(*view_shape)

    bce = F.binary_cross_entropy_with_logits(logits_sel, targets_sel, pos_weight=pw)
    dice = dice_mean_over_channels(torch.sigmoid(logits_sel), targets_sel,
                                   channels=range(logits_sel.shape[1]))
    return bce_w * bce + dice_w * (1.0 - dice)


def weighted_multiclass_dice_loss(pred, target, weights=None, smooth=1e-5, active_classes=None):
    """
    General multi-class Dice (softmax-based). Not used by default in the 3-class trainer,
    but kept here for reuse elsewhere.
    """
    if active_classes is None:
        active_classes = list(range(pred.shape[1]))
    pred = F.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice = dice[:, active_classes]

    if weights is not None:
        weights = weights.to(pred.device)[active_classes].unsqueeze(0)
        loss = (1 - dice) * weights
    else:
        loss = 1 - dice
    return loss.mean()

import torch
import torch.nn.functional as F

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
    bce = F.binary_cross_entropy_with_logits(
        logits[:, (0,1,2), ...],
        targets[:, (0,1,2), ...].float(),
        pos_weight=pos_weight.view(1, -1, *([1] * (logits.ndim - 2)))
    )
    p = torch.sigmoid(logits)
    # reuse your existing dice_mean_over_channels
    from .dice_loss import dice_mean_over_channels as _dice
    dice_term = 1.0 - _dice(p, targets, channels=(0,1,2))
    tv_term   = tversky_loss(p, targets, channels=(1,2), alpha=tv_alpha, beta=tv_beta)
    return bce + dice_w*dice_term + tv_w*tv_term, {'bce': bce.item(), 'dice': dice_term.item(), 'tversky': tv_term.item()}
