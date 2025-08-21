# losses/utils_pos_weight.py
import torch

@torch.no_grad()
def compute_pos_weight_for_multilabel(dataset, max_cases=None):
    """
    dataset[i] -> (image [1,D,H,W], target [3,D,H,W] where:
      ch0: kidney ∪ tumor ∪ cyst
      ch1: tumor ∪ cyst
      ch2: tumor
    Returns pos_weight tensor of shape [3] with (neg/pos) per channel, clipped to [1, 9].
    """
    pos = torch.zeros(3, dtype=torch.float64)
    neg = torch.zeros(3, dtype=torch.float64)
    n = len(dataset) if max_cases is None else min(max_cases, len(dataset))
    if n == 0:
        return torch.tensor([1., 2., 3.], dtype=torch.float32)
    for i in range(n):
        _, target = dataset[i]
        t = target.reshape(3, -1)
        pos += t.sum(dim=1)
        neg += (1.0 - t).sum(dim=1)
    pw = (neg / (pos + 1e-8)).clamp(1.0, 9.0)
    return pw.to(torch.float32)
