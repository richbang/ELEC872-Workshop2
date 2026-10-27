"""Additions to train.py for Workshop 2: Part E (learning-rate warm-up) and Part F (Focal Loss)."""
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017):

        FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)

    p_t is the softmax probability of the true class. Windows the model already gets right
    (p_t -> 1; in OPPORTUNITY mostly Null) are scaled down by (1 - p_t)^gamma, so the gradient
    concentrates on hard, usually rare, gestures. gamma = 0 gives plain cross-entropy.
    `weight` (optional, one value per class) is the alpha_t term of the paper."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()) ** self.gamma * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def warmup_factor(step, warmup_steps):
    """Linear warm-up: LR multiplier for update `step` (0-based), rising from 1/warmup_steps to 1."""
    return min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
