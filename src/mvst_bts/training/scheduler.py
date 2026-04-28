"""
scheduler.py — Cosine annealing with linear warmup.
"""
from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Optimizer, cfg, total_steps: int) -> LambdaLR:
    """
    Build a cosine decay schedule with linear warmup.

    Args:
        optimizer:    Base optimizer.
        cfg:          Full config (reads cfg.scheduler.warmup_ratio).
        total_steps:  Total number of training steps (epochs * steps_per_epoch).
    """
    warmup_steps = int(total_steps * cfg.scheduler.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup from 0 → 1
            return float(current_step) / max(1, warmup_steps)
        # Cosine decay from 1 → 0
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)