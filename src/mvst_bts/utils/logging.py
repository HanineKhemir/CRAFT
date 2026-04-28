"""
logging.py — structured logger and TensorBoard writer helpers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, log_file: Optional[str | Path] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.

    Usage:
        log = get_logger(__name__, log_file="experiments/runs/my_run/train.log")
        log.info("Epoch 1 started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TBLogger:
    """
    Thin wrapper around SummaryWriter for clean metric logging.

    Usage:
        tb = TBLogger("experiments/runs/my_run/tb")
        tb.log_metrics({"loss": 0.4, "sensitivity": 68.5}, step=100, prefix="train")
    """

    def __init__(self, log_dir: str | Path):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, global_step=step)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self.writer.close()