"""Thin TensorBoard logging wrapper."""

from __future__ import annotations

import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: int | None = None):
        self.writer.add_scalar(tag, value, step if step is not None else self.step)

    def log_scalars(self, prefix: str, mapping: dict, step: int | None = None):
        s = step if step is not None else self.step
        for k, v in mapping.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, s)

    def log_histogram(self, tag: str, values, step: int | None = None):
        self.writer.add_histogram(tag, values, step if step is not None else self.step)

    def close(self):
        self.writer.close()
