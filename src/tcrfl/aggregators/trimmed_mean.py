from __future__ import annotations

import torch

from .base import Aggregator


class TrimmedMean(Aggregator):
    def __init__(self, trim_ratio: float):
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = float(trim_ratio)

    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(deltas, dim=0)  # (n,d)
        n = stacked.shape[0]
        k = int(self.trim_ratio * n)
        if k == 0:
            return stacked.mean(dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[k : n - k, :]
        return trimmed.mean(dim=0)
