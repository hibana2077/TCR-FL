from __future__ import annotations

import torch

from .base import Aggregator


class CoordinateMedian(Aggregator):
    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(deltas, dim=0)
        return stacked.median(dim=0).values
