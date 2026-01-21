from __future__ import annotations

import torch

from .base import Aggregator


class FedAvg(Aggregator):
    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(deltas, dim=0)
        return stacked.mean(dim=0)
