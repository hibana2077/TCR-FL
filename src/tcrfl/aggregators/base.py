from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        """Return aggregated delta (flattened)."""

    def on_round_end(self) -> None:
        return

    def get_last_alphas(self) -> list[float] | None:
        return None
