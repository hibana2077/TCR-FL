from __future__ import annotations

import torch

from .base import Aggregator


class Krum(Aggregator):
    def __init__(self, f: int):
        self.f = int(f)

    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        # Krum selects a single update with minimal sum distances to closest (n-f-2) updates.
        stacked = torch.stack(deltas, dim=0)  # (n,d)
        n = stacked.shape[0]
        f = self.f
        m = n - f - 2
        if m <= 0:
            raise ValueError(f"Invalid f={f} for n={n} in Krum")

        # Pairwise squared distances
        # dist_ij = ||u_i - u_j||^2
        diff = stacked[:, None, :] - stacked[None, :, :]
        dist = torch.sum(diff * diff, dim=-1)

        scores = []
        for i in range(n):
            d = dist[i]
            d_sorted, _ = torch.sort(d)
            # include closest m distances excluding self distance at index 0
            score = torch.sum(d_sorted[1 : 1 + m])
            scores.append(score)
        scores_t = torch.stack(scores)
        idx = int(torch.argmin(scores_t).item())
        return stacked[idx]
