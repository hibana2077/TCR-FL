from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .base import Aggregator
from .trimmed_mean import TrimmedMean


def _safe_cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    an = torch.linalg.norm(a) + eps
    bn = torch.linalg.norm(b) + eps
    cos = torch.dot(a, b) / (an * bn)
    return 1.0 - cos


@dataclass
class TemporalConsistencyReweighting:
    beta: float
    lam: float
    distance: str = "l2"  # l2|cosine
    drift_aware_reset: bool = False
    drift_reset_threshold: float = 2.5

    def __post_init__(self) -> None:
        self.beta = float(self.beta)
        self.lam = float(self.lam)
        self._ema: dict[int, torch.Tensor] = {}
        self._last_alphas: list[float] | None = None
        self._residual_history: list[float] = []

    def _residual(self, client_id: int, g: torch.Tensor) -> torch.Tensor:
        if client_id not in self._ema:
            return torch.tensor(0.0)
        base = self._ema[client_id]
        if self.distance == "cosine":
            return _safe_cosine_distance(g, base)
        return torch.linalg.norm(g - base)

    def compute_weights(self, client_ids: list[int], deltas: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = torch.stack([self._residual(cid, d) for cid, d in zip(client_ids, deltas)]).to(torch.float32)

        # Drift-aware reset: if many clients jump together, soften penalty.
        eff_lam = self.lam
        if self.drift_aware_reset and len(self._residual_history) >= 5:
            hist_med = float(np.median(self._residual_history[-20:]))
            cur_med = float(torch.median(residuals).item())
            if hist_med > 1e-9 and (cur_med / hist_med) >= self.drift_reset_threshold:
                eff_lam = 0.0

        w = torch.exp(-eff_lam * residuals)
        w_sum = torch.sum(w)
        if float(w_sum.item()) <= 0:
            alphas = torch.ones_like(w) / float(len(w))
        else:
            alphas = w / w_sum

        self._last_alphas = [float(a.item()) for a in alphas]
        self._residual_history.append(float(torch.median(residuals).item()))
        return w, alphas

    def update_ema(self, client_ids: list[int], deltas: list[torch.Tensor]) -> None:
        for cid, d in zip(client_ids, deltas):
            if cid not in self._ema:
                self._ema[cid] = d.detach().clone()
            else:
                self._ema[cid] = (self.beta * self._ema[cid] + (1.0 - self.beta) * d.detach()).clone()


class TCRFedAvg(Aggregator):
    def __init__(
        self,
        beta: float,
        lam: float,
        distance: str = "l2",
        drift_aware_reset: bool = False,
        drift_reset_threshold: float = 2.5,
    ):
        self.tcr = TemporalConsistencyReweighting(
            beta=beta,
            lam=lam,
            distance=distance,
            drift_aware_reset=drift_aware_reset,
            drift_reset_threshold=drift_reset_threshold,
        )
        self._client_ids: list[int] | None = None

    def set_client_ids(self, client_ids: list[int]) -> None:
        self._client_ids = list(client_ids)

    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        if self._client_ids is None:
            raise RuntimeError("TCRFedAvg requires client_ids; call set_client_ids()")
        _, alphas = self.tcr.compute_weights(self._client_ids, deltas)
        stacked = torch.stack(deltas, dim=0)
        agg = torch.sum(alphas[:, None] * stacked, dim=0)
        self.tcr.update_ema(self._client_ids, deltas)
        return agg

    def get_last_alphas(self) -> list[float] | None:
        return self.tcr._last_alphas


class TCRThenTrimmedMean(Aggregator):
    """Implements idea.md: pre-weight (delta <- w_i * delta) then robust aggregation."""

    def __init__(
        self,
        trim_ratio: float,
        beta: float,
        lam: float,
        distance: str = "l2",
        drift_aware_reset: bool = False,
        drift_reset_threshold: float = 2.5,
    ):
        self.tcr = TemporalConsistencyReweighting(
            beta=beta,
            lam=lam,
            distance=distance,
            drift_aware_reset=drift_aware_reset,
            drift_reset_threshold=drift_reset_threshold,
        )
        self.trimmed = TrimmedMean(trim_ratio=trim_ratio)
        self._client_ids: list[int] | None = None

    def set_client_ids(self, client_ids: list[int]) -> None:
        self._client_ids = list(client_ids)

    def aggregate(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        if self._client_ids is None:
            raise RuntimeError("TCRThenTrimmedMean requires client_ids; call set_client_ids()")
        w, _ = self.tcr.compute_weights(self._client_ids, deltas)
        scaled = [wi.item() * d for wi, d in zip(w, deltas)]
        agg = self.trimmed.aggregate(scaled)
        self.tcr.update_ema(self._client_ids, deltas)
        return agg

    def get_last_alphas(self) -> list[float] | None:
        return self.tcr._last_alphas
