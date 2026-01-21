from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FlatParamsSpec:
    shapes: list[torch.Size]
    numels: list[int]

    @property
    def total(self) -> int:
        return int(sum(self.numels))


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat_params_(model: nn.Module, flat: torch.Tensor) -> None:
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset : offset + n].view_as(p))
            offset += n


def get_params_spec(model: nn.Module) -> FlatParamsSpec:
    shapes = [p.shape for p in model.parameters()]
    numels = [p.numel() for p in model.parameters()]
    return FlatParamsSpec(shapes=shapes, numels=numels)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).to(y.dtype)
    return float((preds == y).float().mean().item())


def bce_logits_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def clone_model(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)
