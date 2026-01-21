from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .model import accuracy_from_logits


@dataclass
class RoundMetrics:
    round: int
    accuracy: float
    error: float


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> RoundMetrics:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        logits = model(xt)
        acc = accuracy_from_logits(logits, yt)
        err = 1.0 - acc
    return acc, err


def asr_from_errors(round_errors: list[float], poison_every: int) -> float:
    poisoned = [e for t, e in enumerate(round_errors, start=1) if (t % poison_every == 0)]
    clean = [e for t, e in enumerate(round_errors, start=1) if (t % poison_every != 0)]
    if len(poisoned) == 0 or len(clean) == 0:
        return float("nan")
    return float(np.mean(poisoned) - np.mean(clean))
