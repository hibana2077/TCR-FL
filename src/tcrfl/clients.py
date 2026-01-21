from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import LocalTrainConfig
from .model import bce_logits_loss, clone_model, get_flat_params, set_flat_params_


@dataclass
class ClientResult:
    client_id: int
    delta: torch.Tensor  # flattened parameter delta
    num_samples: int


class FederatedClient:
    def __init__(self, client_id: int, local_cfg: LocalTrainConfig, device: torch.device):
        self.client_id = client_id
        self.local_cfg = local_cfg
        self.device = device

    def local_update(
        self,
        global_model: nn.Module,
        x: np.ndarray,
        y: np.ndarray,
    ) -> ClientResult:
        model = clone_model(global_model).to(self.device)
        model.train()

        x_t = torch.from_numpy(x).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        opt = torch.optim.SGD(
            model.parameters(),
            lr=self.local_cfg.lr,
            weight_decay=self.local_cfg.weight_decay,
        )

        n = x_t.shape[0]
        bs = min(int(self.local_cfg.batch_size), n)

        for _ in range(int(self.local_cfg.epochs)):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                batch = idx[start : start + bs]
                xb = x_t[batch]
                yb = y_t[batch]

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = bce_logits_loss(logits, yb)
                loss.backward()
                opt.step()

        global_flat = get_flat_params(global_model).to(self.device)
        local_flat = get_flat_params(model)
        delta = (local_flat - global_flat).detach().cpu()

        return ClientResult(client_id=self.client_id, delta=delta, num_samples=int(n))
