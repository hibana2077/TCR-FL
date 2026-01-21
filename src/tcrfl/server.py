from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from .aggregators import CoordinateMedian, FedAvg, Krum, TCRFedAvg, TCRThenTrimmedMean, TrimmedMean
from .config import AggregatorConfig, ExperimentConfig
from .clients import FederatedClient
from .metrics import asr_from_errors, evaluate
from .model import TinyMLP, get_flat_params, set_flat_params_
from .synthetic import SyntheticFLData
from .utils import ensure_dir, save_json, set_global_seed


@dataclass
class ExperimentArtifacts:
    accuracy: list[float]
    error: list[float]
    alphas: list[list[float] | None]


def build_aggregator(cfg: AggregatorConfig):
    name = cfg.name.lower()
    if name == "fedavg":
        return FedAvg()
    if name == "median":
        return CoordinateMedian()
    if name == "trimmed_mean":
        return TrimmedMean(trim_ratio=cfg.trim_ratio)
    if name == "krum":
        return Krum(f=cfg.f)
    if name == "tcr":
        return TCRFedAvg(
            beta=cfg.tcr_beta,
            lam=cfg.tcr_lambda,
            distance=cfg.tcr_distance,
            drift_aware_reset=cfg.drift_aware_reset,
            drift_reset_threshold=cfg.drift_reset_threshold,
        )
    if name == "tcr_trimmed_mean":
        return TCRThenTrimmedMean(
            trim_ratio=cfg.trim_ratio,
            beta=cfg.tcr_beta,
            lam=cfg.tcr_lambda,
            distance=cfg.tcr_distance,
            drift_aware_reset=cfg.drift_aware_reset,
            drift_reset_threshold=cfg.drift_reset_threshold,
        )
    raise ValueError(f"Unknown aggregator name: {cfg.name}")


class FederatedServer:
    def __init__(self, exp: ExperimentConfig, device: torch.device | None = None):
        self.exp = exp
        self.device = device or torch.device("cpu")

        set_global_seed(exp.fl.seed)

        self.data = SyntheticFLData(exp.data)
        self.model: nn.Module = TinyMLP().to(self.device)
        self.aggregator = build_aggregator(exp.agg)

        self.clients = [FederatedClient(i, exp.local, self.device) for i in range(exp.data.n_clients)]

        # Some aggregators need client ids for per-client state
        if hasattr(self.aggregator, "set_client_ids"):
            self.aggregator.set_client_ids(list(range(exp.data.n_clients)))

        self.out_dir = ensure_dir(exp.out_dir)

    def run(self) -> ExperimentArtifacts:
        acc_hist: list[float] = []
        err_hist: list[float] = []
        alpha_hist: list[list[float] | None] = []

        for t in trange(1, int(self.exp.fl.rounds) + 1, desc=f"FL({self.exp.agg.name})"):
            # Drift step (honest+malicious share the same movement model; key diff is label flip)
            self.data.step_client_means()

            deltas: list[torch.Tensor] = []
            for client in self.clients:
                x, y = self.data.sample_client_batch(client.client_id, t)
                res = client.local_update(self.model, x, y)
                deltas.append(res.delta)

            agg_delta = self.aggregator.aggregate(deltas)

            # Apply global update
            global_flat = get_flat_params(self.model).to(self.device)
            new_flat = global_flat + float(self.exp.fl.global_lr) * agg_delta.to(self.device)
            set_flat_params_(self.model, new_flat)

            # Eval on clean global test set (current task map)
            x_te, y_te = self.data.sample_global_test(t, self.exp.eval.test_size)
            acc, err = evaluate(self.model, x_te, y_te, self.device)
            acc_hist.append(float(acc))
            err_hist.append(float(err))

            alpha_hist.append(self.aggregator.get_last_alphas())

        return ExperimentArtifacts(accuracy=acc_hist, error=err_hist, alphas=alpha_hist)

    def summarize_and_save(self, artifacts: ExperimentArtifacts) -> dict:
        poison_every = int(self.exp.data.poison_every)
        asr = asr_from_errors(artifacts.error, poison_every=poison_every) if self.exp.eval.compute_asr else None

        run_name = self.exp.run_name or f"toy_{self.exp.agg.name}_T{self.exp.fl.rounds}"
        out = ensure_dir(f"{self.out_dir}/{run_name}")

        payload = {
            "run_name": run_name,
            "config": {
                "fl": self.exp.fl.__dict__,
                "local": self.exp.local.__dict__,
                "agg": self.exp.agg.__dict__,
                "data": {
                    **self.exp.data.__dict__,
                    "obstacle_map": self.exp.data.obstacle_map.__dict__,
                    "task_switch": self.exp.data.task_switch.__dict__,
                },
                "eval": self.exp.eval.__dict__,
            },
            "metrics": {
                "accuracy": artifacts.accuracy,
                "error": artifacts.error,
                "asr": asr,
            },
            "alphas": artifacts.alphas,
            "malicious_ids": self.data.malicious_ids,
        }

        save_json(f"{out}/results.json", payload)
        return payload
