from __future__ import annotations

import argparse
from datetime import datetime

import torch

from tcrfl.config import AggregatorConfig, ExperimentConfig
from tcrfl.server import FederatedServer
from tcrfl.viz import plot_results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TCR-FL toy experiment (synthetic 2D obstacle classification)")
    p.add_argument("--agg", type=str, default="fedavg", help="fedavg|median|trimmed_mean|krum|tcr|tcr_trimmed_mean")
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tcr_beta", type=float, default=0.9)
    p.add_argument("--tcr_lambda", type=float, default=6.0)
    p.add_argument("--tcr_distance", type=str, default="l2", help="l2|cosine")

    p.add_argument("--trim_ratio", type=float, default=0.2)
    p.add_argument("--krum_f", type=int, default=2)

    p.add_argument("--task_switch", action="store_true")
    p.add_argument("--switch_every", type=int, default=15)

    p.add_argument("--drift_reset", action="store_true", help="Enable drift-aware reset for TCR")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    exp = ExperimentConfig()

    exp = ExperimentConfig(
        out_dir=args.out_dir,
        run_name=args.run_name or f"{args.agg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        fl=type(exp.fl)(rounds=args.rounds, global_lr=exp.fl.global_lr, seed=args.seed),
        data=type(exp.data)(
            **{**exp.data.__dict__},
            task_switch=type(exp.data.task_switch)(enabled=args.task_switch, switch_every=args.switch_every, seed=exp.data.task_switch.seed),
        ),
        local=exp.local,
        eval=exp.eval,
        agg=AggregatorConfig(
            name=args.agg,
            trim_ratio=args.trim_ratio,
            f=args.krum_f,
            tcr_beta=args.tcr_beta,
            tcr_lambda=args.tcr_lambda,
            tcr_distance=args.tcr_distance,
            drift_aware_reset=args.drift_reset,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server = FederatedServer(exp, device=device)
    artifacts = server.run()
    payload = server.summarize_and_save(artifacts)
    plot_results(payload, out_dir=f"{exp.out_dir}/{payload['run_name']}")


if __name__ == "__main__":
    main()
