from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


DistanceType = Literal["l2", "cosine"]
TrajectoryType = Literal["velocity", "ellipse"]


@dataclass(frozen=True)
class ObstacleMapConfig:
    # Default per docs/dataset_syn.md
    centers: list[tuple[float, float]] = ((0.3, 0.3), (-0.4, 0.2), (0.0, -0.5))
    radii: list[float] = (0.25, 0.20, 0.30)


@dataclass(frozen=True)
class TaskSwitchConfig:
    enabled: bool = False
    switch_every: int = 15
    seed: int = 123


@dataclass(frozen=True)
class SyntheticDataConfig:
    domain_min: float = -1.0
    domain_max: float = 1.0

    n_clients: int = 10
    n_honest: int = 8
    n_malicious: int = 2

    samples_per_client_per_round: int = 200
    uniform_mix: float = 0.05  # rho

    # Drift / trajectory
    trajectory: TrajectoryType = "velocity"
    v_min: float = 0.005
    v_max: float = 0.02
    sigma_m: float = 0.01

    # Gaussian shape
    sigma1_min: float = 0.05
    sigma1_max: float = 0.12
    sigma2_min: float = 0.05
    sigma2_max: float = 0.12

    obstacle_map: ObstacleMapConfig = ObstacleMapConfig()
    task_switch: TaskSwitchConfig = TaskSwitchConfig()

    # Poisoning
    poison_every: int = 3  # flip label when (t % poison_every == 0)


@dataclass(frozen=True)
class LocalTrainConfig:
    lr: float = 0.1
    epochs: int = 1
    batch_size: int = 64
    weight_decay: float = 0.0


@dataclass(frozen=True)
class FLConfig:
    rounds: int = 60
    global_lr: float = 1.0  # server applies aggregated delta with this step size
    seed: int = 42


@dataclass(frozen=True)
class AggregatorConfig:
    name: str = "fedavg"  # fedavg|median|trimmed_mean|krum|tcr|tcr_trimmed_mean

    # For trimmed mean
    trim_ratio: float = 0.2

    # For krum
    f: int = 2

    # For TCR-FL
    tcr_beta: float = 0.9
    tcr_lambda: float = 6.0
    tcr_distance: DistanceType = "l2"

    # Optional: if a global shift happens, reduce penalty briefly
    drift_aware_reset: bool = False
    drift_reset_threshold: float = 2.5


@dataclass(frozen=True)
class EvalConfig:
    test_size: int = 4000
    # ASR for label flip: avg error on poisoned rounds minus avg error on clean rounds
    compute_asr: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data: SyntheticDataConfig = SyntheticDataConfig()
    local: LocalTrainConfig = LocalTrainConfig()
    fl: FLConfig = FLConfig()
    agg: AggregatorConfig = AggregatorConfig()
    eval: EvalConfig = EvalConfig()

    out_dir: str = "outputs"
    run_name: Optional[str] = None
