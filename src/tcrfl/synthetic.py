from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import ObstacleMapConfig, SyntheticDataConfig, TaskSwitchConfig


@dataclass(frozen=True)
class ObstacleMap:
    centers: np.ndarray  # (K,2)
    radii: np.ndarray  # (K,)

    def label(self, x: np.ndarray) -> np.ndarray:
        # x: (N,2)
        # y(x)=1 if inside any circle
        diffs = x[:, None, :] - self.centers[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        inside_any = np.any(d2 <= (self.radii[None, :] ** 2), axis=1)
        return inside_any.astype(np.float32)


def _rotation(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


@dataclass
class ClientDistributionState:
    mean: np.ndarray  # (2,)
    v: np.ndarray  # (2,)
    cov: np.ndarray  # (2,2)


class SyntheticFLData:
    """On-the-fly per-round per-client dataset generator matching docs/dataset_syn.md."""

    def __init__(self, cfg: SyntheticDataConfig):
        self.cfg = cfg
        if cfg.n_honest + cfg.n_malicious != cfg.n_clients:
            raise ValueError("n_honest + n_malicious must equal n_clients")

        self._rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 123)
        self._domain_min = float(cfg.domain_min)
        self._domain_max = float(cfg.domain_max)

        self._task_switch = cfg.task_switch
        self._base_map_cfg = cfg.obstacle_map
        self._maps = self._build_maps(self._base_map_cfg, self._task_switch)

        self.client_states: list[ClientDistributionState] = []
        self._init_clients()

        # malicious clients are last n_malicious by default
        self.malicious_ids = list(range(cfg.n_honest, cfg.n_clients))

    def _build_maps(self, base: ObstacleMapConfig, ts: TaskSwitchConfig) -> list[ObstacleMap]:
        centers = np.array(list(base.centers), dtype=np.float32)
        radii = np.array(list(base.radii), dtype=np.float32)
        base_map = ObstacleMap(centers=centers, radii=radii)
        if not ts.enabled:
            return [base_map]

        rng = np.random.default_rng(ts.seed)
        maps: list[ObstacleMap] = [base_map]
        # Create a few alternative maps (simple: jitter centers)
        for _ in range(3):
            jitter = rng.normal(0.0, 0.15, size=centers.shape).astype(np.float32)
            new_centers = np.clip(centers + jitter, -0.85, 0.85)
            maps.append(ObstacleMap(centers=new_centers, radii=radii.copy()))
        return maps

    def _init_clients(self) -> None:
        cfg = self.cfg
        self.client_states = []

        for _ in range(cfg.n_clients):
            mean0 = self._rng.uniform(cfg.domain_min * 0.6, cfg.domain_max * 0.6, size=(2,)).astype(np.float32)

            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(cfg.v_min, cfg.v_max)
            v = (speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)).astype(np.float32)

            sigma1 = self._rng.uniform(cfg.sigma1_min, cfg.sigma1_max)
            sigma2 = self._rng.uniform(cfg.sigma2_min, cfg.sigma2_max)
            theta = self._rng.uniform(0, 2 * np.pi)
            R = _rotation(theta)
            D = np.array([[sigma1 * sigma1, 0.0], [0.0, sigma2 * sigma2]], dtype=np.float32)
            cov = (R @ D @ R.T).astype(np.float32)

            self.client_states.append(ClientDistributionState(mean=mean0, v=v, cov=cov))

    def task_id(self, round_t: int) -> int:
        ts = self._task_switch
        if not ts.enabled:
            return 0
        return int((round_t - 1) // max(1, ts.switch_every)) % len(self._maps)

    def obstacle_map(self, round_t: int) -> ObstacleMap:
        return self._maps[self.task_id(round_t)]

    def step_client_means(self) -> None:
        cfg = self.cfg
        for st in self.client_states:
            noise = self._rng.normal(0.0, cfg.sigma_m, size=(2,)).astype(np.float32)
            st.mean = (st.mean + st.v + noise).astype(np.float32)
            st.mean = np.clip(st.mean, cfg.domain_min, cfg.domain_max)

    def sample_client_batch(self, client_id: int, round_t: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = int(cfg.samples_per_client_per_round)
        rho = float(cfg.uniform_mix)
        n_u = int(round(n * rho))
        n_g = n - n_u

        st = self.client_states[client_id]

        xg = self._rng.multivariate_normal(mean=st.mean, cov=st.cov, size=(n_g,)).astype(np.float32)
        xu = self._rng.uniform(cfg.domain_min, cfg.domain_max, size=(n_u, 2)).astype(np.float32)

        x = np.concatenate([xg, xu], axis=0)
        x = np.clip(x, cfg.domain_min, cfg.domain_max)

        y = self.obstacle_map(round_t).label(x)

        # Poisoning: flip label every poison_every rounds for malicious clients
        if client_id in self.malicious_ids and (round_t % cfg.poison_every == 0):
            y = 1.0 - y

        return x, y

    def sample_global_test(self, round_t: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        x = self._rng.uniform(cfg.domain_min, cfg.domain_max, size=(int(size), 2)).astype(np.float32)
        y = self.obstacle_map(round_t).label(x)
        return x, y
