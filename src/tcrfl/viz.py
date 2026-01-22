from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(payload: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = np.array(payload["metrics"]["accuracy"], dtype=np.float32)
    err = np.array(payload["metrics"]["error"], dtype=np.float32)
    alphas = payload.get("alphas")
    malicious_ids = set(payload.get("malicious_ids", []))

    # Accuracy / error curves
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(acc, label="accuracy")
    ax2 = ax.twinx()
    ax2.plot(err, color="tab:red", alpha=0.6, label="error")
    ax.set_xlabel("round")
    ax.set_ylabel("accuracy")
    ax2.set_ylabel("error")
    ax.set_title(payload.get("run_name", "run"))

    # Twin axes need manual legend combination
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out / "acc_error.png", dpi=150)
    plt.close(fig)

    # Alpha curves if available
    if alphas and any(a is not None for a in alphas):
        A = []
        for a in alphas:
            if a is None:
                A.append(None)
            else:
                A.append(np.array(a, dtype=np.float32))
        # find n_clients from first non-None
        first = next(x for x in A if x is not None)
        n_clients = int(first.shape[0])

        mat = np.zeros((len(A), n_clients), dtype=np.float32)
        for t, a in enumerate(A):
            if a is None:
                if t > 0:
                    mat[t] = mat[t - 1]
            else:
                mat[t] = a

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(n_clients):
            style = "--" if i in malicious_ids else "-"
            ax.plot(mat[:, i], style, linewidth=1.3, alpha=0.9, label=f"c{i}{'*' if i in malicious_ids else ''}")
        ax.set_xlabel("round")
        ax.set_ylabel("alpha")
        ax.set_title("Client weights (malicious dashed)")
        ax.legend(ncol=5, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "alphas.png", dpi=150)
        plt.close(fig)
