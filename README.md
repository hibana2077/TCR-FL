# TCR-FL
Temporal Consistency Regularized Federated Learning

## Quickstart (toy experiment)

1) Install deps

`pip install -r requirements.txt`

2) Run baselines

`python -m experiments.run_toy --agg fedavg --rounds 60`

`python -m experiments.run_toy --agg trimmed_mean --trim_ratio 0.2 --rounds 60`

`python -m experiments.run_toy --agg krum --krum_f 2 --rounds 60`

3) Run TCR-FL

`python -m experiments.run_toy --agg tcr --tcr_beta 0.9 --tcr_lambda 6.0 --rounds 60`

4) Run TCR-FL + TrimmedMean (stacked defense)

`python -m experiments.run_toy --agg tcr_trimmed_mean --trim_ratio 0.2 --tcr_lambda 6.0 --rounds 60`

Outputs are saved under `outputs/<run_name>/` (JSON + plots).
