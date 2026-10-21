# Phase 4: DQN Routing Evaluation

Recorded: 2026-10-21. This is a provisional development evaluation, not a calibrated research result.

## Protocol

`experiments/train_dqn.py` trained five independent policies with seeds 11, 23, 37, 53, and 71 using `experiments/dqn_training.toml` (150 episodes per seed). Training used the same synthetic Walker configuration at epoch 2026-10-15. Evaluation used the frozen 2026-10-16 topology, fixed 20-packet schedule, packet sizes, deadlines, and energy settings from `experiments/baseline_config.toml` and `results/phase4-baseline/`.

The runner verifies that the evaluation trace hash equals the frozen baseline trace hash. Each trained policy was evaluated twice; packet results matched for every seed. Baseline metrics are from the frozen deterministic run. DQN results vary across training seeds but share the same evaluation scenario and packet schedule.

The DQN selects among currently available outgoing links plus DROP. It has no WAIT action, while contact-aware routing can wait for future contacts. The policies therefore have different action capabilities; this comparison measures the current implementation and must not be interpreted as a controlled comparison of equivalent action spaces.

## Results

| Policy | Runs | Delivered / generated | PDR | Mean delivered delay (s) | Mean hops | Mean communication energy (J) |
|---|---:|---:|---:|---:|---:|---:|
| Hop-count shortest path | 1 | 1 / 20 | 0.05 | 0.044151 | 3 | 0.153147 |
| Delay-weighted shortest path | 1 | 1 / 20 | 0.05 | 0.044151 | 3 | 0.153147 |
| Energy-constrained routing | 1 | 1 / 20 | 0.05 | 0.044151 | 3 | 0.153147 |
| Contact-aware routing | 1 | 6 / 20 | 0.30 | 176.699699 | 4 | 7.241122 |
| DQN | 5 | 0 / 20 per run | 0.00 each | Undefined (no deliveries) | Undefined | 2.795962 ± 5.879785 across training seeds |

Every DQN seed dropped all 20 packets. Seeds 11 and 23 incurred hop-limit drops (one and sixteen packets, respectively); all remaining packets were dropped by policy choice. With no delivered packets, DQN delivered-packet delay and hop metrics are undefined, not zero.

## Interpretation and Limitations

- DQN did not demonstrate competitive packet delivery in this scenario. Its mean PDR was zero, below the 0.05 snapshot baselines and 0.30 contact-aware baseline.
- The current action set excludes WAIT, making the learned policy structurally unable to reproduce the contact-aware baseline's future-contact behavior.
- Five training seeds on one fixed topology and traffic schedule do not establish statistical significance, generalization, or a general ranking of routing methods.
- The constellation, contact cutoff, traffic schedule, and energy coefficients are provisional. No FL energy is charged in this packet comparison.
- The trace uses sampled contacts and the simulator's documented single-radio deterministic model; see [the packet simulator documentation](phase3-packet-simulator.md) and [the frozen baseline report](phase4-baseline-freeze.md).

## Reproduction and Artifacts

From the repository root:

```bash
python -m pytest -q
python experiments/run_baselines.py
python experiments/train_dqn.py
```

The DQN runner writes `dqn_training_episodes.csv`, `dqn_evaluation_per_seed.csv`, `dqn_summary.json`, and one checkpoint per training seed under `results/phase4-dqn/`. The summary records the training/evaluation trace hashes and configuration hashes. The baseline runner writes to `results/phase4-baseline/`; preserve the committed baseline artifacts when rerunning experiments intended only for verification.

The recorded run used baseline-config SHA-256 `b52ee9b789e247c615bf81fc83805368a954871841bf16323a47719e38a9f86c`, training-config SHA-256 `c3a55d60657d16f9a1df918aec591a797acd5e5db77822c753aa2bbcd9494073`, frozen evaluation topology SHA-256 `f6dfe16a04ccf32ae983531d74f09ced5ea39688963898192401e3fe6e7f987d`, and training topology SHA-256 `629f9c760141fc239b00be2ccb787abd614b2e3ce6867f2159504e2ce46b2449`.
