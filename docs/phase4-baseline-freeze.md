# Phase 4 Baseline Freeze

Recorded: 2026-10-20. This freezes the current deterministic packet-simulator baseline for future DQN comparisons. Values are provisional development assumptions, not calibrated hardware or research claims.

## Frozen Scenario

| Setting | Value |
|---|---|
| Constellation | Synthetic Walker-Delta, 24 satellites, 6 planes, phase factor 1 |
| Orbit | Circular synthetic elements, 550 km altitude, 53 deg inclination |
| Simulation | 2026-10-16 UTC epoch, 6,000 s duration, 10 s topology step |
| Link model | ISL cutoff 10,000 km, spherical LOS Earth radius 6378.137 km; ground station at 0 deg latitude/longitude, 10 deg elevation mask |
| Flow | `sat:1` to `gs:gateway`, 20 packets at 0, 300, ..., 5,700 s |
| Packet | 4,096 bits, 600 s deadline |
| Initial satellite energy | 100 J per satellite |
| Radio coefficients | TX electronics 1e-9 J/bit; RX electronics 1e-9 J/bit; TX amplifier 1e-12 J/bit/km2 |
| Link rate | 1,000,000 bit/s |
| Processing | 0 s and 0 J per hop |
| Contention | Single half-duplex radio per endpoint; deterministic event insertion order |
| Randomness | No topology or packet randomness in this scenario; seed 123 retained as run metadata |

The complete machine-readable inputs are in [`baseline_config.toml`](../experiments/baseline_config.toml). The runner and outputs are [`run_baselines.py`](../experiments/run_baselines.py), [`baseline_runs.csv`](../results/phase4-baseline/baseline_runs.csv), [`baseline_summary.json`](../results/phase4-baseline/baseline_summary.json), and [`baseline_topology.json`](../results/phase4-baseline/baseline_topology.json).

## Frozen Outputs

| Policy | Delivered | PDR | Mean delivered delay | Mean waiting delay | Mean hops | Communication energy | Drop causes |
|---|---:|---:|---:|---:|---:|---:|---|
| Hop-count shortest path | 1/20 | 0.05 | 0.044151 s | 0 s | 3 | 0.153147 J | 19 no-feasible-route |
| Delay-weighted shortest path | 1/20 | 0.05 | 0.044151 s | 0 s | 3 | 0.153147 J | 19 no-feasible-route |
| Energy-constrained routing | 1/20 | 0.05 | 0.044151 s | 0 s | 3 | 0.153147 J | 19 no-feasible-route |
| Contact-aware routing | 6/20 | 0.30 | 176.699699 s | 176.633219 s | 4 | 7.241122 J | 14 deadline-expired |

These results expose the scenario's current behavior; they do not show that contact routing is generally superior. The first three policies selected identical routes for this source/trace. Contact-aware routing delivered more packets but incurred much larger waits and delay. No FL communication or training energy is included.

Each policy was repeated with identical inputs and produced identical packet results/event logs. The runner checks this before saving output.

## Reproducibility Identifiers

- Git revision: `2479fc42dab09b1ad262384a9149038022529699`
- Git working tree: dirty; use source hashes in `baseline_summary.json` to identify the exact uncommitted implementation.
- Baseline config SHA-256: `b52ee9b789e247c615bf81fc83805368a954871841bf16323a47719e38a9f86c`
- Topology trace SHA-256: `f6dfe16a04ccf32ae983531d74f09ced5ea39688963898192401e3fe6e7f987d`
- Results CSV SHA-256: `bb7abc8c38f480eccf7fb44d1c92303daf394c286acec86bd3d13d763d9069a1`
- Summary JSON SHA-256: `f479129285a94888df7347ad074a8050d0e9dc7dd39474e2b5ce8690210a1407`
- Test command/result: `python -m pytest -q`, 48 passed.
- Environment: Python 3.14.2, NetworkX 3.7, NumPy 2.5.4, SGP4 2.27, Skyfield 1.55, PyTorch 2.14.1+cpu.

## Comparison Protocol

For subsequent DQN comparisons, preserve this topology trace, packet schedule, packet sizes/deadlines, initial energy, routing/packet-drop rules, and policy evaluation settings. Record any change as a separate scenario. Train DQN only on training scenarios/seeds; report held-out scenarios separately. Do not infer calibrated physical energy or general algorithm superiority from this single provisional flow.
