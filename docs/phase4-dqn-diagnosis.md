



# Phase 4.1: DQN Failure Diagnosis

Recorded: 2026-10-21. This is a provisional development diagnosis, not a calibrated research result.

## Diagnosis

The frozen-scenario failure is a policy/learning failure, not evidence of invalid actions or an unavailable packet simulator:

- Legacy seed 11 dropped packet `f0-p0` immediately at `t=0`, despite three valid next hops. Its candidate Q-values were approximately `[-0.7071, -0.6976, -0.6987]`; DROP was `-0.4324`, so greedy selection correctly followed the learned values and chose DROP.
- Legacy seed 23 sent `f0-p1` from `sat:1` to `sat:14`, then repeatedly returned to `sat:1`. The route alternated between those satellites for 16 decisions and ended at the hop limit. The actions were valid and unmasked; they were simply unproductive.
- In the revised five-seed run, no training episode delivered a packet and the captured training decisions contained no positive delivery reward. Mean returns remained close to the immediate DROP return of `-5`. Four evaluation seeds dropped every packet by policy choice. Seed 37 made forwarding decisions for three packets, but each looped until the hop limit; its other 17 packets were policy drops.
- The one-hop controlled graph is learnable: `test_dqn_learns_the_unambiguous_one_hop_route` trains the DQN and verifies delivery using the learned greedy action. The frozen contact-aware reference also delivers 6 of 20 packets, so successful packet delivery is possible in the simulator.

Taken together, the evidence points to insufficient successful exploration/credit assignment on the sparse, dynamic pilot task, with valid-but-cyclic choices and DROP becoming preferable under the learned Q-values. The evidence does **not** isolate one single cause or establish that changing the reward is justified. Adding WAIT by itself did not improve delivery. The original state also omits some temporal context, so partial observability and the training/evaluation environment gap remain plausible contributors.

## Audit and Correction

- Candidate features and Q-values are row-aligned by the shared candidate scorer. The score rows are permutation-equivariant; action masks are applied during exploration, greedy selection, and next-state TD maximization. DROP remains valid when there are no forwarding actions. Tests cover changing candidate counts, candidate reordering, isolated/disconnected nodes, and masked next-state choices.
- Training reward components are logged separately. A route step applies its configured delay, normalized energy, and link-quality penalties once; delivery adds `+10`, and DROP adds `-5`. The packet simulator separately debits TX/RX energy once per transmission. Evaluation has no RL reward or exploration epsilon, so those fields are null in its decision trace; training decisions include epsilon, rewards, components, next state, and termination flags.
- The training environment has no packet deadline. When enabled, its WAIT advances one topology sample if a future contact-graph route exists. Packet evaluation additionally requires the future route to fit the packet's remaining deadline. The contact-aware baseline plans and waits automatically.
- A termination-semantics bug was corrected: hop-limit exhaustion is now a terminal packet drop (`loss_cause="hop_limit"`), not an environment truncation. Trainer time-limit truncations are recorded separately and bootstrap in TD targets; only true terminal transitions suppress bootstrap. Regression tests cover both cases.

The termination correction fixes target semantics and diagnostics; it is not claimed as the cause of the zero-PDR result or as a performance improvement.

## Controlled and Pilot Results

The hand-checkable one-hop training test delivers successfully. The revised pilot used 150 episodes for each of seeds 11, 23, 37, 53, and 71, with WAIT enabled and 640 maximum decisions per episode. Its evaluation trace is verified against the frozen baseline trace.

| Policy / seed | Delivered / generated | PDR | Mean delay | Mean hops | Communication energy | Drop outcomes |
|---|---:|---:|---:|---:|---:|---|
| Hop-count shortest path | 1 / 20 | 0.05 | 0.044151 s | 3 | 0.153147 J | 19 no-feasible-route |
| Delay-weighted shortest path | 1 / 20 | 0.05 | 0.044151 s | 3 | 0.153147 J | 19 no-feasible-route |
| Energy-constrained routing | 1 / 20 | 0.05 | 0.044151 s | 3 | 0.153147 J | 19 no-feasible-route |
| Contact-aware routing | 6 / 20 | 0.30 | 176.699699 s | 4 | 7.241122 J | 14 deadline-expired |
| DQN, seed 11 | 0 / 20 | 0.00 | Undefined | Undefined | 0.000000 J | 20 policy drops |
| DQN, seed 23 | 0 / 20 | 0.00 | Undefined | Undefined | 0.000000 J | 20 policy drops |
| DQN, seed 37 | 0 / 20 | 0.00 | Undefined | Undefined | 1.605351 J | 17 policy drops, 3 hop-limit drops |
| DQN, seed 53 | 0 / 20 | 0.00 | Undefined | Undefined | 0.000000 J | 20 policy drops |
| DQN, seed 71 | 0 / 20 | 0.00 | Undefined | Undefined | 0.000000 J | 20 policy drops |

DQN PDR was `0.00 ± 0.00` across training seeds. Mean communication energy was `0.321070 ± 0.717935 J`; delay and hop metrics are undefined because no packet was delivered. The inference-time no-WAIT ablation also had zero deliveries for every seed. It uses the same WAIT-trained weights with WAIT masked at evaluation, so it is not a separately trained action-space comparison.

**Conclusion:** DQN is still failing on the satellite pilot. It demonstrates useful behavior only on the controlled one-hop graph; it has not demonstrated repeatable useful routing on the 24-satellite scenario. No improvement is claimed.

## Reproduction and Artifacts

Run from the repository root:

```bash
python -m pytest -q
python experiments/train_dqn.py --output-dir results/phase4-dqn-diagnostics/phase4-1-final
```

The run uses the fixed packet schedule and frozen baseline trace. The baseline runner was not rerun, and the original `results/phase4-baseline/` and `results/phase4-dqn/` outputs were preserved.

| Artifact | Purpose |
|---|---|
| [`failed_packet_trajectories.json`](../results/phase4-dqn-diagnostics/failed_packet_trajectories.json) | Legacy seed 11/23 failed packet decisions, candidate features, masks, Q-values, and packet routes |
| [`dqn_training_decisions.json`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_training_decisions.json) | Captured training decisions for seeds 11/23, including episode/decision, packet ID, time, feature rows, masks, Q-values, epsilon, reward breakdown, next state, and terminal/truncation flags |
| [`dqn_training_episodes.csv`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_training_episodes.csv) | Episode returns, losses, delivery, transition counts, and truncation state for all five seeds |
| [`dqn_evaluation_decisions.json`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_evaluation_decisions.json) | Greedy per-packet decision traces for all five seeds |
| [`dqn_evaluation_packet_outcomes.json`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_evaluation_packet_outcomes.json) | Per-packet route, status, delay, hops, and drop reason |
| [`dqn_evaluation_per_seed.csv`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_evaluation_per_seed.csv) | Per-seed packet metrics |
| [`dqn_wait_ablation_per_seed.csv`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_wait_ablation_per_seed.csv) | Inference-time WAIT-mask ablation |
| [`dqn_summary.json`](../results/phase4-dqn-diagnostics/phase4-1-final/dqn_summary.json) | Baseline comparison, seed summaries, checkpoint hashes, and configuration/trace identifiers |

Reproducibility identifiers:

- Baseline config SHA-256: `b52ee9b789e247c615bf81fc83805368a954871841bf16323a47719e38a9f86c`
- Training config SHA-256: `af2b83d03676b7bc78fb96a66dc24e5546699362df04697ad519ae5de74244f1`
- Frozen evaluation trace SHA-256: `f6dfe16a04ccf32ae983531d74f09ced5ea39688963898192401e3fe6e7f987d`
- Training trace SHA-256: `629f9c760141fc239b00be2ccb787abd614b2e3ce6867f2159504e2ce46b2449`
- Baseline CSV SHA-256: `bb7abc8c38f480eccf7fb44d1c92303daf394c286acec86bd3d13d763d9069a1`
- Final evaluation CSV SHA-256: `173b4d9dbea3f748b66c3b31d2bd67ad0ed93c4c505ee1ebed1d58752a3f1eeb`
- Final run summary SHA-256: `766c702580d4e2ba7699985f6975232c63cdcff21000dc52ed17f3278a2f063f`
- Frozen baseline config, CSV, summary, and topology file hashes were rechecked after the run and unchanged.

## Limitations

This remains a single synthetic topology and fixed traffic schedule with provisional energy coefficients. The five training seeds do not establish statistical significance or generalization. The DQN training state/action/reward loop is not the full event-driven packet simulator, and the training environment has no packet deadline. No FL energy, fuzzy clustering, or federated learning is part of this packet-routing comparison.
