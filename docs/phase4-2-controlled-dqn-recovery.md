# Phase 4.2: Controlled DQN Recovery

Recorded: 2026-10-22. This is a controlled pilot on one frozen topology, trace, and packet schedule; it is not evidence of generalization.

## Outcome

Horizon-normalizing the WAIT penalty reduced the DQN's premature DROP behavior and produced 7 deliveries in 100 evaluation packets, whereas Phase 4 and Phase 4.1 produced none. The deliveries came from only two of five seeds (4/20 and 3/20). The corrected DQN still had forwarding cycles in 92/100 packets, and its aggregate 7% PDR was below both the frozen contact-aware baseline and the simple greedy reference (each 30% on their 20-packet evaluation). The improvement over the zero-delivery DQN runs is not statistically supported by five seeds on one trace.

**The completion criterion for useful, reproducible frozen-scenario multi-hop routing is not met. Do not proceed to fuzzy clustering or federated learning.**

## Diagnosis and evidence

The failure is not one universal bug:

* **Premature DROP in the original DQN:** Across the five Phase 4 checkpoints, all 100 packets were lost. Of 83 DROP decisions made while at least one forwarding action was valid, DROP's Q-value exceeded the best valid forwarding Q-value. The seed-level mean margin was +0.014 to +0.216. Thus these were policy choices under the learned values, not invalid actions rejected by the environment.
* **Forwarding cycles:** The original seed 23 looped between two satellites for 16 hops on 16 packets; seed 11 did so once. Those are 17 cyclic packet routes and 255 total repeated-node traversals. Seeds 37, 53, and 71 mostly chose DROP instead.
* **WAIT cost in Phase 4.1:** WAIT incurred `-0.2 * (wait_seconds / sample_interval)`, or `-0.2` per 10-second sample. A 30-sample wait therefore costs `-6`, more than the `-5` DROP penalty, before delivery is credited. Phase 4.1 had 97 valid-alternative DROP decisions across its five evaluations and still delivered 0/100 packets.
* **Not a universally unlearnable task:** In controlled training, all seeds learned one- and two-hop static routes. Four of five learned the three-hop graph; only one of five learned the four-hop graph. A three-hop graph with a dead-end branch was learned by all five. A disconnected graph yielded no deliveries, as expected. All five seeds learned a two-sample time-varying contact case.
* **Representation and credit assignment remain limitations:** The candidate scorer ranks current local candidate rows. Those features and the action mask do not encode a packet's visited-node history. The observed repeated routes therefore remain locally valid and can persist until the hop limit. Sparse delivery rewards and the increased difficulty at three to four hops also show weak long-horizon credit assignment. The experiments do not isolate which of these residual factors dominates.

Candidate/action alignment and action masking were audited and tested; candidate rows remain aligned with their actions under reordering. The earlier termination fix is preserved: hop-limit exhaustion is terminal, while trainer time-limit truncations bootstrap. None of these tests makes a performance claim.

## Controlled scenario ladder

Each result below is a five-seed run with 150 training episodes per seed. “Greedy delivery” is a deterministic greedy rollout using the learned policy.

| Scenario | Greedy deliveries | Training deliveries / 750 episodes |
|---|---:|---:|
| Static, one hop | 5/5 | 553 |
| Static, two hops | 5/5 | 442 |
| Static, three hops | 4/5 | 271 |
| Static, four hops | 1/5 | 40 |
| Static, three hops with dead-end branch | 5/5 | 308 |
| Disconnected | 0/5 | 0 |
| Time-varying contact, two-sample wait | 5/5 | 331 |

WAIT-reward ablation on delayed-contact routes:

| Contact delay | Original per-sample WAIT penalty: training deliveries / 750; greedy seeds | Horizon-normalized WAIT penalty: training deliveries / 750; greedy seeds |
|---:|---:|---:|
| 5 samples | 17; 0/5 | 181; 4/5 |
| 10 samples | 0; 0/5 | 91; 4/5 |
| 20 samples | 0; 0/5 | 41; 5/5 |
| 30 samples | 0; 0/5 | 18; 5/5 |

This is direct evidence for the WAIT-cost change on the controlled delayed-contact task. It does not demonstrate that this one change solves the frozen satellite task.

## Five-seed training diagnostics

Phase 4.2 used 150 episodes per seed, up to 640 steps per episode, `gamma=0.99`, epsilon annealed from 1.0 to 0.05, and the same five seeds as earlier phases. Episode returns remained negative. The lower TD loss is reported as a training diagnostic only, not as evidence of routing quality.

| Seed | Training deliveries / 150 | Mean return (first 20 → last 20) | Mean episode TD loss (range) | Evaluation Q mean (range over valid actions) |
|---:|---:|---:|---:|---:|
| 11 | 0 | -5.282 → -5.532 | 4.252 (1.298–9.358) | -0.362 (-1.091–0.121) |
| 23 | 5 | -5.253 → -5.940 | 4.486 (2.185–7.843) | 1.667 (-0.989–11.292) |
| 37 | 1 | -5.436 → -4.824 | 3.893 (0.861–7.455) | -0.422 (-1.291–0.070) |
| 53 | 0 | -5.270 → -5.537 | 3.664 (0.901–8.439) | -0.609 (-1.318–-0.101) |
| 71 | 5 | -5.272 → -4.338 | 4.671 (1.874–8.114) | 0.906 (-0.985–10.396) |

Training reward-component sums over captured decisions (not per-episode averages):

| Seed | Delivery | DROP | WAIT delay | Hop limit | Delay | Energy | Link quality |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0 | -735 | -0.382 | -15 | -0.123 | -0.128 | -52.587 |
| 23 | +50 | -645 | -0.256 | -80 | -0.217 | -0.241 | -93.865 |
| 37 | +10 | -740 | -0.488 | -5 | -0.118 | -0.124 | -50.730 |
| 53 | 0 | -745 | -0.518 | -5 | -0.122 | -0.124 | -52.033 |
| 71 | +50 | -685 | -0.285 | -40 | -0.178 | -0.184 | -75.998 |

The full per-step reward breakdown, Q-values, action masks, epsilon, and next-state data are in [`dqn_training_decisions.json`](../results/phase4-2/dqn_training_decisions.json). In the frozen evaluation, the Phase 4.2 DQN selected no DROP actions; it replaced premature drops with valid forwarding but usually entered cycles. There were no training episodes truncated by the episode step limit.

## Frozen-trace policy comparison

All learned policies and the random-feasible reference were run against the same frozen 20-packet schedule for each of the five training seeds. The greedy snapshot-progress policy and frozen contact-aware baseline each have the single frozen 20-packet baseline run. The evaluator repeated every policy run and asserted identical results.

| Policy | Delivered / generated | PDR | Drop outcomes | Packets with cycles | Cycle revisits |
|---|---:|---:|---|---:|---:|
| Original Phase 4 DQN | 0/100 | 0% | 83 policy drops; 17 hop-limit drops | 17/100 | 255 |
| Phase 4.1 DQN | 0/100 | 0% | 97 policy drops; 3 hop-limit drops | 3/100 | 45 |
| Phase 4.2 DQN | 7/100 | 7% | 92 hop-limit drops; 1 simulation-horizon drop | 92/100 | 1,333 |
| Random feasible action | 2/100 | 2% | 98 hop-limit drops | 100/100 | 963 |
| Greedy snapshot progress | 6/20 | 30% | 14 hop-limit drops | 14/20 | 207 |
| Frozen contact-aware baseline | 6/20 | 30% | 14 deadline-expired drops | Not instrumented in frozen artifact | Not instrumented |

The Phase 4.2 per-seed DQN outcomes are:

| Seed | Delivered / 20 | PDR | Drops | Packets with cycles | Cycle revisits |
|---:|---:|---:|---|---:|---:|
| 11 | 0/20 | 0% | 20 hop limit | 20 | 275 |
| 23 | 4/20 | 20% | 16 hop limit | 16 | 232 |
| 37 | 0/20 | 0% | 20 hop limit | 20 | 297 |
| 53 | 0/20 | 0% | 19 hop limit; 1 simulation horizon | 19 | 276 |
| 71 | 3/20 | 15% | 17 hop limit | 17 | 253 |

For the two seeds with deliveries, mean end-to-end delay was 180.063 s (seed 23) and 240.070 s (seed 71); mean hop counts were 2.75 and 2.67. Mean communication energy per seed ranged from 16.0519 J to 30.8884 J for Phase 4.2. The frozen contact baseline delivered 6/20 with 176.700 s mean delay, 4 hops, and 7.2411 J communication energy. The greedy reference also delivered 6/20, with 12.863 J mean communication energy across that run.

Two of five seed-level PDRs increased from zero; three remained zero. The exact one-sided paired sign test over the two non-tied seed outcomes is `p=0.25` (two-sided `p=0.50`). This is not statistically significant, and one frozen schedule cannot support a generalization claim. The Phase 4.2 result is a small, seed-concentrated improvement over the earlier DQNs, not useful or reliable routing relative to the contact-aware reference.

## Change, tests, and limitations

The single training-objective change was an opt-in `normalize_wait_penalty_by_horizon` setting, disabled by default to preserve existing behavior. When enabled, a WAIT penalty is scaled by wait duration divided by the trace duration. No baseline or Phase 4/4.1 artifact was modified. The change avoids making a long but feasible WAIT sequence more costly than immediate DROP solely because the trace is sampled many times.

`python -m pytest -q`: **63 passed**. `git diff --check`: passed. The controlled experiment completed 35 scenario runs and 40 WAIT ablation runs. The policy evaluator verified deterministic repeats and asserted that frozen-baseline results match the preserved baseline metrics.

Known limitations: only five training seeds, one evaluation topology/trace/schedule, sparse deliveries, substantial cycle rates, and no confidence about generalization. The Q-value, return, and loss statistics are diagnostics, not standalone success criteria. The correction addresses excessive WAIT cost but does not address missing path-history information or the remaining long-horizon routing failure.

## Reproduction and integrity

From the repository root:

```bash
python experiments/controlled_dqn_phase4_2.py
python experiments/train_dqn.py \
  --training-config experiments/dqn_training_phase4_2.toml \
  --output-dir results/phase4-2
python experiments/evaluate_phase4_2_policies.py
python -m pytest -q
```

The default evaluation trace is the frozen baseline trace and the packet schedule comes from the frozen baseline configuration. SHA-256 identifiers:

| Input or source | SHA-256 |
|---|---|
| Phase 4.2 training config | `57b1172745f3fb6cd8db48bcd9d2fa070397c7e924fd3d2bba8b9f72ece5b21c` |
| Baseline config | `b52ee9b789e247c615bf81fc83805368a954871841bf16323a47719e38a9f86c` |
| Frozen baseline runs CSV | `bb7abc8c38f480eccf7fb44d1c92303daf394c286acec86bd3d13d763d9069a1` |
| Frozen baseline summary | `f479129285a94888df7347ad074a8050d0e9dc7dd39474e2b5ce8690210a1407` |
| Frozen baseline topology | `9a69e9bdc05c394d9bfeadd1322c10ed66a815521fba1b350b33eb9fd6e44ce0` |
| Frozen evaluation trace | `f6dfe16a04ccf32ae983531d74f09ced5ea39688963898192401e3fe6e7f987d` |
| Training trace | `629f9c760141fc239b00be2ccb787abd614b2e3ce6867f2159504e2ce46b2449` |
| `src/fedroute/routing.py` | `4b6fcaf1c3eb3830894f70fc00c4118b77cce88568c2f0351000e14d3c83256b` |
| `src/fedroute/dqn.py` | `4d3aba870fecbce688467f12033fd2d1561ac8a481b59fa7cbe126f60f16c365` |
| `src/fedroute/packet_sim.py` | `98dcd0b53def473581c333140123d579ff530d8ab996ee8bc1b1b3ef8199e075` |
| `experiments/train_dqn.py` | `7cb703779f62e704300d75b3286990086794f62a1aebec44a17833aaf25234f6` |
| `experiments/controlled_dqn_phase4_2.py` | `1a097c761b93b7b01682d1c1447a0dccb2d0e8c54743ca3cf12e08305efecbf2` |
| `experiments/evaluate_phase4_2_policies.py` | `7a2fd33a0b3b3161c526bb94ee3bad32e4ff6bcfda85baf5b82003a4fbf54384` |

Checkpoint hashes by seed:

| Seed | SHA-256 |
|---:|---|
| 11 | `dccc35a46662e5d6a28bf642727910c71ba1c8a6b8c34b06522bf53aa19d3110` |
| 23 | `b400e5154db91888ac2a699dc608638a3e251aa1b0efa2b4d42b121a4a888603` |
| 37 | `18261bf80c1fa9049550d0be66061d22508335a7c49fe1e358fd82018c0a317e` |
| 53 | `45bbbb9d653314753a9c1828f9c6e44813c4d0150ecdb99525a0b9a124c2f60c` |
| 71 | `0a24dd03f22cdcc1db315983f076bc049e2ec42334f0db8fdedd8e6efb656adb` |

The machine-readable run manifests, per-packet outcomes, complete training decisions, episode curves, and policy comparisons are in [`results/phase4-2/`](../results/phase4-2/). The original `results/phase4-baseline/`, `results/phase4-dqn/`, and `results/phase4-dqn-diagnostics/phase4-1-final/` outputs were preserved.
