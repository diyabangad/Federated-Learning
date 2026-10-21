# Energy-Efficient Federated Deep Learning for Adaptive Routing and Cluster Formation in Dynamic Inter-Satellite Networks

Research implementation for the project proposal. The goal is to evaluate fuzzy cluster-head selection and federated deep reinforcement learning for routing in a time-varying LEO inter-satellite network, while charging the energy used by local training and federated communication to the satellites' energy budgets.

This repository is being built in the order of Section 7 of the proposal. The system model and sampled topology are implemented, with prototype routing, centralized DQN training, and fuzzy cluster-head selection now available for development and validation. These prototypes do not yet constitute calibrated research experiments.

## Research scope

The eventual evaluation follows the proposal's metrics and baselines: network lifetime (`T1`, `T50`, `T90`, and connectivity lifetime), deadline-bounded packet delivery ratio with loss causes, end-to-end delay, throughput, hop count, control overhead, federated-learning energy and communication volume, convergence, and residual-energy fairness. Planned comparisons include shortest path, contact-graph routing, Q-routing, centralized DQN, isolated learners, fuzzy/DRL ablations, and federated variants.

The implementation must not silently turn unresolved proposal decisions into claimed experimental facts. In particular, the constellation/TLE data, traffic model, link budget and energy constants, agent/reward settings, and federation schedule must be recorded and justified before evaluation runs. Values such as the maximum ISL range are required inputs, not guessed defaults.

## Current implementation

### Step 1: system-model inputs

`SimulationConfig` fixes a UTC start time, duration, and sampling interval. `TopologyConfig` accepts exactly one orbital source (a local TLE file or a synthetic Walker constellation), a maximum ISL range, and optional ground-station coordinates and elevation limits. These are validated at construction so invalid or ambiguous runs fail early.

`ExperimentConfig.from_toml(path)` combines those topology settings with explicit traffic flows, energy coefficients, and metric endpoints. `configs/experiment.template.toml` is a complete, loadable synthetic validation scenario. It is explicitly marked `provisional`, so it can exercise the pipeline without being mistaken for calibrated research input. Replace its assumptions and update the provenance note before research runs.

| Parameter | Current entry | Unit / definition | Status |
|---|---|---|---|
| Orbit | Walker-Delta, 24 satellites / 6 planes, 550 km, 53 deg | constellation | provisional pilot |
| ISL cutoff | 10,000 | km | provisional pilot |
| Simulation start / duration / step | 2026-10-16 / 600 / 10 | UTC / s / s | provisional pilot |
| Ground station | gateway at 0 deg, 0 deg; elevation 10 deg | coordinates / deg | provisional pilot |
| Experiment run seeds | 123 | integer seed list | smoke test; use 10+ for evaluation |
| Traffic process and seed | Poisson, seed 123 | arrival distribution / integer | provisional synthetic load |
| Traffic flow | `sat:1` to `gs:gateway`; rate 0.1, packet 4096, deadline 2 | packets/s, bits, s | provisional synthetic load |
| Satellite battery capacity | 100 | J | provisional test value |
| Transmit electronics | 1e-9 | J/bit | provisional test value |
| Transmit amplifier | 1e-12 | J/bit/km2 | provisional test value |
| Receive electronics | 1e-9 | J/bit | provisional test value |
| FL local training / communication | 0.01 / 1e-6 | J/transition / J/byte | provisional test values |
| T1 / T50 / T90 | 1%, 50%, 90% depleted satellites | fraction of satellites depleted | explicit pilot thresholds |
| Connectivity lifetime | `sat:1` to `gs:gateway` | elapsed time until no path | explicit pilot endpoints |
| PDR | deadline-delivered packets / generated packets | ratio; late and other drops remain loss causes | fixed definition |
| Delay / throughput / hops | mean delay and hops of deadline-delivered packets; delivered payload bits / simulated seconds | s / hops / bit/s | fixed definitions |
| Control overhead | transmitted control bytes by configured component; also bytes per deadline-delivered packet | bytes | component list required |
| FL convergence | absolute difference between consecutive 10-episode mean-return windows is at most 0.01 return units for 3 successive comparisons | episodes | provisional criterion |
| Residual-energy fairness | Jain index $J(E)=(\sum_i E_i)^2/(N\sum_i E_i^2)$ over all satellite residual energies | dimensionless | fixed formula |

For development only, the repository uses a provisional synthetic Walker-Delta pilot: 24 satellites in 6 planes, phase factor 1, 550 km altitude, and 53-degree inclination. The 10,000 km ISL cutoff and the example ground station are also pilot inputs, not values validated by the proposal or a link-budget study. They exist to make topology generation reproducible while the model is developed; publication experiments must justify them and include sensitivity analysis. The proposal's traffic and energy parameters remain undecided.

### Step 2: orbital positions and contact topology

`build_topology_trace` uses SGP4 propagation from either TLEs or the configured synthetic Walker elements to sample each satellite's Earth-centred position. At each sample it builds an undirected NetworkX graph: satellite-to-satellite edges require both an unobstructed Earth line of sight and range within the configured limit; satellite-to-ground-station edges require the configured minimum elevation. It also groups consecutive visible samples into contact windows.

Contact windows are sampled approximations: each observed active sample represents one sampling interval, so window boundaries have up to one time-step of discretization. The satellite-satellite Earth-blockage test uses a spherical Earth with the documented default equatorial radius; it is a geometry filter, not a complete link-budget or antenna model. Ground-station visibility uses Skyfield's WGS84 observer geometry.

Validation tests check the SGP4 period at 780 km against the Kepler two-body period (approximately 100 minutes), compare generated ground-contact elevation/range against a direct Skyfield observer calculation, and check a known Cartesian satellite range. These checks validate internal consistency; a GMAT or independently sourced orbit/contact cross-validation has not yet been performed.

The 24-satellite pilot was compared over 6,000 s at 20, 10, 5, and 2.5 s resolution. All runs detected 236 contacts, repeat hashes matched, and common-timestamp availability agreed. Maximum adjacent-resolution boundary differences were 10, 5, and 2.5 s. Instrumented build times ranged from 5.800 s at 20 s resolution to 50.642 s at 2.5 s; peak traced Python memory ranged from 9.24 to 162.31 MiB. Full metrics, limitations, source/dependency hashes, and saved traces are in [the Phase 1 validation report](docs/phase1-topology-validation.md) and `results/phase1-topology/`. These are measurements from one synthetic scenario, not general performance claims.

### Step 3: fuzzy cluster-head selection

`FuzzyClusterSelector` scores satellites from normalized residual energy, processing capacity, traffic load, contact duration, link quality, neighbor count, and ground-station distance. `select_cluster_heads` applies those scores to a graph snapshot and forms deterministic, non-overlapping one-hop clusters: the highest-scoring unassigned satellite becomes a head, and its unassigned satellite neighbors join its cluster. Isolated satellites become single-node clusters. The current rule groups and feature normalization are prototype choices and must be replaced or calibrated against the proposal before interpreting experimental results.

### Steps 4 and 5: routing environment and centralized DQN

`SatelliteRoutingEnv` provides a masked next-hop action space over a topology trace, with a 9-value feature row per candidate. The final feature identifies whether that candidate can reach the destination in the current snapshot. Unreachable candidates remain selectable so the agent may move toward a contact available in a later snapshot; an isolated satellite has only the explicit drop action. Set `cluster_aware=True` to expose same-cluster membership at observation index 1; the default leaves it zero. `DQNPolicy` scores each candidate row once and appends one state-independent drop-action value. The current `DQNTrainer` uses replay, a target network, masked candidate actions, and epsilon scheduling. `experiments/train_dqn.py` trains independent seeds and evaluates each policy through `PacketSimulator` on the frozen baseline trace and packet schedule. Its current action set is available outgoing links plus DROP; unlike contact-aware routing, it cannot wait for future contacts. The pilot evaluation delivered 0/20 packets for each of five training seeds, so DQN is integrated but not competitive in this scenario; see the [Phase 4 DQN evaluation report](docs/phase4-dqn-evaluation.md). Routing requires separate TX/RX electronics coefficients and tracks per-satellite initial, communication, compute, and residual joules; over-budget sends are rejected before energy is charged. Coefficients are still provisional.

### Step 6: federated DQN prototype

`FederatedDQNTrainer` initializes a local copy of the shared policy for each client environment, trains each copy, then applies sample-count-weighted FedAvg using the number of local transitions. `FederatedEnergyConfig` requires caller-supplied joules per local training transition and joules per transmitted byte; costs are deducted from persistent per-client energy budgets, and a round is rejected if a client cannot afford it. Each round reports full model state-dict payload bytes and assumes one full download and one full upload per participating client. These coefficients are explicit accounting inputs, not calibrated hardware measurements. Transport framing, retries, compression, non-IID client partitions, hierarchical cluster-head/ground-station aggregation, and asynchronous updates remain unimplemented and must be addressed before research results are reported.

### Step 7: deterministic routing references

`shortest_path_route` supplies hop-count and delay-weighted routes on a single graph snapshot with lexicographic tie-breaking; `calculate_route_metrics` reports hop, propagation, transmission, processing, waiting, and annotated energy totals. `energy_constrained_route` rejects paths that exceed per-node annotated energy budgets. `contact_graph_route` plans over sampled graphs and can wait for a future contact. Intermediate routing decisions occur only at sample boundaries; a link must cover packet delay within its remaining contact, and an in-flight hop occupies enough sample slots to complete. The final hop reports exact delay components. This is a discretized reference, not continuous-time contact-plan routing. Q-routing and the proposal's clustering/federation ablations are not implemented yet.

The current 24-satellite deterministic packet baseline is frozen in `experiments/baseline_config.toml`. Run `python experiments/run_baselines.py` to rebuild the same topology/traffic scenario and write per-policy metrics, topology, hashes, and repeat checks. Current measurements and comparison limits are recorded in [the baseline freeze report](docs/phase4-baseline-freeze.md); do not interpret this single provisional scenario as a general performance result. To train and evaluate DQN against the frozen scenario, run:

```bash
python experiments/train_dqn.py
```

This writes checkpoints, per-seed metrics, training history, and a machine-readable summary under `results/phase4-dqn/`. The committed report documents the current run and its limitations.

### Phase 3: deterministic packet simulation

`PacketSimulator` processes fixed-time packet generation, route selection, contact/radio waiting, TX start, TX completion, propagation, arrival, delivery, and drops. It uses a single half-duplex radio per endpoint and serializes competing packets in deterministic event order. Satellite TX/RX and configured forwarding-compute energy are recorded separately; aggregate PDR, delivered-packet delay components, hops, drops by cause, and residual energy are reported. Empty denominators return `None`. See [the packet simulator assumptions and units](docs/phase3-packet-simulator.md). This is not yet a finite-buffer/MAC/channel model, and FL costs remain separate from packet energy.

## Setup

Requires Python 3.11 or newer.

```bash
python -m pip install -e '.[dev]'
```

To reproduce the current pilot trace, no external orbital data is required. To use a real constellation, provide a TLE file with one or more satellites (standard two-line elements, optionally preceded by satellite names), and record the catalog source and epoch in any experiment.

Example of configuring and generating a trace:

```python
from datetime import datetime, timezone
from pathlib import Path

from fedroute.config import (
    GroundStationConfig,
    SimulationConfig,
    TopologyConfig,
    WalkerConfig,
)
from fedroute.topology import build_topology_trace

start = datetime(2026, 10, 16, tzinfo=timezone.utc)
settings = SimulationConfig(
    start_utc=start,
    duration_seconds=600,
    time_step_seconds=10,
)
topology = TopologyConfig(
    max_isl_range_km=10_000.0,
    walker=WalkerConfig(
        total_satellites=24,
        planes=6,
        phase_factor=1,
        altitude_km=550.0,
        inclination_deg=53.0,
        epoch_utc=start,
    ),
    ground_stations=(
        GroundStationConfig(
            "gateway", latitude_deg=0.0, longitude_deg=0.0,
            minimum_elevation_deg=10.0,
        ),
    ),
)
trace = build_topology_trace(settings, topology)
print(len(trace.graphs), "time samples")
print(len(trace.contact_windows), "sampled contact windows")
```

This example uses provisional pilot values, not calibrated experimental parameters. Replace them with justified values and record the source and rationale before running research experiments. To use TLEs instead, pass `tle_file=Path("data/constellation.tle")` and omit `walker`.

To use the TOML system model in code:

```python
from pathlib import Path

from fedroute.config import ExperimentConfig

experiment = ExperimentConfig.from_toml(Path("configs/experiment.toml"))
```

Copy the template to `configs/experiment.toml`, update the name and provenance note, and keep the completed file with experiment outputs. The loader validates structure and numeric domains; provenance and calibration evidence must still be recorded by the researcher. The template uses one run seed for smoke testing; the final evaluation should supply at least ten distinct seeds. All template values are smoke-test assumptions, not literature or hardware measurements.

## Project layout

```text
src/fedroute/config.py    validated simulation and topology inputs
src/fedroute/topology.py  SGP4 propagation, time-indexed graphs, contact windows
src/fedroute/routing.py   masked next-hop routing environment
src/fedroute/dqn.py       centralized DQN policy and training prototype
src/fedroute/fuzzy.py     fuzzy scores and graph-based cluster assignment
src/fedroute/federated.py sample-weighted DQN aggregation prototype
src/fedroute/baselines.py snapshot and sampled contact-graph route references
src/fedroute/packet_sim.py deterministic packet event simulator and metrics
scripts/validate_topology_resolution.py resolution, geometry, route, and reproducibility checks
scripts/validate_orbit_reference.py SGP4 versus analytical two-body position comparison
scripts/validate_contact_crossings.py bisection reference for sampled contact boundaries
experiments/baseline_config.toml frozen deterministic packet-baseline inputs
experiments/run_baselines.py shared-scenario routing baseline runner
experiments/dqn_training.toml  DQN training and reward settings
experiments/train_dqn.py       reproducible DQN training and packet evaluation
results/phase1-topology/  saved JSON traces and comparison report
results/phase4-baseline/   baseline CSV, summary, and topology trace
results/phase4-dqn/        DQN checkpoints, training history, and evaluation summary
docs/phase1-topology-validation.md measured Phase 1 validation report
docs/phase3-packet-simulator.md packet event, energy, metric, and limitation definitions
docs/phase4-baseline-freeze.md frozen baseline scenario and results
docs/phase4-dqn-evaluation.md DQN comparison against the frozen baseline
tests/                    focused geometry and prototype behavior tests
```

## Verification

```bash
python -m pytest
```

The tests cover geometric line of sight, sampled contact-window boundaries and censoring, synthetic short-contact aliasing, a known bisection crossing, orbital periods, direct Skyfield ground checks and threshold behavior, deterministic Walker propagation, and routing, DQN, fuzzy scoring, clustering, federated aggregation, packet event simulation, and deterministic route baselines. The DQN pilot was also run against the frozen packet scenario; it is reproducible but delivered no packets. These checks do not validate a particular TLE catalog, external ephemeris truth, real link budget, calibrated fuzzy rules or energy coefficients, DQN convergence or competitive performance, hierarchical/asynchronous federation, or generalization beyond this scenario.

## Next proposal steps

Before treating pilot results as research results, justify the constellation, orbital sampling interval, ISL range, ground-station locations/elevation limits, energy/reward parameters, fuzzy rules, and FL energy coefficients. Remaining work includes completing step 6 with cluster-head and ground-station aggregation plus asynchronous updates, defining realistic client partitions, implementing Q-routing and the Section 8 ablations, and adding repeated-experiment metrics and figures.