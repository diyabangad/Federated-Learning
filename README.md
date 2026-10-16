# Energy-Efficient Federated Deep Learning for Adaptive Routing and Cluster Formation in Dynamic Inter-Satellite Networks

Research implementation for the project proposal. The goal is to evaluate fuzzy cluster-head selection and federated deep reinforcement learning for routing in a time-varying LEO inter-satellite network, while charging the energy used by local training and federated communication to the satellites' energy budgets.

This repository is being built in the order of Section 7 of the proposal. The current foundation covers implementation steps 1 and 2: explicit experiment/topology settings, then orbital positions and time-indexed link availability. Routing, clustering, energy accounting, and learning are not implemented yet.

## Research scope

The eventual evaluation follows the proposal's metrics and baselines: network lifetime (`T1`, `T50`, `T90`, and connectivity lifetime), deadline-bounded packet delivery ratio with loss causes, end-to-end delay, throughput, hop count, control overhead, federated-learning energy and communication volume, convergence, and residual-energy fairness. Planned comparisons include shortest path, contact-graph routing, Q-routing, centralized DQN, isolated learners, fuzzy/DRL ablations, and federated variants.

The implementation must not silently turn unresolved proposal decisions into claimed experimental facts. In particular, the constellation/TLE data, traffic model, link budget and energy constants, agent/reward settings, and federation schedule must be recorded and justified before evaluation runs. Values such as the maximum ISL range are required inputs, not guessed defaults.

## Current implementation

### Step 1: system-model inputs

`SimulationConfig` fixes a UTC start time, duration, and sampling interval. `TopologyConfig` accepts exactly one orbital source (a local TLE file or a synthetic Walker constellation), a maximum ISL range, and optional ground-station coordinates and elevation limits. These are validated at construction so invalid or ambiguous runs fail early.

For development only, the repository uses a provisional synthetic Walker-Delta pilot: 24 satellites in 6 planes, phase factor 1, 550 km altitude, and 53-degree inclination. The 10,000 km ISL cutoff and the example ground station are also pilot inputs, not values validated by the proposal or a link-budget study. They exist to make topology generation reproducible while the model is developed; publication experiments must justify them and include sensitivity analysis. The proposal's traffic and energy parameters remain undecided.

### Step 2: orbital positions and contact topology

`build_topology_trace` uses SGP4 propagation from either TLEs or the configured synthetic Walker elements to sample each satellite's Earth-centred position. At each sample it builds an undirected NetworkX graph: satellite-to-satellite edges require both an unobstructed Earth line of sight and range within the configured limit; satellite-to-ground-station edges require the configured minimum elevation. It also groups consecutive visible samples into contact windows.

Contact windows are sampled approximations: each observed active sample represents one sampling interval, so window boundaries have up to one time-step of discretization. The satellite-satellite Earth-blockage test uses a spherical Earth with the documented default equatorial radius; it is a geometry filter, not a complete link-budget or antenna model. Ground-station visibility uses Skyfield's WGS84 observer geometry.

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

## Project layout

```text
src/fedroute/config.py    validated simulation and topology inputs
src/fedroute/topology.py  SGP4 propagation, time-indexed graphs, contact windows
tests/                    focused tests for geometry, contacts, and Walker propagation
```

## Verification

```bash
python -m pytest
```

The current tests cover geometric line of sight, sampled contact-window boundaries, and deterministic Walker propagation. They do not validate a particular TLE catalog, real link budget, or full experiment; those require project decisions and suitable source data.

## Next proposal steps

Before treating pilot results as research results, justify the constellation, orbital sampling interval, ISL range, and ground-station locations/elevation limits. The next implementation work follows the proposal's recommended order: steps 4 and 5 (routing environment and centralized DQN validation), then step 3 (Mamdani fuzzy cluster-head election), followed by federated training in step 6.