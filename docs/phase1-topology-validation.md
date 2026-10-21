# Phase 1: Satellite System Model Validation

## Scenario and Method

The test uses the existing 24-satellite, six-plane Walker-Delta configuration in `configs/experiment.template.toml`: 550 km altitude, 53 degree inclination, phase factor 1, 10,000 km ISL cutoff, one ground station at 0 degrees latitude/longitude with a 10 degree elevation mask, and UTC epoch 2026-10-16. All of these values remain **provisional synthetic test assumptions**. Traffic and energy inputs are recorded in the config but are not used to build the topology.

The validation command was:

```bash
python scripts/validate_topology_resolution.py \
  --duration-seconds 6000 \
  --time-steps 20 10 5 2.5 \
  --output-dir results/phase1-topology
```

The 6,000-second duration is approximately one orbit for the pilot constellation. Each run builds the topology twice and compares canonical trace hashes. Every generated ISL sample is checked against the configured range, its stored Cartesian range, and the spherical-Earth line-of-sight predicate. Runtime and peak traced Python allocations are measured; the latter excludes memory allocated outside Python's tracing system.

## Results

| Time step | Samples | Contacts | Snapshot routes | Build runtime | Build peak Python | Process peak Python | Repeat hash |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20 s | 301 | 236 | 22/301 (7.31%) | 5.959 s | 9.24 MiB | 9.24 MiB | match |
| 10 s | 601 | 236 | 43/601 (7.15%) | 12.383 s | 17.96 MiB | 35.94 MiB | match |
| 5 s | 1,201 | 236 | 85/1,201 (7.08%) | 25.269 s | 35.38 MiB | 78.30 MiB | match |
| 2.5 s | 2,401 | 236 | 170/2,401 (7.08%) | 50.374 s | 70.45 MiB | 162.31 MiB | match |

All four settings had a time-expanded route from `sat:1` to `gs:gateway`. Every generated ISL sample was checked (8,692 / 17,355 / 34,681 / 69,340 respectively) against the configured cutoff, stored Cartesian range, and spherical-Earth LOS predicate.

| Adjacent resolution pair | Contact counts | Matched | Maximum boundary delta | Common-frame availability mismatch |
|---|---:|---:|---:|---:|
| 20 s vs 10 s | 236 / 236 | 236 | 10 s | 0 / 301 |
| 10 s vs 5 s | 236 / 236 | 236 | 5 s | 0 / 601 |
| 5 s vs 2.5 s | 236 / 236 | 236 | 2.5 s | 0 / 1,201 |

### Independent Boundary Crossing Calculation

`scripts/validate_contact_crossings.py` evaluates each uncensored start/end bracket with bisection to 0.01 s. It independently computes the threshold predicate instead of reusing contact-window grouping: for ISLs it calculates distance from propagated positions and applies a separately implemented spherical-Earth segment test; for ground links it evaluates Skyfield observer elevation. It uses the same SGP4 satellite states as the graph builder, so it validates crossing-time discretization against those states, not orbit propagation against independent truth.

| Time step | Uncensored boundaries | Mean absolute sample-to-crossing error | Maximum absolute error | Sampling-bound violations |
|---:|---:|---:|---:|---:|
| 20 s | 411 | 10.256 s | 19.956 s | 0 |
| 10 s | 411 | 5.049 s | 9.976 s | 0 |
| 5 s | 411 | 2.555 s | 4.995 s | 0 |
| 2.5 s | 411 | 1.284 s | 2.495 s | 0 |

At every resolution, 61 censored trace-edge boundary sides were skipped individually; the opposite uncensored side of each contact was still evaluated. These are errors against continuous threshold crossings computed from SGP4 states, not errors against an external ephemeris.

The contact comparison pairs contact occurrences by edge and chronological order. Boundary offsets are reported as fine-resolution time minus coarse-resolution time. Maximum paired differences decrease with the coarse interval in this run; these are measured resolution-to-resolution differences, not error against true orbital contact crossings.

The window builder's nominal uncertainty for an uncensored start/end boundary is at most one sample interval. A contact already active at the trace start or still active at the trace end is marked censored; its uncertainty on that side is unbounded by the observed trace and stored as `null` rather than incorrectly assigned a finite bound.

Synthetic threshold tests use an explicitly defined contact interval [2.3 s, 6.7 s): the 10 s sampler misses it, while the 5 s and 2.5 s samplers detect it with boundary errors no greater than their sample interval. A second case places boundaries near sample times. Separate geometry tests include/exclude an ISL at exact/just-below range cutoffs and include/exclude a ground link at exact/just-above the computed elevation threshold. These test the sampler and threshold semantics; they do not independently determine a continuous-time crossing for the orbital scenario.

## Orbital and Geometry Checks

- At 780 km, the Kepler two-body period is approximately 100 minutes and agrees with the initialized SGP4 mean motion within the test tolerance.
- For the configured 24-satellite, 550 km Walker pilot, the Kepler period is approximately 95.7 minutes; all generated satellites' SGP4 periods agree within 0.01 minute.
- The orbit-reference script compares all 24 SGP4 satellite positions with an analytic circular two-body position at matching epochs:

| Offset from epoch | Mean position difference | Maximum position difference |
|---:|---:|---:|
| 0 s | 9.404 km | 12.029 km |
| 600 s | 9.613 km | 12.633 km |
| 1,800 s | 11.139 km | 15.572 km |
| 3,600 s | 15.142 km | 22.570 km |
| 6,000 s | 21.800 km | 33.419 km |

- The comparison uses SGP4 TEME states and an epoch-aligned reference frame, with WGS72 gravitational parameters. The reference is a circular two-body Kepler propagator; SGP4 adds mean-element and periodic/secular corrections. TEME is treated as inertial over this short interval. This independently implemented reference establishes the expected model difference but is not a precise ephemeris truth source.
- Generated ground-contact elevation/range metadata agrees with a direct Skyfield observer calculation in a dedicated underpass test.
- Each generated ISL is checked for configured range and Earth line-of-sight; a mocked Cartesian-position case checks the exact 100 km range result.

These are analytical and internal-consistency checks. The ground comparison uses Skyfield both for propagation and for the direct observer reference; no GMAT or independently sourced ephemeris comparison has been performed. The ISL validator asserts the same geometric constraint used by the builder, so it confirms emitted-edge invariants but is not an independent LOS implementation.

## Reproducibility Artifacts

- [`resolution_report.json`](../results/phase1-topology/resolution_report.json) records configuration SHA-256, effective simulation settings, measured comparison, runtime, route-feasibility counts, and per-run trace hashes.
- [`topology_10s.json`](../results/phase1-topology/topology_10s.json) and [`topology_5s.json`](../results/phase1-topology/topology_5s.json) contain every sampled graph, node positions, edge metadata, contact windows, and censoring/uncertainty flags.
- The 20 s and 2.5 s traces are also saved in the same directory; `resolution_report.json` lists all four configurations and adjacent comparisons.
- [`orbit_reference.json`](../results/phase1-topology/orbit_reference.json) contains the 24-satellite SGP4 versus analytic two-body comparison, including frame and model assumptions.
- [`contact_crossings_20s.json`](../results/phase1-topology/contact_crossings_20s.json), [`contact_crossings_10s.json`](../results/phase1-topology/contact_crossings_10s.json), [`contact_crossings_5s.json`](../results/phase1-topology/contact_crossings_5s.json), and [`contact_crossings_2.5s.json`](../results/phase1-topology/contact_crossings_2.5s.json) contain per-boundary crossing times and errors.
- Configuration SHA-256: `a21e7c47818d85a9bab9c67837b59b83ce590f02af67d412d893d3ee4418dc09`.
- 10-second trace SHA-256: `816084f3904edc6508c592f3bb78ad7e73f416bef89dc369f5e787dc1120fbea`.
- 5-second trace SHA-256: `a123f8d732ae80789dda2724c071914569e8a9c7ffbdb26e52bc37a3e37fabaf`.
- The four-resolution report contains the final 20 s and 2.5 s trace hashes, runtime and memory values, and per-run repeat checks.
- Runtime: Python 3.14.2, NetworkX 3.7, NumPy 2.5.4, scikit-fuzzy 0.5.0, SGP4 2.27, Skyfield 1.55, PyTorch 2.14.1+cpu.
- Git revision: `2479fc42dab09b1ad262384a9149038022529699`; working tree was dirty. The JSON report contains SHA-256 hashes for package metadata, configuration, topology, and validation-script sources.

The topology generator is deterministic and has no random draws; the config seed is retained in the report for consistent experiment bookkeeping. Reproduction requires the same config and code/dependency versions. Runtime and traced memory are host-dependent. The final matrix was collected with tracemalloc enabled, so runtime is not directly comparable with earlier uninstrumented runs.

## Phase 1 Status

- **1.1 Configuration:** A complete, unit-documented pilot config exists; values are provisional, not calibrated from cited link budgets or satellite power data.
- **1.2 Orbital propagation:** Kepler period tests pass at 550 km and 780 km; independent position/ephemeris validation remains open.
- **1.3 ISL connectivity:** All emitted links in both saved traces satisfy the configured range and current spherical-LOS predicate; external independent LOS validation remains open.
- **1.4 Ground-station connectivity:** Direct Skyfield elevation/range agreement is tested; independent tool/reference validation remains open.
- **1.5 Contact windows:** Determinism, trace-edge censoring, threshold tests, and bisection crossing comparisons against SGP4-based geometry are recorded. Independent ephemeris crossing validation remains open.
- **1.6 Resolution sensitivity:** The 20 / 10 / 5 / 2.5 s matrix is measured above for one scenario and one 6,000 s horizon. Broader scenarios remain open.
- **1.7 Reproducible traces:** Both resolution traces and a hash-bearing report are saved; repeated hashes match.

**Phase 1 is not complete.** Remaining gates include independent ephemeris/contact validation, calibrated source provenance, and broader scenario sensitivity. The analytic two-body comparison differs from SGP4 by a mean 9.4 km at epoch and 21.8 km at 6,000 s (maximum 33.4 km); this reflects different orbit models and is not an observed SGP4 error against truth. Contact crossing calculations use SGP4 states and therefore do not independently validate the orbit. No fuzzy-clustering or federated-learning work was added for this phase.
