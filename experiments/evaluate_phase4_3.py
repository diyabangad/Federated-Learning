"""Compare Phase 4.3 cycle-aware policies on frozen and unseen graph scenarios."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import tomllib

import networkx as nx
from fedroute.config import ExperimentConfig
from fedroute.dqn import DQNPolicy
from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace, build_topology_trace

try:
    from experiments.evaluate_phase4_2_policies import (
        _ReferenceCandidatePolicy,
        _fixed_flows,
        _load_policy,
        _packet_cycle_metrics,
        _trace_hash,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from evaluate_phase4_2_policies import (
        _ReferenceCandidatePolicy,
        _fixed_flows,
        _load_policy,
        _packet_cycle_metrics,
        _trace_hash,
    )


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 37, 53, 71)
UNSEEN_SEEDS = (1041, 2053, 3067, 4079, 5081)
CYCLE_PENALTY = 0.5


def _packet_metrics(packet_simulation) -> dict:
    delivered = [
        packet
        for packet in packet_simulation.packets
        if packet.status == "delivered"
    ]
    return {
        "generated_packets": packet_simulation.generated_packets,
        "delivered_packets": packet_simulation.delivered_packets,
        "dropped_packets": packet_simulation.dropped_packets,
        "packet_delivery_ratio": packet_simulation.packet_delivery_ratio,
        "average_end_to_end_delay_seconds": (
            packet_simulation.average_end_to_end_delay_seconds
        ),
        "average_hop_count": packet_simulation.average_hop_count,
        "total_communication_energy_j": (
            packet_simulation.total_communication_energy_j
        ),
        "dropped_by_reason": dict(packet_simulation.dropped_by_reason),
        **_packet_cycle_metrics(packet_simulation),
        "packets": [
            {
                "packet_id": packet.packet_id,
                "status": packet.status,
                "drop_reason": packet.drop_reason,
                "route": packet.route,
                "hop_count": packet.hop_count,
                "delay_seconds": packet.accumulated_delay_seconds,
            }
            for packet in packet_simulation.packets
        ],
        "delivered_delay_values": [
            packet.accumulated_delay_seconds for packet in delivered
        ],
    }


def _run_policy(
    *,
    label: str,
    trace: TopologyTrace,
    flows: tuple[PacketFlow, ...],
    energy: RoutingEnvConfig,
    config: PacketSimulationConfig,
    policy,
    policy_kind: str,
    scenario: str,
    training_seed: int | None = None,
    checkpoint_sha256: str | None = None,
) -> tuple[dict, list[dict]]:
    simulator = PacketSimulator(
        trace,
        energy,
        config,
        dqn_policy=deepcopy(policy),
        record_dqn_decisions=True,
    )
    result = simulator.run(flows, policy_kind)
    repeated = PacketSimulator(
        trace,
        energy,
        config,
        dqn_policy=deepcopy(policy),
    ).run(flows, policy_kind)
    if result != repeated:
        raise AssertionError(
            f"Policy evaluation was not deterministic: {scenario}/{label}/{training_seed}"
        )
    row = {
        "scenario": scenario,
        "policy": label,
        "training_seed": training_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "cycle_avoidance_penalty": config.cycle_avoidance_penalty,
        **_packet_metrics(result),
    }
    decisions = [
        {
            "scenario": scenario,
            "policy": label,
            "training_seed": training_seed,
            **decision,
        }
        for decision in simulator.dqn_decision_trace
    ]
    return row, decisions


def _unseen_scenario(topology_seed: int) -> tuple[TopologyTrace, tuple[PacketFlow, ...], dict]:
    rng = random.Random(topology_seed)
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    graph = nx.Graph()
    satellite_ids = [f"sat:u{index:02d}" for index in range(14)]
    graph.add_nodes_from(
        (node, {"kind": "satellite"}) for node in satellite_ids
    )
    destination = "gs:unseen"
    graph.add_node(destination, kind="ground_station")
    shuffled = satellite_ids[:]
    rng.shuffle(shuffled)
    for index in range(1, len(shuffled)):
        parent = shuffled[rng.randrange(index)]
        child = shuffled[index]
        graph.add_edge(parent, child)
    gateway_neighbor = shuffled[0]
    graph.add_edge(gateway_neighbor, destination)
    for left_index, left in enumerate(satellite_ids):
        for right in satellite_ids[left_index + 1 :]:
            if not graph.has_edge(left, right) and rng.random() < 0.11:
                graph.add_edge(left, right)
    for left, right in graph.edges:
        distance = rng.uniform(100.0, 1200.0)
        graph[left][right].update(
            range_km=distance,
            remaining_contact_s=60.0,
            link_quality=rng.uniform(0.65, 1.0),
        )

    topology_trace = TopologyTrace(
        tuple(start + timedelta(seconds=60 * index) for index in range(101)),
        tuple(graph.copy() for _ in range(101)),
        (),
    )
    schedule_rng = random.Random(topology_seed ^ 0x5A17)
    sources = schedule_rng.sample(satellite_ids, 4)
    flows = []
    for source in sources:
        times = tuple(
            sorted(schedule_rng.uniform(0.0, 5000.0) for _ in range(5))
        )
        flows.append(
            PacketFlow(
                source,
                destination,
                times,
                packet_size_bits=4096,
                deadline_seconds=1200.0,
            )
        )
    schedule_payload = [
        {
            "source": flow.source_satellite,
            "destination": flow.destination,
            "generation_times_seconds": flow.generation_times_seconds,
            "packet_size_bits": flow.packet_size_bits,
            "deadline_seconds": flow.deadline_seconds,
        }
        for flow in flows
    ]
    return topology_trace, tuple(flows), {
        "topology_seed": topology_seed,
        "traffic_schedule_sha256": hashlib.sha256(
            json.dumps(schedule_payload, sort_keys=True).encode()
        ).hexdigest(),
        "traffic_schedule": schedule_payload,
    }


def _load_seeded_policies(
    original_dir: Path,
    phase41_dir: Path,
    phase42_dir: Path,
    history_dir: Path,
    nstep_dir: Path,
    seed: int,
) -> list[tuple[str, DQNPolicy, str, bool, float]]:
    policies = []
    for label, directory, path_history, penalty in (
        ("original_phase4_dqn", original_dir, False, 0.0),
        ("phase41_dqn", phase41_dir, False, 0.0),
        ("phase42_dqn", phase42_dir, False, 0.0),
        ("phase42_dqn_soft_cycle_penalty", phase42_dir, False, CYCLE_PENALTY),
        ("phase43_dqn_path_history", history_dir, True, 0.0),
        ("phase43_dqn_n_step_3", nstep_dir, False, 0.0),
    ):
        policy, checkpoint_hash = _load_policy(directory / f"dqn_seed_{seed}.pt")
        policies.append((label, policy, checkpoint_hash, path_history, penalty))
    return policies


def evaluate(
    *,
    baseline_config_path: Path,
    baseline_results_dir: Path,
    original_dir: Path,
    phase41_dir: Path,
    phase42_dir: Path,
    history_dir: Path,
    nstep_dir: Path,
    output_dir: Path,
) -> dict:
    baseline_config_path = baseline_config_path.resolve()
    raw = tomllib.loads(baseline_config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.from_toml(baseline_config_path)
    baseline_summary = json.loads(
        (baseline_results_dir / "baseline_summary.json").read_text(encoding="utf-8")
    )
    frozen_trace = build_topology_trace(experiment.simulation, experiment.topology)
    frozen_trace_hash = _trace_hash(frozen_trace)
    if frozen_trace_hash != baseline_summary["topology_trace_sha256"]:
        raise ValueError("Frozen evaluation trace hash differs from the baseline manifest")
    frozen_flows = _fixed_flows(raw, experiment)
    phase42_reference = json.loads(
        (phase42_dir / "policy_comparison.json").read_text(encoding="utf-8")
    )
    phase42_rows = {
        int(row["training_seed"]): row
        for row in phase42_reference["rows"]
        if row["policy"] == "phase42_dqn"
    }
    routing = raw["routing"]
    base_energy = RoutingEnvConfig(
        energy_capacity_j=experiment.energy.satellite_capacity_j,
        tx_electronics_j_per_bit=experiment.energy.tx_electronics_j_per_bit,
        rx_electronics_j_per_bit=experiment.energy.rx_electronics_j_per_bit,
        amplifier_j_per_bit_km2=experiment.energy.tx_amplifier_j_per_bit_km2,
        transmission_rate_bps=float(routing["transmission_rate_bps"]),
        packet_bits=experiment.traffic.flows[0].packet_bits,
        max_neighbors=8,
        max_hops=int(routing["max_hops"]),
        wait_action_enabled=True,
    )
    packet_config = PacketSimulationConfig(
        horizon_seconds=experiment.simulation.duration_seconds,
        max_hops=int(routing["max_hops"]),
        processing_delay_seconds=float(routing["processing_delay_seconds"]),
        processing_energy_j_per_hop=float(routing["processing_energy_j_per_hop"]),
        dqn_wait_action_enabled=True,
    )
    rows: list[dict] = []
    decisions: list[dict] = []
    scenarios = [
        (
            "frozen_satellite_trace",
            frozen_trace,
            frozen_flows,
            {
                "trace_sha256": frozen_trace_hash,
                "schedule_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            {
                                "source": flow.source_satellite,
                                "destination": flow.destination,
                                "times": flow.generation_times_seconds,
                            }
                            for flow in frozen_flows
                        ],
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "topology_seed": None,
            },
        )
    ]
    for topology_seed in UNSEEN_SEEDS:
        trace, flows, metadata = _unseen_scenario(topology_seed)
        scenarios.append(
            (
                f"unseen_graph_{topology_seed}",
                trace,
                flows,
                {
                    **metadata,
                    "trace_sha256": _trace_hash(trace),
                },
            )
        )

    scenario_metadata = {}
    for scenario_name, trace, flows, metadata in scenarios:
        scenario_metadata[scenario_name] = metadata
        for seed in SEEDS:
            for label, policy, checkpoint_hash, path_history, penalty in _load_seeded_policies(
                original_dir,
                phase41_dir,
                phase42_dir,
                history_dir,
                nstep_dir,
                seed,
            ):
                policy_energy = replace(
                    base_energy,
                    include_path_history=path_history,
                )
                policy_config = replace(
                    packet_config,
                    dqn_wait_action_enabled=(
                        scenario_name == "frozen_satellite_trace"
                    ),
                    cycle_avoidance_penalty=penalty,
                )
                row, decision_rows = _run_policy(
                    label=label,
                    trace=trace,
                    flows=flows,
                    energy=policy_energy,
                    config=policy_config,
                    policy=policy,
                    policy_kind="dqn",
                    scenario=scenario_name,
                    training_seed=seed,
                    checkpoint_sha256=checkpoint_hash,
                )
                rows.append(row)
                decisions.extend(decision_rows)

            for strategy in ("random", "greedy"):
                reference = _ReferenceCandidatePolicy(
                    strategy,
                    seed=seed + (topology_seed if scenario_name != "frozen_satellite_trace" else 0),
                )
                row, decision_rows = _run_policy(
                    label=f"{strategy}_feasible" if strategy == "random" else "greedy_snapshot_progress",
                    trace=trace,
                    flows=flows,
                    energy=base_energy,
                    config=replace(
                        packet_config,
                        dqn_wait_action_enabled=(
                            scenario_name == "frozen_satellite_trace"
                        ),
                    ),
                    policy=reference,
                    policy_kind="dqn",
                    scenario=scenario_name,
                    training_seed=seed,
                )
                rows.append(row)
                decisions.extend(decision_rows)

        contact_row, _ = _run_policy(
            label="contact_aware",
            trace=trace,
            flows=flows,
            energy=base_energy,
            config=replace(packet_config, dqn_wait_action_enabled=False),
            policy=None,
            policy_kind="contact",
            scenario=scenario_name,
        )
        rows.append(contact_row)
        if scenario_name == "frozen_satellite_trace":
            frozen_contact = next(
                row for row in baseline_summary["results"] if row["algorithm"] == "contact"
            )
            if (
                contact_row["delivered_packets"] != int(frozen_contact["delivered_packets"])
                or abs(
                    contact_row["total_communication_energy_j"]
                    - float(frozen_contact["total_communication_energy_j"])
                )
                > 1e-9
            ):
                raise AssertionError("Re-evaluated frozen contact baseline differs from frozen results")
            for row in rows:
                if (
                    row["scenario"] != scenario_name
                    or row["policy"] != "phase42_dqn"
                ):
                    continue
                previous = phase42_rows[int(row["training_seed"])]
                for metric in (
                    "generated_packets",
                    "delivered_packets",
                    "dropped_packets",
                    "dropped_by_reason",
                    "packets_with_forwarding_cycles",
                    "total_cycle_revisits",
                ):
                    if row[metric] != previous[metric]:
                        raise AssertionError(
                            f"Unchanged Phase 4.2 policy changed {metric} for seed "
                            f"{row['training_seed']}"
                        )
                for metric in (
                    "packet_delivery_ratio",
                    "average_end_to_end_delay_seconds",
                    "average_hop_count",
                    "total_communication_energy_j",
                ):
                    current = row[metric]
                    previous_value = previous[metric]
                    if current is None or previous_value is None:
                        if current is not previous_value:
                            raise AssertionError(
                                f"Unchanged Phase 4.2 policy changed {metric} for seed "
                                f"{row['training_seed']}"
                            )
                    elif abs(current - previous_value) > 1e-9:
                        raise AssertionError(
                            f"Unchanged Phase 4.2 policy changed {metric} for seed "
                            f"{row['training_seed']}"
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline_config_sha256": hashlib.sha256(
            baseline_config_path.read_bytes()
        ).hexdigest(),
        "frozen_trace_sha256": frozen_trace_hash,
        "baseline_runs_sha256": hashlib.sha256(
            (baseline_results_dir / "baseline_runs.csv").read_bytes()
        ).hexdigest(),
        "phase43_training_config_sha256": hashlib.sha256(
            (ROOT / "experiments/dqn_training_phase4_3_history.toml").read_bytes()
        ).hexdigest(),
        "phase43_n_step_training_config_sha256": hashlib.sha256(
            (ROOT / "experiments/dqn_training_phase4_3_nstep.toml").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "experiments/train_dqn.py",
                ROOT / "src/fedroute/packet_sim.py",
                ROOT / "src/fedroute/routing.py",
                ROOT / "src/fedroute/dqn.py",
            )
        },
        "training_seeds": list(SEEDS),
        "unseen_topology_seeds": list(UNSEEN_SEEDS),
        "cycle_avoidance_penalty": CYCLE_PENALTY,
        "evaluation_config": {
            "frozen_trace": {
                "config_path": str(baseline_config_path.relative_to(ROOT)),
                "duration_seconds": experiment.simulation.duration_seconds,
                "max_hops": int(routing["max_hops"]),
                "processing_delay_seconds": float(
                    routing["processing_delay_seconds"]
                ),
                "processing_energy_j_per_hop": float(
                    routing["processing_energy_j_per_hop"]
                ),
                "wait_action_enabled": True,
            },
            "unseen_graphs": {
                "satellite_count": 14,
                "topology_construction": (
                    "random spanning tree plus independent satellite links"
                ),
                "extra_satellite_link_probability": 0.11,
                "sample_count": 101,
                "sample_interval_seconds": 60,
                "ground_station_count": 1,
                "flow_count": 4,
                "packets_per_flow": 5,
                "packet_size_bits": 4096,
                "deadline_seconds": 1200.0,
                "generation_time_range_seconds": [0.0, 5000.0],
                "traffic_seed_rule": "topology_seed XOR 0x5A17",
                "wait_action_enabled": False,
            },
            "controls": [
                "original_phase4_dqn",
                "phase41_dqn",
                "phase42_dqn",
                "phase42_dqn_soft_cycle_penalty",
                "phase43_dqn_path_history",
                "phase43_dqn_n_step_3",
                "random_feasible",
                "greedy_snapshot_progress",
                "contact_aware",
            ],
            "repeatability_check": "Each policy evaluation is rerun and compared",
            "reported_metrics": [
                "packet_delivery_ratio",
                "delivered_packets",
                "dropped_by_reason",
                "packets_with_forwarding_cycles",
                "total_cycle_revisits",
                "average_hop_count",
                "average_end_to_end_delay_seconds",
                "total_communication_energy_j",
            ],
        },
        "scenarios": scenario_metadata,
        "rows": rows,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "evaluation_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        fieldnames = (
            "scenario",
            "policy",
            "training_seed",
            "generated_packets",
            "delivered_packets",
            "dropped_packets",
            "packet_delivery_ratio",
            "average_end_to_end_delay_seconds",
            "average_hop_count",
            "total_communication_energy_j",
            "dropped_by_reason",
            "packets_with_forwarding_cycles",
            "total_cycle_revisits",
            "max_cycle_revisits_per_packet",
            "cycle_avoidance_penalty",
            "checkpoint_sha256",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames},
                    "dropped_by_reason": json.dumps(
                        row["dropped_by_reason"], sort_keys=True
                    ),
                }
            )
    (output_dir / "evaluation_decisions.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("experiments/baseline_config.toml"),
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("results/phase4-baseline"),
    )
    parser.add_argument("--original-dir", type=Path, default=Path("results/phase4-dqn"))
    parser.add_argument(
        "--phase41-dir",
        type=Path,
        default=Path("results/phase4-dqn-diagnostics/phase4-1-final"),
    )
    parser.add_argument("--phase42-dir", type=Path, default=Path("results/phase4-2"))
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("results/phase4-3/history-dqn-final"),
    )
    parser.add_argument(
        "--nstep-dir",
        type=Path,
        default=Path("results/phase4-3/nstep-dqn"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4-3"))
    args = parser.parse_args()
    result = evaluate(
        baseline_config_path=args.baseline_config,
        baseline_results_dir=args.baseline_results,
        original_dir=args.original_dir,
        phase41_dir=args.phase41_dir,
        phase42_dir=args.phase42_dir,
        history_dir=args.history_dir,
        nstep_dir=args.nstep_dir,
        output_dir=args.output_dir,
    )
    for scenario in result["scenarios"]:
        scenario_rows = [row for row in result["rows"] if row["scenario"] == scenario]
        print(scenario, len(scenario_rows), "policy rows")
    print(f"Saved Phase 4.3 comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
