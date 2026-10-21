"""Compare Phase 4 DQN checkpoints and simple policies on the frozen trace."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import torch
from torch import nn

from fedroute.config import ExperimentConfig
from fedroute.dqn import DQNPolicy
from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace, build_topology_trace


def _trace_hash(trace: TopologyTrace) -> str:
    trace_data = {
        "times_utc": [time.isoformat() for time in trace.times_utc],
        "graphs": [
            {
                "time_utc": trace.times_utc[index].isoformat(),
                "nodes": [
                    {"id": node, **attributes}
                    for node, attributes in sorted(graph.nodes(data=True))
                ],
                "edges": [
                    {"source": node_a, "target": node_b, **attributes}
                    for node_a, node_b, attributes in sorted(
                        graph.edges(data=True),
                        key=lambda edge: tuple(sorted((edge[0], edge[1]))),
                    )
                ],
            }
            for index, graph in enumerate(trace.graphs)
        ],
        "contact_windows": [
            {
                "node_a": window.node_a,
                "node_b": window.node_b,
                "start_utc": window.start_utc.isoformat(),
                "end_utc": window.end_utc.isoformat(),
                "start_censored": window.start_censored,
                "end_censored": window.end_censored,
                "start_uncertainty_seconds": window.start_uncertainty_seconds,
                "end_uncertainty_seconds": window.end_uncertainty_seconds,
            }
            for window in trace.contact_windows
        ],
    }
    encoded = json.dumps(trace_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fixed_flows(raw: dict, experiment: ExperimentConfig) -> tuple[PacketFlow, ...]:
    schedule = raw["packet_schedule"]
    interval = float(schedule["generation_interval_seconds"])
    first = float(schedule["first_generation_seconds"])
    flows = []
    for flow in experiment.traffic.flows:
        generation_times = []
        generation_time = first
        while generation_time < experiment.simulation.duration_seconds:
            generation_times.append(generation_time)
            generation_time += interval
        flows.append(
            PacketFlow(
                source_satellite=flow.source,
                destination=flow.destination,
                generation_times_seconds=tuple(generation_times),
                packet_size_bits=flow.packet_bits,
                deadline_seconds=flow.deadline_seconds,
            )
        )
    return tuple(flows)


class _ReferenceCandidatePolicy(nn.Module):
    def __init__(self, strategy: str, seed: int = 0):
        super().__init__()
        if strategy not in {"random", "greedy"}:
            raise ValueError(f"Unsupported reference strategy {strategy!r}")
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def forward(self, candidates):
        features = torch.as_tensor(np.asarray(candidates), dtype=torch.float32)
        if self.strategy == "random":
            values = torch.as_tensor(
                self.rng.random(features.shape[-2]),
                dtype=features.dtype,
            )
        else:
            reachable = features[:, 8] > 0.5
            progress = 1.0 - features[:, 6]
            quality = features[:, 3]
            values = torch.where(reachable, 2.0 + progress, quality - 2.0)
        return torch.cat(
            (values, torch.full((1,), -100.0, dtype=features.dtype))
        )


def _load_policy(path: Path) -> tuple[DQNPolicy, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = DQNPolicy(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, hashlib.sha256(path.read_bytes()).hexdigest()


def _packet_cycle_metrics(result) -> dict:
    packet_revisits = []
    for packet in result.packets:
        visited = set()
        revisits = 0
        for node in packet.route:
            if node in visited:
                revisits += 1
            visited.add(node)
        packet_revisits.append(revisits)
    return {
        "packets_with_forwarding_cycles": sum(value > 0 for value in packet_revisits),
        "total_cycle_revisits": sum(packet_revisits),
        "max_cycle_revisits_per_packet": max(packet_revisits, default=0),
    }


def _evaluate(
    trace,
    flows,
    energy: RoutingEnvConfig,
    packet_config: PacketSimulationConfig,
    label: str,
    policy,
    policy_kind: str,
    *,
    training_seed: int | None = None,
    checkpoint_sha256: str | None = None,
) -> dict:
    simulator = PacketSimulator(
        trace,
        energy,
        packet_config,
        dqn_policy=policy,
        record_dqn_decisions=True,
    )
    repeated_policy = deepcopy(policy)
    result = simulator.run(flows, policy_kind)
    repeated = PacketSimulator(
        trace,
        energy,
        packet_config,
        dqn_policy=repeated_policy,
    ).run(flows, policy_kind)
    if result != repeated:
        raise AssertionError(f"Repeated policy evaluation was not deterministic: {label}")
    drop_decisions = [
        decision
        for decision in simulator.dqn_decision_trace
        if decision["selected_action"] == "DROP"
    ]
    premature_drops = sum(
        any(decision["action_mask"][:-1])
        for decision in drop_decisions
    )
    return {
        "policy": label,
        "training_seed": training_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "generated_packets": result.generated_packets,
        "delivered_packets": result.delivered_packets,
        "dropped_packets": result.dropped_packets,
        "packet_delivery_ratio": result.packet_delivery_ratio,
        "average_end_to_end_delay_seconds": result.average_end_to_end_delay_seconds,
        "average_hop_count": result.average_hop_count,
        "total_communication_energy_j": result.total_communication_energy_j,
        "dropped_by_reason": dict(result.dropped_by_reason),
        "premature_drop_decisions": premature_drops,
        "decision_count": len(simulator.dqn_decision_trace),
        **_packet_cycle_metrics(result),
        "packets": [
            {
                "packet_id": packet.packet_id,
                "status": packet.status,
                "drop_reason": packet.drop_reason,
                "route": packet.route,
                "hop_count": packet.hop_count,
                "delay_seconds": packet.accumulated_delay_seconds,
            }
            for packet in result.packets
        ],
    }


def evaluate(
    baseline_config_path: Path,
    baseline_results_dir: Path,
    original_checkpoint_dir: Path,
    phase41_checkpoint_dir: Path,
    phase42_checkpoint_dir: Path,
    output_dir: Path,
) -> dict:
    raw = tomllib.loads(baseline_config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.from_toml(baseline_config_path)
    baseline_summary = json.loads(
        (baseline_results_dir / "baseline_summary.json").read_text(encoding="utf-8")
    )
    trace = build_topology_trace(experiment.simulation, experiment.topology)
    trace_hash = _trace_hash(trace)
    if trace_hash != baseline_summary["topology_trace_sha256"]:
        raise ValueError("Policy comparison trace differs from frozen baseline")
    flows = _fixed_flows(raw, experiment)
    routing = raw["routing"]
    energy = RoutingEnvConfig(
        energy_capacity_j=experiment.energy.satellite_capacity_j,
        tx_electronics_j_per_bit=experiment.energy.tx_electronics_j_per_bit,
        rx_electronics_j_per_bit=experiment.energy.rx_electronics_j_per_bit,
        amplifier_j_per_bit_km2=experiment.energy.tx_amplifier_j_per_bit_km2,
        transmission_rate_bps=float(routing["transmission_rate_bps"]),
        packet_bits=experiment.traffic.flows[0].packet_bits,
        max_hops=int(routing["max_hops"]),
    )
    common_packet_config = PacketSimulationConfig(
        horizon_seconds=experiment.simulation.duration_seconds,
        max_hops=int(routing["max_hops"]),
        processing_delay_seconds=float(routing["processing_delay_seconds"]),
        processing_energy_j_per_hop=float(routing["processing_energy_j_per_hop"]),
    )
    rows = []

    for label, checkpoint_dir, wait_enabled in (
        ("legacy_phase4_dqn", original_checkpoint_dir, False),
        ("phase41_dqn", phase41_checkpoint_dir, True),
        ("phase42_dqn", phase42_checkpoint_dir, True),
    ):
        packet_config = PacketSimulationConfig(
            horizon_seconds=common_packet_config.horizon_seconds,
            max_hops=common_packet_config.max_hops,
            processing_delay_seconds=common_packet_config.processing_delay_seconds,
            processing_energy_j_per_hop=common_packet_config.processing_energy_j_per_hop,
            dqn_wait_action_enabled=wait_enabled,
        )
        for seed in (11, 23, 37, 53, 71):
            policy, checkpoint_hash = _load_policy(
                checkpoint_dir / f"dqn_seed_{seed}.pt"
            )
            rows.append(
                _evaluate(
                    trace,
                    flows,
                    energy,
                    packet_config,
                    label,
                    policy,
                    "dqn",
                    training_seed=seed,
                    checkpoint_sha256=checkpoint_hash,
                )
            )

    wait_packet_config = PacketSimulationConfig(
        horizon_seconds=common_packet_config.horizon_seconds,
        max_hops=common_packet_config.max_hops,
        processing_delay_seconds=common_packet_config.processing_delay_seconds,
        processing_energy_j_per_hop=common_packet_config.processing_energy_j_per_hop,
        dqn_wait_action_enabled=True,
    )
    for seed in (11, 23, 37, 53, 71):
        rows.append(
            _evaluate(
                trace,
                flows,
                energy,
                wait_packet_config,
                "random_feasible",
                _ReferenceCandidatePolicy("random", seed),
                "dqn",
                training_seed=seed,
            )
        )
    rows.append(
        _evaluate(
            trace,
            flows,
            energy,
            wait_packet_config,
            "greedy_snapshot_progress",
            _ReferenceCandidatePolicy("greedy"),
            "dqn",
        )
    )

    contact_result = PacketSimulator(trace, energy, common_packet_config).run(
        flows,
        "contact",
    )
    frozen_contact = next(
        row for row in baseline_summary["results"] if row["algorithm"] == "contact"
    )
    if (
        contact_result.delivered_packets != int(frozen_contact["delivered_packets"])
        or contact_result.total_communication_energy_j
        != float(frozen_contact["total_communication_energy_j"])
    ):
        raise AssertionError("Contact-aware comparator differs from frozen baseline")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline_config_sha256": hashlib.sha256(
            baseline_config_path.read_bytes()
        ).hexdigest(),
        "evaluation_trace_sha256": trace_hash,
        "baseline_runs_csv_sha256": hashlib.sha256(
            (baseline_results_dir / "baseline_runs.csv").read_bytes()
        ).hexdigest(),
        "rows": rows,
        "frozen_contact_baseline": {
            "delivered_packets": contact_result.delivered_packets,
            "generated_packets": contact_result.generated_packets,
            "packet_delivery_ratio": contact_result.packet_delivery_ratio,
            "average_end_to_end_delay_seconds": (
                contact_result.average_end_to_end_delay_seconds
            ),
            "average_hop_count": contact_result.average_hop_count,
            "total_communication_energy_j": (
                contact_result.total_communication_energy_j
            ),
            "dropped_by_reason": dict(contact_result.dropped_by_reason),
        },
    }
    (output_dir / "policy_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "policy_comparison_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        fieldnames = (
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
            "premature_drop_decisions",
            "decision_count",
            "packets_with_forwarding_cycles",
            "total_cycle_revisits",
            "max_cycle_revisits_per_packet",
            "checkpoint_sha256",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fieldnames},
                    "dropped_by_reason": json.dumps(row["dropped_by_reason"], sort_keys=True),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("experiments/baseline_config.toml"))
    parser.add_argument("--baseline-results", type=Path, default=Path("results/phase4-baseline"))
    parser.add_argument("--original-checkpoints", type=Path, default=Path("results/phase4-dqn"))
    parser.add_argument(
        "--phase41-checkpoints",
        type=Path,
        default=Path("results/phase4-dqn-diagnostics/phase4-1-final"),
    )
    parser.add_argument("--phase42-checkpoints", type=Path, default=Path("results/phase4-2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4-2"))
    args = parser.parse_args()
    result = evaluate(
        args.baseline_config,
        args.baseline_results,
        args.original_checkpoints,
        args.phase41_checkpoints,
        args.phase42_checkpoints,
        args.output_dir,
    )
    for row in result["rows"]:
        print(
            row["policy"],
            row["training_seed"],
            f"{row['delivered_packets']}/{row['generated_packets']}",
            row["packet_delivery_ratio"],
            row["dropped_by_reason"],
        )
    print(f"Saved policy comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
