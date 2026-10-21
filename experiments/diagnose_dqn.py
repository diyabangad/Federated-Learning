"""Record failed legacy DQN packet trajectories without changing frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import torch

from fedroute.config import ExperimentConfig
from fedroute.dqn import DQNPolicy
from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace, build_topology_trace


def _trace_data(trace: TopologyTrace) -> dict:
    return {
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


def _trace_hash(trace: TopologyTrace) -> str:
    encoded = json.dumps(_trace_data(trace), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _flows(raw: dict, experiment: ExperimentConfig) -> tuple[PacketFlow, ...]:
    schedule = raw["packet_schedule"]
    interval = float(schedule["generation_interval_seconds"])
    first = float(schedule["first_generation_seconds"])
    flows = []
    for flow in experiment.traffic.flows:
        times = []
        generation_time = first
        while generation_time < experiment.simulation.duration_seconds:
            times.append(generation_time)
            generation_time += interval
        flows.append(
            PacketFlow(
                source_satellite=flow.source,
                destination=flow.destination,
                generation_times_seconds=tuple(times),
                packet_size_bits=flow.packet_bits,
                deadline_seconds=flow.deadline_seconds,
            )
        )
    return tuple(flows)


def diagnose(
    baseline_config_path: Path,
    baseline_results_dir: Path,
    checkpoint_dir: Path,
    output_path: Path,
    seeds: tuple[int, ...],
) -> dict:
    baseline_config_path = baseline_config_path.resolve()
    baseline_results_dir = baseline_results_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    output_path = output_path.resolve()
    raw = tomllib.loads(baseline_config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.from_toml(baseline_config_path)
    trace = build_topology_trace(experiment.simulation, experiment.topology)
    summary = json.loads((baseline_results_dir / "baseline_summary.json").read_text())
    trace_hash = _trace_hash(trace)
    if trace_hash != summary["topology_trace_sha256"]:
        raise ValueError("Rebuilt diagnostic trace differs from the frozen baseline")
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
    packet_config = PacketSimulationConfig(
        horizon_seconds=experiment.simulation.duration_seconds,
        max_hops=int(routing["max_hops"]),
        processing_delay_seconds=float(routing["processing_delay_seconds"]),
        processing_energy_j_per_hop=float(routing["processing_energy_j_per_hop"]),
    )
    runs = []
    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"dqn_seed_{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        policy = DQNPolicy(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )
        policy.load_state_dict(checkpoint["state_dict"])
        policy.eval()
        simulator = PacketSimulator(
            trace,
            energy,
            packet_config,
            dqn_policy=policy,
            record_dqn_decisions=True,
        )
        result = simulator.run(_flows(raw, experiment), "dqn")
        runs.append(
            {
                "training_seed": seed,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "summary": {
                    "generated_packets": result.generated_packets,
                    "delivered_packets": result.delivered_packets,
                    "dropped_by_reason": dict(result.dropped_by_reason),
                    "total_communication_energy_j": result.total_communication_energy_j,
                },
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
                "decisions": simulator.dqn_decision_trace,
            }
        )
    report = {
        "baseline_config_sha256": hashlib.sha256(baseline_config_path.read_bytes()).hexdigest(),
        "evaluation_trace_sha256": trace_hash,
        "baseline_results_sha256": hashlib.sha256(
            (baseline_results_dir / "baseline_runs.csv").read_bytes()
        ).hexdigest(),
        "policy_semantics": "legacy DQN with available outgoing links plus DROP; no WAIT",
        "epsilon_and_reward": (
            "epsilon is null because these are greedy packet evaluations; packet simulator "
            "does not assign RL rewards. Training-time rewards are in dqn_training_decisions.json."
        ),
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/phase4-dqn"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase4-dqn-diagnostics/legacy_failed_trajectories.json"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23))
    args = parser.parse_args()
    report = diagnose(
        args.baseline_config,
        args.baseline_results,
        args.checkpoint_dir,
        args.output,
        tuple(args.seeds),
    )
    for run in report["runs"]:
        print(
            f"seed {run['training_seed']}: "
            f"{run['summary']['delivered_packets']}/"
            f"{run['summary']['generated_packets']} delivered; "
            f"drops={run['summary']['dropped_by_reason']}"
        )
    print(f"Saved decision traces to {args.output}")


if __name__ == "__main__":
    main()
