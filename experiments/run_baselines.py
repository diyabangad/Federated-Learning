"""Run deterministic routing baselines against one frozen packet scenario."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import tomllib

from fedroute.config import ExperimentConfig
from fedroute.dqn import DQNPolicy
from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace, build_topology_trace
import torch

POLICIES = ("hops", "delay", "energy", "contact")


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


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


def _trace_hash(trace_data: dict) -> str:
    encoded = json.dumps(
        trace_data,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fixed_flows(raw: dict, experiment: ExperimentConfig) -> tuple[PacketFlow, ...]:
    schedule = raw.get("packet_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("[packet_schedule] with fixed generation times is required")
    interval = float(schedule["generation_interval_seconds"])
    first = float(schedule["first_generation_seconds"])
    if interval <= 0.0 or first < 0.0:
        raise ValueError("Packet generation interval must be positive and start non-negative")
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


def _result_row(policy: str, result, runtime_seconds: float, seed: int) -> dict:
    return {
        "algorithm": policy,
        "seed": seed,
        "generated_packets": result.generated_packets,
        "delivered_packets": result.delivered_packets,
        "dropped_packets": result.dropped_packets,
        "packet_delivery_ratio": result.packet_delivery_ratio,
        "average_end_to_end_delay_seconds": result.average_end_to_end_delay_seconds,
        "average_propagation_delay_seconds": result.average_propagation_delay_seconds,
        "average_transmission_delay_seconds": result.average_transmission_delay_seconds,
        "average_processing_delay_seconds": result.average_processing_delay_seconds,
        "average_waiting_delay_seconds": result.average_waiting_delay_seconds,
        "average_hop_count": result.average_hop_count,
        "total_communication_energy_j": result.total_communication_energy_j,
        "total_compute_energy_j": result.total_compute_energy_j,
        "dropped_by_reason_json": json.dumps(dict(result.dropped_by_reason), sort_keys=True),
        "runtime_seconds": runtime_seconds,
    }


def run_baselines(
    config_path: Path,
    output_dir: Path,
    dqn_checkpoint: Path | None = None,
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = ExperimentConfig.from_toml(config_path)
    schedule_flows = _fixed_flows(raw, experiment)
    routing = raw["routing"]
    schedule = raw["packet_schedule"]
    run_seed = experiment.run_seeds[0]
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
    dqn_policy = None
    if dqn_checkpoint is not None:
        dqn_checkpoint = dqn_checkpoint.resolve()
        checkpoint = torch.load(dqn_checkpoint, map_location="cpu", weights_only=True)
        dqn_policy = DQNPolicy(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )
        dqn_policy.load_state_dict(checkpoint["state_dict"])
        dqn_policy.eval()
    policies = POLICIES + (("dqn",) if dqn_policy is not None else ())

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_start = perf_counter()
    trace = build_topology_trace(experiment.simulation, experiment.topology)
    topology_build_seconds = perf_counter() - trace_start
    topology_data = _trace_data(trace)
    topology_sha256 = _trace_hash(topology_data)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    trace_file = output_dir / "baseline_topology.json"
    trace_file.write_text(
        json.dumps(
            {
                "config_path": str(config_path.relative_to(project_root)),
                "config_sha256": config_sha256,
                "trace_sha256": topology_sha256,
                "trace": topology_data,
            },
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = []
    deterministic_repeats = {}
    for policy in policies:
        simulator = PacketSimulator(
            trace,
            energy,
            packet_config,
            dqn_policy=dqn_policy if policy == "dqn" else None,
        )
        started = perf_counter()
        result = simulator.run(schedule_flows, policy)
        runtime_seconds = perf_counter() - started
        repeated = PacketSimulator(
            trace,
            energy,
            packet_config,
            dqn_policy=dqn_policy if policy == "dqn" else None,
        ).run(schedule_flows, policy)
        if result != repeated:
            raise AssertionError(f"Policy {policy} was not deterministic for frozen inputs")
        deterministic_repeats[policy] = True
        row = _result_row(policy, result, runtime_seconds, run_seed)
        rows.append(row)

    csv_path = output_dir / "baseline_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        working_tree_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_revision, working_tree_dirty = None, None
    source_paths = (
        Path("pyproject.toml"),
        Path("experiments/baseline_config.toml"),
        Path("experiments/run_baselines.py"),
        Path("src/fedroute/config.py"),
        Path("src/fedroute/topology.py"),
        Path("src/fedroute/routing.py"),
        Path("src/fedroute/baselines.py"),
        Path("src/fedroute/packet_sim.py"),
    )

    summary = {
        "experiment_name": experiment.name,
        "parameter_status": experiment.parameter_status,
        "parameter_notes": experiment.parameter_notes,
        "config_file": str(config_path.relative_to(project_root)),
        "config_sha256": config_sha256,
        "topology_trace_sha256": topology_sha256,
        "topology_trace_file": str(trace_file.relative_to(project_root)),
        "topology_build_runtime_seconds": topology_build_seconds,
        "git_revision": git_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_file_sha256": {
            str(path): hashlib.sha256((project_root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in ("networkx", "numpy", "skyfield", "sgp4", "torch")
        },
        "run_seed": run_seed,
        "packet_schedule": schedule,
        "flow_definitions": [asdict(flow) for flow in schedule_flows],
        "routing_energy_config": asdict(energy),
        "packet_simulation_config": asdict(packet_config),
        "dqn_checkpoint": str(dqn_checkpoint) if dqn_checkpoint is not None else None,
        "dqn_checkpoint_sha256": (
            hashlib.sha256(dqn_checkpoint.read_bytes()).hexdigest()
            if dqn_checkpoint is not None
            else None
        ),
        "policy_repeats_deterministic": deterministic_repeats,
        "results_csv": str(csv_path.relative_to(project_root)),
        "results_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "results": rows,
        "limitations": [
            "All physical and energy parameters are provisional, not calibrated.",
            "Traffic is deterministic periodic traffic with one configured flow.",
            "Satellite communication and compute energy are reported; FL costs are separate.",
            "Packet simulation uses one half-duplex radio per endpoint and no finite queues or channel loss.",
        ],
    }
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/baseline_config.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4-baseline"),
    )
    parser.add_argument("--dqn-checkpoint", type=Path)
    args = parser.parse_args()
    summary = run_baselines(args.config, args.output_dir, args.dqn_checkpoint)
    for row in summary["results"]:
        print(
            f"{row['algorithm']}: delivered={row['delivered_packets']}/"
            f"{row['generated_packets']} pdr={row['packet_delivery_ratio']} "
            f"avg_delay_s={row['average_end_to_end_delay_seconds']} "
            f"energy_j={row['total_communication_energy_j']:.6f} "
            f"hops={row['average_hop_count']}"
        )
    print(f"Saved baseline metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
