"""Train standalone DQN policies and evaluate them on the frozen baseline scenario."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tomllib

from skyfield.api import load
import torch

from fedroute.config import ExperimentConfig
from fedroute.dqn import DQNPolicy, DQNTrainer, DQNTrainingConfig
from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.routing import RoutingEnvConfig, SatelliteRoutingEnv
from fedroute.topology import TopologyTrace, build_topology_trace


METRICS = (
    "packet_delivery_ratio",
    "average_end_to_end_delay_seconds",
    "average_propagation_delay_seconds",
    "average_transmission_delay_seconds",
    "average_processing_delay_seconds",
    "average_waiting_delay_seconds",
    "average_hop_count",
    "total_communication_energy_j",
    "total_compute_energy_j",
)


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


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty results to {path}")
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_trace(experiment: ExperimentConfig, training_epoch: datetime) -> TopologyTrace:
    if experiment.topology.walker is None:
        raise ValueError("The initial DQN training runner currently requires Walker elements")
    walker = replace(experiment.topology.walker, epoch_utc=training_epoch)
    topology = replace(experiment.topology, walker=walker)
    simulation = replace(experiment.simulation, start_utc=training_epoch)
    return build_topology_trace(simulation, topology)


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "sample_std": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def train_and_evaluate(
    baseline_config_path: Path,
    training_config_path: Path,
    output_dir: Path,
    baseline_results_dir: Path,
) -> dict:
    baseline_config_path = baseline_config_path.resolve()
    training_config_path = training_config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = tomllib.loads(training_config_path.read_text(encoding="utf-8"))
    training_settings = training_config["training"]
    reward_settings = training_config["reward"]
    protocol_settings = training_config.get("protocol", {})
    experiment = ExperimentConfig.from_toml(baseline_config_path)

    baseline_summary_path = baseline_results_dir / "baseline_summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_trace_path = baseline_results_dir / "baseline_topology.json"
    baseline_trace_payload = json.loads(baseline_trace_path.read_text(encoding="utf-8"))
    if baseline_trace_payload["config_sha256"] != hashlib.sha256(
        baseline_config_path.read_bytes()
    ).hexdigest():
        raise ValueError("Saved baseline topology does not match the frozen config")

    evaluation_trace = build_topology_trace(experiment.simulation, experiment.topology)
    evaluation_trace_hash = _trace_hash(evaluation_trace)
    if evaluation_trace_hash != baseline_summary["topology_trace_sha256"]:
        raise ValueError("Rebuilt DQN evaluation topology differs from the frozen baseline trace")
    flows = _fixed_flows(tomllib.loads(baseline_config_path.read_text()), experiment)
    routing_config = tomllib.loads(baseline_config_path.read_text())["routing"]
    energy = RoutingEnvConfig(
        energy_capacity_j=experiment.energy.satellite_capacity_j,
        tx_electronics_j_per_bit=experiment.energy.tx_electronics_j_per_bit,
        rx_electronics_j_per_bit=experiment.energy.rx_electronics_j_per_bit,
        amplifier_j_per_bit_km2=experiment.energy.tx_amplifier_j_per_bit_km2,
        transmission_rate_bps=float(routing_config["transmission_rate_bps"]),
        packet_bits=experiment.traffic.flows[0].packet_bits,
        max_hops=int(routing_config["max_hops"]),
        delivery_reward=float(reward_settings["delivery_reward"]),
        drop_penalty=float(reward_settings["drop_penalty"]),
        energy_penalty=float(reward_settings["energy_penalty"]),
        delay_penalty=float(reward_settings["delay_penalty"]),
        wait_action_enabled=bool(protocol_settings.get("policy_wait_action", False)),
        normalize_wait_penalty_by_horizon=bool(
            reward_settings.get("normalize_wait_penalty_by_horizon", False)
        ),
        include_path_history=bool(
            protocol_settings.get("include_path_history", False)
        ),
    )
    packet_config = PacketSimulationConfig(
        horizon_seconds=experiment.simulation.duration_seconds,
        max_hops=int(routing_config["max_hops"]),
        processing_delay_seconds=float(routing_config["processing_delay_seconds"]),
        processing_energy_j_per_hop=float(routing_config["processing_energy_j_per_hop"]),
        dqn_wait_action_enabled=energy.wait_action_enabled,
    )
    training_epoch = datetime.fromisoformat(
        training_settings["training_epoch_utc"].replace("Z", "+00:00")
    )
    training_trace = _training_trace(experiment, training_epoch)
    training_start_indices = tuple(range(len(training_trace.graphs)))
    training_episode_rows = []
    training_decision_rows = []
    evaluation_rows = []
    evaluation_decision_rows = []
    evaluation_packet_outcome_rows = []
    wait_ablation_rows = []
    per_seed_summary = []

    for seed in training_settings["training_seeds"]:
        config = DQNTrainingConfig(
            seed=int(seed),
            episodes=int(training_settings["episodes"]),
            max_steps_per_episode=int(training_settings["max_steps_per_episode"]),
            batch_size=int(training_settings["batch_size"]),
            replay_capacity=int(training_settings["replay_capacity"]),
            warmup_transitions=int(training_settings["warmup_transitions"]),
            target_update_interval=int(training_settings["target_update_interval"]),
            learning_rate=float(training_settings["learning_rate"]),
            gamma=float(training_settings["gamma"]),
            epsilon_start=float(training_settings["epsilon_start"]),
            epsilon_end=float(training_settings["epsilon_end"]),
            hidden_dim=int(training_settings["hidden_dim"]),
            n_step=int(training_settings.get("n_step", 1)),
        )
        torch.manual_seed(config.seed)
        input_dim = 9 + int(energy.include_path_history)
        policy = DQNPolicy(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
        )
        trainer = DQNTrainer(policy, config)
        train_env = SatelliteRoutingEnv(training_trace, energy)
        def capture_transition(transition: dict) -> None:
            captured_seeds = (
                training_settings["training_seeds"]
                if protocol_settings.get("capture_all_training_decisions", False)
                else training_settings["training_seeds"][:2]
            )
            if int(seed) in tuple(int(value) for value in captured_seeds):
                graph_time = train_env.trace.times_utc[transition["graph_index"]]
                training_decision_rows.append(
                    {
                        "seed": int(seed),
                        "packet_id": f"training-seed-{seed}-episode-{transition['episode']}",
                        "time_seconds": (
                            graph_time - train_env.trace.times_utc[0]
                        ).total_seconds(),
                        "time_utc": graph_time.isoformat(),
                        **transition,
                    }
                )

        episode_history = trainer.train(
            train_env,
            source=experiment.traffic.flows[0].source,
            destination=experiment.traffic.flows[0].destination,
            training_start_indices=training_start_indices,
            transition_callback=capture_transition,
        )
        checkpoint_path = output_dir / f"dqn_seed_{seed}.pt"
        checkpoint = {
            "seed": int(seed),
            "input_dim": input_dim,
            "hidden_dim": config.hidden_dim,
            "state_dict": policy.state_dict(),
            "training_config": asdict(config),
            "reward_config": reward_settings,
            "training_epoch_utc": training_epoch.isoformat(),
            "training_trace_sha256": _trace_hash(training_trace),
            "evaluation_trace_sha256": evaluation_trace_hash,
        }
        torch.save(checkpoint, checkpoint_path)

        for episode in episode_history:
            training_episode_rows.append({"seed": int(seed), **episode})

        evaluation_simulator = PacketSimulator(
            evaluation_trace,
            energy,
            packet_config,
            dqn_policy=policy,
            record_dqn_decisions=True,
        )
        evaluation = evaluation_simulator.run(flows, "dqn")
        evaluation_decision_rows.extend(
            {"training_seed": int(seed), **decision}
            for decision in evaluation_simulator.dqn_decision_trace
        )
        evaluation_packet_outcome_rows.extend(
            {
                "training_seed": int(seed),
                "packet_id": packet.packet_id,
                "source_satellite": packet.source_satellite,
                "destination": packet.destination,
                "status": packet.status,
                "drop_reason": packet.drop_reason,
                "route": packet.route,
                "hop_count": packet.hop_count,
                "delay_seconds": packet.accumulated_delay_seconds,
            }
            for packet in evaluation.packets
        )
        repeated_evaluation = PacketSimulator(
            evaluation_trace,
            energy,
            packet_config,
            dqn_policy=policy,
        ).run(flows, "dqn")
        if evaluation != repeated_evaluation:
            raise AssertionError(f"DQN packet evaluation was nondeterministic for seed {seed}")
        no_wait_config = replace(packet_config, dqn_wait_action_enabled=False)
        no_wait_evaluation = PacketSimulator(
            evaluation_trace,
            energy,
            no_wait_config,
            dqn_policy=policy,
            record_dqn_decisions=True,
        ).run(flows, "dqn")
        repeated_no_wait_evaluation = PacketSimulator(
            evaluation_trace,
            energy,
            no_wait_config,
            dqn_policy=policy,
        ).run(flows, "dqn")
        if no_wait_evaluation != repeated_no_wait_evaluation:
            raise AssertionError(f"DQN no-WAIT ablation was nondeterministic for seed {seed}")
        row = {
            "algorithm": "dqn",
            "training_seed": int(seed),
            "checkpoint": str(checkpoint_path),
            "generated_packets": evaluation.generated_packets,
            "delivered_packets": evaluation.delivered_packets,
            "dropped_packets": evaluation.dropped_packets,
            "packet_delivery_ratio": evaluation.packet_delivery_ratio,
            "average_end_to_end_delay_seconds": evaluation.average_end_to_end_delay_seconds,
            "average_propagation_delay_seconds": evaluation.average_propagation_delay_seconds,
            "average_transmission_delay_seconds": evaluation.average_transmission_delay_seconds,
            "average_processing_delay_seconds": evaluation.average_processing_delay_seconds,
            "average_waiting_delay_seconds": evaluation.average_waiting_delay_seconds,
            "average_hop_count": evaluation.average_hop_count,
            "total_communication_energy_j": evaluation.total_communication_energy_j,
            "total_compute_energy_j": evaluation.total_compute_energy_j,
            "mean_training_return": statistics.mean(item["return"] for item in episode_history),
            "last_20_episode_return": statistics.mean(
                item["return"] for item in episode_history[-20:]
            ),
            "dropped_by_reason_json": json.dumps(
                dict(evaluation.dropped_by_reason),
                sort_keys=True,
            ),
            "evaluation_repeat_deterministic": True,
        }
        evaluation_rows.append(row)
        wait_ablation_rows.append(
            {
                "algorithm": "dqn_no_wait_ablation",
                "training_seed": int(seed),
                "generated_packets": no_wait_evaluation.generated_packets,
                "delivered_packets": no_wait_evaluation.delivered_packets,
                "dropped_packets": no_wait_evaluation.dropped_packets,
                "packet_delivery_ratio": no_wait_evaluation.packet_delivery_ratio,
                "average_end_to_end_delay_seconds": no_wait_evaluation.average_end_to_end_delay_seconds,
                "average_waiting_delay_seconds": no_wait_evaluation.average_waiting_delay_seconds,
                "average_hop_count": no_wait_evaluation.average_hop_count,
                "total_communication_energy_j": no_wait_evaluation.total_communication_energy_j,
                "dropped_by_reason_json": json.dumps(
                    dict(no_wait_evaluation.dropped_by_reason),
                    sort_keys=True,
                ),
                "evaluation_repeat_deterministic": True,
            }
        )
        per_seed_summary.append(
            {
                "training_seed": int(seed),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "training_episodes": len(episode_history),
                "training_transitions": sum(item["transitions"] for item in episode_history),
                "gradient_steps": trainer.gradient_steps,
                "training_return_mean": row["mean_training_return"],
                "training_return_last_20_mean": row["last_20_episode_return"],
                "packet_metrics": {metric: row[metric] for metric in METRICS},
                "dropped_by_reason": json.loads(row["dropped_by_reason_json"]),
            }
        )

    training_history_path = output_dir / "dqn_training_episodes.csv"
    _write_rows(training_history_path, training_episode_rows)
    training_decisions_path = output_dir / "dqn_training_decisions.json"
    training_decisions_path.write_text(
        json.dumps(training_decision_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_decisions_path = output_dir / "dqn_evaluation_decisions.json"
    evaluation_decisions_path.write_text(
        json.dumps(evaluation_decision_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_packet_outcomes_path = output_dir / "dqn_evaluation_packet_outcomes.json"
    evaluation_packet_outcomes_path.write_text(
        json.dumps(evaluation_packet_outcome_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation_path = output_dir / "dqn_evaluation_per_seed.csv"
    _write_rows(evaluation_path, evaluation_rows)
    wait_ablation_path = output_dir / "dqn_wait_ablation_per_seed.csv"
    _write_rows(wait_ablation_path, wait_ablation_rows)

    aggregate_metrics = {}
    for metric in METRICS:
        values = [row[metric] for row in evaluation_rows if row[metric] is not None]
        aggregate_metrics[metric] = _summary_stats(values)
    baseline_rows = list(csv.DictReader((baseline_results_dir / "baseline_runs.csv").open()))
    repository_root = Path(__file__).resolve().parents[1]
    source_files = (
        Path(__file__).resolve(),
        repository_root / "src/fedroute/dqn.py",
        repository_root / "src/fedroute/routing.py",
        repository_root / "src/fedroute/packet_sim.py",
    )
    summary = {
        "baseline_config": str(baseline_config_path),
        "baseline_config_sha256": hashlib.sha256(baseline_config_path.read_bytes()).hexdigest(),
        "baseline_topology_trace_sha256": evaluation_trace_hash,
        "baseline_results_csv_sha256": hashlib.sha256(
            (baseline_results_dir / "baseline_runs.csv").read_bytes()
        ).hexdigest(),
        "dqn_training_config": str(training_config_path),
        "dqn_training_config_sha256": hashlib.sha256(training_config_path.read_bytes()).hexdigest(),
        "training_trace_sha256": _trace_hash(training_trace),
        "training_epoch_utc": training_epoch.isoformat(),
        "evaluation_trace_is_frozen_baseline": True,
        "original_candidate_action_semantics": "currently available outgoing links plus explicit DROP; no WAIT action",
        "candidate_action_semantics": (
            "currently available outgoing links, WAIT to the next topology sample when a "
            "future contact-graph route exists within deadline, plus explicit DROP"
        ),
        "state_semantics": [
            "local residual energy (feature 0)",
            "same-cluster membership (feature 1; zero when disabled)",
            "candidate residual energy (feature 2)",
            "candidate link quality (feature 3)",
            "remaining sampled contact time (feature 4)",
            "normalized candidate distance (feature 5)",
            "snapshot hop count to destination (feature 6)",
            "ground-station destination flag (feature 7)",
            "current-snapshot destination reachability (feature 8)",
            "WAIT row uses feature 5 for planned future wait divided by trace horizon",
            *(
                ["candidate node already appears in packet route (feature 9)"]
                if energy.include_path_history
                else []
            ),
        ],
        "reward_config": reward_settings,
        "source_sha256": {
            str(source_path.relative_to(repository_root)): hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            for source_path in source_files
        },
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "training_seeds": list(training_settings["training_seeds"]),
        "baseline_results": baseline_rows,
        "dqn_per_seed_results": evaluation_rows,
        "dqn_metric_mean_and_sample_std": aggregate_metrics,
        "training_episode_csv": str(training_history_path),
        "training_decision_log": str(training_decisions_path),
        "evaluation_decision_log": str(evaluation_decisions_path),
        "evaluation_packet_outcomes": str(evaluation_packet_outcomes_path),
        "evaluation_csv": str(evaluation_path),
        "wait_ablation_csv": str(wait_ablation_path),
        "dqn_wait_action_ablation": wait_ablation_rows,
        "per_seed_checkpoints": per_seed_summary,
        "interpretation": [
            "Training uses a distinct Walker epoch; packet evaluation uses the exact frozen baseline config, trace hash, and deterministic flows.",
            "Training returns and losses are diagnostic, not substitutes for packet performance metrics.",
            "Results are provisional; five seeds do not establish statistical significance or generalization.",
            (
                "When enabled, WAIT advances one topology sample. Training masks it when no "
                "future contact-graph route exists; packet evaluation additionally requires "
                "that the planned route fit the remaining packet deadline."
            ),
            (
                "The no-WAIT ablation evaluates the same trained weights with WAIT disabled; "
                "it is an inference-time action-mask ablation, not a separately trained policy."
            ),
            (
                "WAIT reward is normalized by full trace duration only when "
                "reward.normalize_wait_penalty_by_horizon is true; the default preserves "
                "the sample-normalized Phase 4.1 reward."
            ),
            "No fuzzy clustering or federated learning is included.",
        ],
    }
    summary_path = output_dir / "dqn_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("experiments/baseline_config.toml"))
    parser.add_argument("--training-config", type=Path, default=Path("experiments/dqn_training.toml"))
    parser.add_argument("--baseline-results", type=Path, default=Path("results/phase4-baseline"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4-dqn-phase4.1"),
    )
    args = parser.parse_args()
    summary = train_and_evaluate(
        args.baseline_config,
        args.training_config,
        args.output_dir,
        args.baseline_results,
    )
    print("seed pdr avg_delay_s avg_wait_s comm_energy_j hops")
    for row in summary["dqn_per_seed_results"]:
        print(
            row["training_seed"],
            row["packet_delivery_ratio"],
            row["average_end_to_end_delay_seconds"],
            row["average_waiting_delay_seconds"],
            row["total_communication_energy_j"],
            row["average_hop_count"],
        )
    print(f"Saved DQN training/evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
