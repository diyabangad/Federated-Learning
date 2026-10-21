"""Compare vanilla, path-history, and soft cycle-penalty DQNs on route controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics

import networkx as nx
import torch

from fedroute.dqn import DQNPolicy, DQNTrainer, DQNTrainingConfig
from fedroute.routing import RoutingEnvConfig, SatelliteRoutingEnv
from fedroute.topology import TopologyTrace


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 37, 53, 71)
EPISODES = 150
CYCLE_PENALTY = 0.5


def _trace_for_case(case: str) -> tuple[TopologyTrace, str, str]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    if case == "static_disconnected":
        graph = nx.Graph()
        graph.add_nodes_from(
            (
                ("sat:source", {"kind": "satellite"}),
                ("sat:dead-end", {"kind": "satellite"}),
                ("gs:destination", {"kind": "ground_station"}),
            )
        )
        graph.add_edge("sat:source", "sat:dead-end", range_km=100.0)
        return (
            TopologyTrace((start,), (graph,), ()),
            "sat:source",
            "gs:destination",
        )
    if case == "dynamic_revisit_required":
        graphs = []
        for index in range(3):
            graph = nx.Graph()
            graph.add_nodes_from(
                (
                    ("sat:a", {"kind": "satellite"}),
                    ("sat:b", {"kind": "satellite"}),
                    ("gs:destination", {"kind": "ground_station"}),
                )
            )
            if index < 2:
                graph.add_edge("sat:a", "sat:b", range_km=100.0)
            if index == 2:
                graph.add_edge("sat:a", "gs:destination", range_km=100.0)
            graphs.append(graph)
        return (
            TopologyTrace(
                tuple(start + timedelta(seconds=10 * index) for index in range(3)),
                tuple(graphs),
                (),
            ),
            "sat:a",
            "gs:destination",
        )

    route_hops = {
        "static_two_hop": 2,
        "static_three_hop": 3,
        "static_four_hop": 4,
        "static_six_hop": 6,
        "static_three_hop_dead_end": 3,
    }[case]
    destination = "gs:destination"
    route = ["sat:source"] + [
        f"sat:relay-{index}" for index in range(1, route_hops)
    ]
    graph = nx.Graph()
    graph.add_nodes_from((node, {"kind": "satellite"}) for node in route)
    graph.add_node(destination, kind="ground_station")
    for source, target in zip(route, route[1:] + [destination]):
        graph.add_edge(source, target, range_km=100.0)
    if case == "static_three_hop_dead_end":
        graph.add_node("sat:dead-end", kind="satellite")
        graph.add_edge("sat:source", "sat:dead-end", range_km=10.0)
    return TopologyTrace((start,), (graph,), ()), "sat:source", destination


def _make_env(
    trace: TopologyTrace,
    *,
    include_path_history: bool,
    max_hops: int,
) -> SatelliteRoutingEnv:
    return SatelliteRoutingEnv(
        trace,
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1_000_000.0,
            packet_bits=100,
            max_neighbors=4,
            max_hops=max_hops,
            include_path_history=include_path_history,
        ),
    )


def _rollout(
    trainer: DQNTrainer,
    env: SatelliteRoutingEnv,
    source: str,
    destination: str,
    *,
    cycle_penalty: float = 0.0,
) -> dict:
    observation, info = env.reset(options={"source": source, "destination": destination})
    route = [source]
    actions: list[str] = []
    revisit_count = 0
    for _ in range(len(env.trace.graphs) + env.config.max_hops + 2):
        with torch.no_grad():
            q_values = trainer.policy(observation).clone()
        if cycle_penalty:
            for action, node in enumerate(info["candidate_nodes"]):
                if node in route:
                    q_values[action] -= cycle_penalty
        q_values[torch.as_tensor(info["action_mask"] == 0)] = -torch.inf
        action = int(torch.argmax(q_values).item())
        if action == info["drop_action_index"]:
            actions.append("DROP")
        else:
            next_node = info["candidate_nodes"][action]
            actions.append(next_node)
            revisit_count += next_node in route
            route.append(next_node)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {
                "delivered": bool(info.get("delivered", False)),
                "loss_cause": info.get("loss_cause"),
                "route": route,
                "actions": actions,
                "cycle_revisits": revisit_count,
            }
    return {
        "delivered": False,
        "loss_cause": "controlled_rollout_limit",
        "route": route,
        "actions": actions,
        "cycle_revisits": revisit_count,
    }


def _train(
    trace: TopologyTrace,
    source: str,
    destination: str,
    seed: int,
    *,
    include_path_history: bool,
    max_hops: int,
    n_step: int = 1,
) -> tuple[DQNTrainer, list[dict]]:
    torch.manual_seed(seed)
    config = DQNTrainingConfig(
        seed=seed,
        episodes=EPISODES,
        max_steps_per_episode=max_hops + 4,
        batch_size=32,
        replay_capacity=10_000,
        warmup_transitions=16,
        target_update_interval=100,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        hidden_dim=32,
        n_step=n_step,
    )
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9 + int(include_path_history), hidden_dim=32),
        config,
    )
    env = _make_env(
        trace,
        include_path_history=include_path_history,
        max_hops=max_hops,
    )
    history = trainer.train(
        env,
        source,
        destination,
        training_start_indices=tuple(range(len(trace.graphs))),
    )
    return trainer, history


def _trace_hash(trace: TopologyTrace) -> str:
    payload = [
        {
            "time": timestamp.isoformat(),
            "nodes": sorted((node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)),
            "edges": sorted(
                (min(left, right), max(left, right), tuple(sorted(data.items())))
                for left, right, data in graph.edges(data=True)
            ),
        }
        for timestamp, graph in zip(trace.times_utc, trace.graphs)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode()
    ).hexdigest()


def run(output_path: Path) -> dict:
    cases = (
        "static_two_hop",
        "static_three_hop",
        "static_four_hop",
        "static_six_hop",
        "static_three_hop_dead_end",
        "static_disconnected",
        "dynamic_revisit_required",
    )
    runs = []
    for case in cases:
        trace, source, destination = _trace_for_case(case)
        max_hops = 10
        for seed in SEEDS:
            vanilla, vanilla_history = _train(
                trace,
                source,
                destination,
                seed,
                include_path_history=False,
                max_hops=max_hops,
            )
            history_policy, history_history = _train(
                trace,
                source,
                destination,
                seed,
                include_path_history=True,
                max_hops=max_hops,
            )
            n_step_policy, n_step_history = _train(
                trace,
                source,
                destination,
                seed,
                include_path_history=False,
                max_hops=max_hops,
                n_step=3,
            )
            for variant, trainer, training, with_history, penalty, n_step in (
                ("vanilla", vanilla, vanilla_history, False, 0.0, 1),
                ("vanilla_soft_cycle_penalty", vanilla, vanilla_history, False, CYCLE_PENALTY, 1),
                ("path_history", history_policy, history_history, True, 0.0, 1),
                ("n_step_3_return", n_step_policy, n_step_history, False, 0.0, 3),
            ):
                env = _make_env(
                    trace,
                    include_path_history=with_history,
                    max_hops=max_hops,
                )
                rollout = _rollout(
                    trainer,
                    env,
                    source,
                    destination,
                    cycle_penalty=penalty,
                )
                losses = [
                    row["mean_loss"]
                    for row in training
                    if row["mean_loss"] is not None
                ]
                runs.append(
                    {
                        "case": case,
                        "seed": seed,
                        "variant": variant,
                        "n_step": n_step,
                        "trace_sha256": _trace_hash(trace),
                        "training_episodes": len(training),
                        "training_deliveries": sum(
                            bool(row["delivered"]) for row in training
                        ),
                        "training_return_mean": statistics.mean(
                            float(row["return"]) for row in training
                        ),
                        "training_return_last_20": statistics.mean(
                            float(row["return"]) for row in training[-20:]
                        ),
                        "training_td_loss_mean": (
                            statistics.mean(float(loss) for loss in losses)
                            if losses
                            else None
                        ),
                        "training_curve": [
                            {
                                "episode": row["episode"],
                                "return": row["return"],
                                "mean_loss": row["mean_loss"],
                                "epsilon": row["epsilon"],
                                "delivered": row["delivered"],
                            }
                            for row in training
                        ],
                        "greedy_rollout": rollout,
                    }
                )

    source_paths = (
        Path(__file__).resolve(),
        ROOT / "src/fedroute/routing.py",
        ROOT / "src/fedroute/dqn.py",
    )
    result = {
        "seeds": list(SEEDS),
        "episodes_per_seed": EPISODES,
        "cycle_avoidance_penalty": CYCLE_PENALTY,
        "training_config": asdict(
            DQNTrainingConfig(
                seed=SEEDS[0],
                episodes=EPISODES,
                max_steps_per_episode=14,
                batch_size=32,
                replay_capacity=10_000,
                warmup_transitions=16,
                target_update_interval=100,
                learning_rate=1e-3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                hidden_dim=32,
                n_step=1,
            )
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase4-3/controlled_results_full.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(f"Saved {len(report['runs'])} controlled runs.")
