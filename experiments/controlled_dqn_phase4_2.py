"""Run a controlled DQN graph ladder and isolated WAIT-reward ablation."""

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


SEEDS = (11, 23, 37, 53, 71)
EPISODES = 150


def _make_env(
    *,
    route_hops: int,
    dead_end: bool = False,
    disconnected: bool = False,
    contact_delay_samples: int | None = None,
    normalize_wait_penalty_by_horizon: bool = False,
) -> SatelliteRoutingEnv:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    destination = "gs:destination"
    route_nodes = ["sat:source"] + [
        f"sat:relay-{index}" for index in range(1, route_hops)
    ]
    if contact_delay_samples is None:
        graph = nx.Graph()
        graph.add_nodes_from(
            (node, {"kind": "satellite"}) for node in route_nodes
        )
        graph.add_node(destination, kind="ground_station")
        for index in range(route_hops - 1):
            graph.add_edge(
                route_nodes[index],
                route_nodes[index + 1],
                range_km=100.0,
            )
        if not disconnected:
            graph.add_edge(route_nodes[-1], destination, range_km=100.0)
        if dead_end:
            graph.add_node("sat:dead-end", kind="satellite")
            graph.add_edge("sat:source", "sat:dead-end", range_km=10.0)
        graphs = (graph,)
        times = (start,)
    else:
        if route_hops != 1:
            raise ValueError("Time-varying controlled graphs currently use a one-hop route")
        graphs_list = []
        for index in range(contact_delay_samples + 2):
            graph = nx.Graph()
            graph.add_node("sat:source", kind="satellite")
            graph.add_node(destination, kind="ground_station")
            if index >= contact_delay_samples:
                graph.add_edge(
                    "sat:source",
                    destination,
                    range_km=0.0,
                    remaining_contact_s=10.0,
                )
            graphs_list.append(graph)
        graphs = tuple(graphs_list)
        times = tuple(
            start + timedelta(seconds=10 * index)
            for index in range(len(graphs))
        )

    trace = TopologyTrace(times, graphs, ())
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
            max_hops=8,
            wait_action_enabled=contact_delay_samples is not None,
            normalize_wait_penalty_by_horizon=normalize_wait_penalty_by_horizon,
        ),
    )


def _greedy_rollout(
    trainer: DQNTrainer,
    env: SatelliteRoutingEnv,
) -> dict:
    observation, info = env.reset(
        options={"source": "sat:source", "destination": "gs:destination"}
    )
    route = ["sat:source"]
    actions = []
    max_decisions = len(env.trace.graphs) + env.config.max_hops + 2
    for _ in range(max_decisions):
        action = trainer.select_greedy_action(observation, info["action_mask"])
        if action == info["drop_action_index"]:
            actions.append("DROP")
        elif action == info["wait_action_index"]:
            actions.append("WAIT")
        else:
            next_node = info["candidate_nodes"][action]
            actions.append(next_node)
            route.append(next_node)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {
                "delivered": bool(info.get("delivered", False)),
                "loss_cause": info.get("loss_cause"),
                "route": route,
                "actions": actions,
            }
    return {
        "delivered": False,
        "loss_cause": "controlled_rollout_limit",
        "route": route,
        "actions": actions,
    }


def _train_case(
    name: str,
    seed: int,
    *,
    route_hops: int,
    dead_end: bool = False,
    disconnected: bool = False,
    contact_delay_samples: int | None = None,
    normalize_wait_penalty_by_horizon: bool = False,
) -> dict:
    torch.manual_seed(seed)
    env = _make_env(
        route_hops=route_hops,
        dead_end=dead_end,
        disconnected=disconnected,
        contact_delay_samples=contact_delay_samples,
        normalize_wait_penalty_by_horizon=normalize_wait_penalty_by_horizon,
    )
    max_steps = (
        contact_delay_samples + 8
        if contact_delay_samples is not None
        else 16
    )
    config = DQNTrainingConfig(
        seed=seed,
        episodes=EPISODES,
        max_steps_per_episode=max_steps,
        batch_size=32,
        replay_capacity=10_000,
        warmup_transitions=16,
        target_update_interval=100,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        hidden_dim=32,
    )
    trainer = DQNTrainer(DQNPolicy(input_dim=9, hidden_dim=32), config)
    history = trainer.train(
        env,
        "sat:source",
        "gs:destination",
        training_start_indices=(0,),
    )
    rollout = _greedy_rollout(trainer, env)
    losses = [float(row["mean_loss"]) for row in history if row["mean_loss"] is not None]
    return {
        "case": name,
        "seed": seed,
        "training_episodes": EPISODES,
        "training_deliveries": sum(bool(row["delivered"]) for row in history),
        "mean_return": statistics.mean(float(row["return"]) for row in history),
        "last_20_return": statistics.mean(
            float(row["return"]) for row in history[-20:]
        ),
        "mean_episode_td_loss": statistics.mean(losses) if losses else None,
        "gradient_steps": trainer.gradient_steps,
        "greedy_rollout": rollout,
        "wait_normalized_by_trace_horizon": normalize_wait_penalty_by_horizon,
        "contact_delay_samples": contact_delay_samples,
    }


def run_controlled_experiment() -> dict:
    cases = (
        ("static_one_hop", {"route_hops": 1}),
        ("static_two_hop", {"route_hops": 2}),
        ("static_three_hop", {"route_hops": 3}),
        ("static_four_hop", {"route_hops": 4}),
        ("static_three_hop_dead_end", {"route_hops": 3, "dead_end": True}),
        (
            "disconnected",
            {"route_hops": 2, "disconnected": True},
        ),
        (
            "time_varying_two_sample_wait",
            {"route_hops": 1, "contact_delay_samples": 2},
        ),
    )
    scenario_runs = [
        _train_case(name, seed, **settings)
        for name, settings in cases
        for seed in SEEDS
    ]
    wait_ablation_runs = [
        _train_case(
            f"wait_{delay}_samples",
            seed,
            route_hops=1,
            contact_delay_samples=delay,
            normalize_wait_penalty_by_horizon=normalize,
        )
        for delay in (5, 10, 20, 30)
        for normalize in (False, True)
        for seed in SEEDS
    ]
    return {
        "seeds": list(SEEDS),
        "episodes_per_seed": EPISODES,
        "training_config": asdict(
            DQNTrainingConfig(
                seed=SEEDS[0],
                episodes=EPISODES,
                max_steps_per_episode=16,
                batch_size=32,
                replay_capacity=10_000,
                warmup_transitions=16,
                target_update_interval=100,
                learning_rate=1e-3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                hidden_dim=32,
            )
        ),
        "scenario_runs": scenario_runs,
        "wait_reward_ablation_runs": wait_ablation_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase4-2/controlled_dqn_results.json"),
    )
    args = parser.parse_args()
    report = run_controlled_experiment()
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(),
            root / "src/fedroute/dqn.py",
            root / "src/fedroute/routing.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(report['scenario_runs'])} scenario and "
          f"{len(report['wait_reward_ablation_runs'])} ablation runs to {args.output}")


if __name__ == "__main__":
    main()
