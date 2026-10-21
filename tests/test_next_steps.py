from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
import torch

from fedroute.baselines import (
    calculate_route_metrics,
    contact_graph_route,
    energy_constrained_route,
    shortest_path_route,
)
from fedroute.config import GroundStationConfig, SimulationConfig, TopologyConfig, WalkerConfig
from fedroute.dqn import (
    CentralizedDQNTrainer,
    DQNPolicy,
    DQNReplayBuffer,
    DQNTrainer,
    DQNTrainingConfig,
    DQNTransition,
    train_centralized_dqn,
)
from fedroute.federated import FederatedDQNTrainer, FederatedEnergyConfig, aggregate_state_dicts
from fedroute.fuzzy import FuzzyClusterSelector, select_cluster_heads
from fedroute.routing import RoutingEnvConfig, SatelliteRoutingEnv
from fedroute.topology import TopologyTrace


def _make_trace():
    graph0 = nx.Graph()
    graph0.add_node("sat:1", kind="satellite")
    graph0.add_node("sat:2", kind="satellite")
    graph0.add_node("gs:gateway", kind="ground_station")
    graph0.add_edge("sat:1", "sat:2", range_km=1000.0, link_type="isl")
    graph0.add_edge("sat:2", "gs:gateway", range_km=3000.0, link_type="ground")
    graph1 = nx.Graph()
    graph1.add_node("sat:1", kind="satellite")
    graph1.add_node("sat:2", kind="satellite")
    graph1.add_node("gs:gateway", kind="ground_station")
    graph1.add_edge("sat:1", "sat:2", range_km=1200.0, link_type="isl")
    graph1.add_edge("sat:2", "gs:gateway", range_km=3100.0, link_type="ground")
    return TopologyTrace(
        times_utc=(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        graphs=(graph0, graph1),
        contact_windows=(),
    )


def test_routing_env_validates_masked_next_hop_choice():
    env = SatelliteRoutingEnv(
        _make_trace(),
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=5,
        ),
    )
    obs, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    assert obs.shape == (5, 9)
    assert info["action_mask"].tolist() == [1, 0, 0, 0, 0, 1]
    assert obs[0, 8] == 1.0
    assert env.action_space.n == 6


def test_path_history_feature_marks_revisits_without_masking_them():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
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
    env = SatelliteRoutingEnv(
        TopologyTrace(
            tuple(start + timedelta(seconds=index) for index in range(3)),
            tuple(graphs),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=2,
            max_hops=4,
            include_path_history=True,
        ),
    )

    observation, info = env.reset(
        options={"source": "sat:a", "destination": "gs:destination"}
    )
    assert observation.shape == (2, 10)
    assert observation[0, 9] == 0.0
    observation, _, terminated, _, info = env.step(0)
    assert not terminated
    assert info["candidate_nodes"] == ("sat:a",)
    assert info["action_mask"][0] == 1
    assert observation[0, 9] == 1.0

    observation, _, terminated, _, info = env.step(0)
    assert not terminated
    assert info["candidate_nodes"] == ("gs:destination",)
    assert info["action_mask"][0] == 1
    assert observation[0, 9] == 0.0
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["delivered"]
    assert reward > 0.0


def test_cluster_aware_routing_exposes_same_cluster_candidate():
    config = RoutingEnvConfig(
        energy_capacity_j=100.0,
        tx_electronics_j_per_bit=1e-9,
        rx_electronics_j_per_bit=1e-9,
        amplifier_j_per_bit_km2=1e-12,
        transmission_rate_bps=1_000_000,
        max_neighbors=5,
    )
    env = SatelliteRoutingEnv(_make_trace(), config, cluster_aware=True)
    observation, _ = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    assert observation.shape == (5, 9)
    assert observation[0, 1] == 1.0

    default_env = SatelliteRoutingEnv(_make_trace(), config)
    default_observation, _ = default_env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    assert default_observation[0, 1] == 0.0


def test_isolated_satellite_only_has_drop_action():
    graph = nx.Graph()
    graph.add_node("sat:1", kind="satellite")
    graph.add_node("gs:gateway", kind="ground_station")
    trace = TopologyTrace(
        times_utc=(datetime(2026, 1, 1, tzinfo=timezone.utc),),
        graphs=(graph,),
        contact_windows=(),
    )
    env = SatelliteRoutingEnv(
        trace,
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=3,
        ),
    )

    observation, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    assert np.all(observation == 0.0)
    assert info["action_mask"].tolist() == [0, 0, 0, 1]
    _, _, terminated, truncated, drop_info = env.step(3)
    assert terminated
    assert not truncated
    assert drop_info["loss_cause"] == "no_feasible_next_hop"


def test_disconnected_component_is_marked_but_feasible_hop_remains_allowed():
    graph = nx.Graph()
    graph.add_node("sat:1", kind="satellite")
    graph.add_node("sat:2", kind="satellite")
    graph.add_node("gs:gateway", kind="ground_station")
    graph.add_edge("sat:1", "sat:2", range_km=100.0)
    trace = TopologyTrace(
        times_utc=(datetime(2026, 1, 1, tzinfo=timezone.utc),),
        graphs=(graph,),
        contact_windows=(),
    )
    env = SatelliteRoutingEnv(
        trace,
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=3,
        ),
    )

    observation, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    assert observation[0, 8] == 0.0
    assert info["action_mask"].tolist() == [1, 0, 0, 1]

    _, _, terminated, truncated, drop_info = env.step(3)
    assert terminated
    assert not truncated
    assert drop_info["loss_cause"] == "voluntary_drop"

    env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    next_observation, _, terminated, truncated, _ = env.step(0)
    assert next_observation.shape == (3, 9)
    assert not terminated
    assert not truncated


def test_route_can_continue_when_destination_contact_appears_next_snapshot():
    graph0 = nx.Graph()
    graph0.add_node("sat:1", kind="satellite")
    graph0.add_node("sat:2", kind="satellite")
    graph0.add_node("gs:gateway", kind="ground_station")
    graph0.add_edge("sat:1", "sat:2", range_km=100.0)
    graph1 = graph0.copy()
    graph1.add_edge("sat:2", "gs:gateway", range_km=100.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    env = SatelliteRoutingEnv(
        TopologyTrace((start, start.replace(second=10)), (graph0, graph1), ()),
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=3,
        ),
    )

    observation, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    assert observation[0, 8] == 0.0
    assert info["action_mask"][0] == 1

    next_observation, _, terminated, _, next_info = env.step(0)
    assert not terminated
    assert next_info["graph_index"] == 1
    assert next_info["action_mask"][0] == 1
    assert next_observation[0, 8] == 1.0


def test_routing_step_info_mask_describes_returned_next_state():
    graph0 = nx.Graph()
    graph0.add_nodes_from(
        (
            ("sat:1", {"kind": "satellite"}),
            ("sat:2", {"kind": "satellite"}),
            ("sat:3", {"kind": "satellite"}),
            ("gs:gateway", {"kind": "ground_station"}),
        )
    )
    graph0.add_edge("sat:1", "sat:2", range_km=100.0)
    graph1 = nx.Graph()
    graph1.add_nodes_from(graph0.nodes(data=True))
    graph1.add_edge("sat:2", "sat:3", range_km=100.0)
    graph1.add_edge("sat:3", "gs:gateway", range_km=100.0)
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)),
            (graph0, graph1),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=4,
        ),
    )
    env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    next_observation, _, _, _, info = env.step(0)

    assert info["current_node"] == "sat:2"
    assert info["graph_index"] == 1
    assert info["action_mask"].tolist() == [1, 0, 0, 0, 1]
    assert next_observation[0, 7] == 0.0


def test_wait_action_advances_one_sample_only_when_future_route_exists():
    graph0 = nx.Graph()
    graph0.add_node("sat:1", kind="satellite")
    graph0.add_node("gs:gateway", kind="ground_station")
    graph1 = nx.Graph()
    graph1.add_node("sat:1", kind="satellite")
    graph1.add_node("gs:gateway", kind="ground_station")
    graph1.add_edge("sat:1", "gs:gateway", range_km=0.0, remaining_contact_s=10.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    config = RoutingEnvConfig(
        energy_capacity_j=10.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1000.0,
        packet_bits=100,
        max_neighbors=3,
        wait_action_enabled=True,
    )
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (start, start + timedelta(seconds=10)),
            (graph0, graph1),
            (),
        ),
        config,
    )
    observation, info = env.reset(
        options={"source": "sat:1", "destination": "gs:gateway"}
    )

    assert observation.shape == (4, 9)
    assert info["action_mask"].tolist() == [0, 0, 0, 1, 1]
    assert info["wait_action_index"] == 3
    assert info["drop_action_index"] == 4

    next_observation, reward, terminated, truncated, info = env.step(3)

    assert reward == pytest.approx(-config.delay_penalty)
    assert info["reward_components"] == {"wait_delay": pytest.approx(reward)}
    assert not terminated and not truncated
    assert info["graph_index"] == 1
    assert info["action_mask"].tolist() == [1, 0, 0, 0, 1]
    assert next_observation[0, 7] == 1.0
    assert info["communication_energy_j"] == {"sat:1": 0.0}

    _, delivered_reward, terminated, truncated, delivered = env.step(0)
    assert terminated and not truncated
    assert delivered["delivered"]
    assert delivered["reward_components"]["delivery"] == config.delivery_reward
    assert delivered_reward == pytest.approx(sum(delivered["reward_components"].values()))


def test_wait_penalty_horizon_normalization_reduces_long_wait_cost():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    graphs = []
    for index in range(3):
        graph = nx.Graph()
        graph.add_node("sat:1", kind="satellite")
        graph.add_node("gs:gateway", kind="ground_station")
        if index == 2:
            graph.add_edge("sat:1", "gs:gateway", range_km=0.0)
        graphs.append(graph)
    trace = TopologyTrace(
        (start, start + timedelta(seconds=10), start + timedelta(seconds=20)),
        tuple(graphs),
        (),
    )
    rewards = []
    for normalize_by_horizon in (False, True):
        env = SatelliteRoutingEnv(
            trace,
            RoutingEnvConfig(
                energy_capacity_j=10.0,
                tx_electronics_j_per_bit=0.01,
                rx_electronics_j_per_bit=0.002,
                amplifier_j_per_bit_km2=0.0,
                transmission_rate_bps=1000.0,
                packet_bits=100,
                max_neighbors=3,
                wait_action_enabled=True,
                normalize_wait_penalty_by_horizon=normalize_by_horizon,
            ),
        )
        _, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
        assert info["action_mask"].tolist() == [0, 0, 0, 1, 1]

        _, reward, terminated, truncated, _ = env.step(3)

        assert not terminated and not truncated
        rewards.append(reward)

    assert rewards == pytest.approx([-0.2, -0.1])


def test_wait_action_is_masked_when_no_future_contact_route_exists():
    graph = nx.Graph()
    graph.add_node("sat:1", kind="satellite")
    graph.add_node("gs:gateway", kind="ground_station")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (start, start + timedelta(seconds=10)),
            (graph, graph.copy()),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            max_neighbors=3,
            wait_action_enabled=True,
        ),
    )

    _, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    assert info["action_mask"].tolist() == [0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="not feasible"):
        env.step(3)


def test_routing_energy_ledger_uses_distinct_tx_rx_and_compute_costs():
    config = RoutingEnvConfig(
        energy_capacity_j=1.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1_000_000,
        packet_bits=10,
        max_neighbors=3,
    )
    env = SatelliteRoutingEnv(_make_trace(), config)
    _, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    _, _, _, _, info = env.step(0)

    assert info["communication_energy_j"] == {"sat:1": pytest.approx(0.1), "sat:2": pytest.approx(0.02)}
    assert info["compute_energy_j"] == {"sat:1": 0.0, "sat:2": 0.0}
    assert info["residual_energy_j"] == {"sat:1": pytest.approx(0.9), "sat:2": pytest.approx(0.98)}
    env.charge_compute_energy("sat:2", 0.3)
    assert env.energies["sat:2"] == pytest.approx(0.68)
    with pytest.raises(ValueError, match="insufficient energy"):
        env.charge_compute_energy("sat:2", 0.69)
    assert min(env.energies.values()) >= 0.0


def test_routing_rejects_unaffordable_sender_transmission_without_negative_energy():
    config = RoutingEnvConfig(
        energy_capacity_j=0.05,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1_000_000,
        packet_bits=10,
        max_neighbors=3,
    )
    env = SatelliteRoutingEnv(_make_trace(), config)
    env.reset(options={"source": "sat:1", "destination": "gs:gateway"})

    _, _, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert info["loss_cause"] == "energy_depletion"
    assert info["communication_energy_j"] == {"sat:1": 0.0, "sat:2": 0.0}
    assert info["residual_energy_j"] == {"sat:1": 0.05, "sat:2": 0.05}
    assert min(info["residual_energy_j"].values()) >= 0.0


def test_routing_rejects_unaffordable_receiver_without_charging_sender():
    config = RoutingEnvConfig(
        energy_capacity_j=1.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1_000_000,
        packet_bits=10,
        max_neighbors=3,
    )
    env = SatelliteRoutingEnv(_make_trace(), config)
    env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
    env.charge_compute_energy("sat:2", 0.99)

    _, _, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert info["loss_cause"] == "receiver_energy_depletion"
    assert info["communication_energy_j"] == {"sat:1": 0.0, "sat:2": 0.0}
    assert info["residual_energy_j"] == {"sat:1": 1.0, "sat:2": pytest.approx(0.01)}


def test_identical_routing_inputs_reproduce_route_and_energy_ledger():
    config = RoutingEnvConfig(
        energy_capacity_j=1.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=1e-12,
        transmission_rate_bps=1_000_000,
        packet_bits=10,
        max_neighbors=3,
    )
    results = []
    for _ in range(2):
        env = SatelliteRoutingEnv(_make_trace(), config)
        _, info = env.reset(options={"source": "sat:1", "destination": "gs:gateway"})
        action = int(np.flatnonzero(info["action_mask"][:-1])[0])
        _, reward, terminated, truncated, info = env.step(action)
        results.append(
            (
                action,
                reward,
                terminated,
                truncated,
                info["communication_energy_j"],
                info["compute_energy_j"],
                info["residual_energy_j"],
            )
        )

    assert results[0] == results[1]


def test_shortest_path_baseline_supports_hops_and_delay():
    graph = nx.Graph()
    graph.add_edge("sat:1", "gs:gateway", range_km=10_000.0)
    graph.add_edge("sat:1", "sat:2", range_km=100.0)
    graph.add_edge("sat:2", "gs:gateway", range_km=100.0)

    assert shortest_path_route(graph, "sat:1", "gs:gateway") == ["sat:1", "gs:gateway"]
    assert shortest_path_route(
        graph,
        "sat:1",
        "gs:gateway",
        metric="delay",
        transmission_rate_bps=1_000_000,
    ) == ["sat:1", "sat:2", "gs:gateway"]
    hop_route = shortest_path_route(graph, "sat:1", "gs:gateway")
    delay_route = shortest_path_route(
        graph,
        "sat:1",
        "gs:gateway",
        metric="delay",
        transmission_rate_bps=1_000_000,
    )
    assert hop_route == ["sat:1", "gs:gateway"]
    assert delay_route == ["sat:1", "sat:2", "gs:gateway"]
    hop_metrics = calculate_route_metrics(
        graph,
        hop_route,
        transmission_rate_bps=1_000_000,
        packet_bits=1000,
    )
    delay_metrics = calculate_route_metrics(
        graph,
        delay_route,
        transmission_rate_bps=1_000_000,
        packet_bits=1000,
    )
    assert hop_metrics.hop_count == 1
    assert hop_metrics.total_delay_seconds == pytest.approx(10_000 / 299_792.458 + 0.001)
    assert hop_metrics.communication_energy_j is None
    assert hop_metrics.compute_energy_j is None
    assert delay_metrics.hop_count == 2
    assert delay_metrics.total_delay_seconds == pytest.approx(200 / 299_792.458 + 0.002)
    assert delay_metrics.total_delay_seconds < hop_metrics.total_delay_seconds

    graph["sat:1"]["gs:gateway"].update(tx_energy_j=0.5, rx_energy_j=0.2)
    graph["sat:1"]["sat:2"].update(tx_energy_j=0.1, rx_energy_j=0.05)
    graph["sat:2"]["gs:gateway"].update(tx_energy_j=0.1, rx_energy_j=0.05)
    for node in graph:
        graph.nodes[node]["compute_energy_j"] = 0.1
    accounted = calculate_route_metrics(
        graph,
        delay_route,
        transmission_rate_bps=1_000_000,
        packet_bits=1000,
        waiting_seconds=2.0,
        initial_energy_by_node_j={"sat:1": 10.0, "sat:2": 10.0, "gs:gateway": 10.0},
    )
    assert accounted.communication_energy_j == pytest.approx(0.3)
    assert accounted.compute_energy_j == pytest.approx(0.3)
    assert accounted.waiting_seconds == 2.0
    assert accounted.total_delay_seconds == pytest.approx(
        accounted.propagation_seconds + accounted.transmission_seconds + 2.0
    )
    assert dict(accounted.residual_energy_j) == pytest.approx(
        {"sat:1": 9.8, "sat:2": 9.75, "gs:gateway": 9.85}
    )


def test_energy_constrained_route_rejects_unaffordable_path_and_handles_no_path():
    graph = nx.Graph()
    graph.add_node("A", compute_energy_j=0.0)
    graph.add_node("B", compute_energy_j=0.0)
    graph.add_node("C", compute_energy_j=0.0)
    graph.add_node("G", compute_energy_j=0.0)
    graph.add_edge("A", "B", range_km=10.0, tx_energy_j=0.8, rx_energy_j=0.1)
    graph.add_edge("B", "G", range_km=10.0, tx_energy_j=0.1, rx_energy_j=0.1)
    graph.add_edge("A", "C", range_km=100.0, tx_energy_j=0.1, rx_energy_j=0.1)
    graph.add_edge("C", "G", range_km=100.0, tx_energy_j=0.1, rx_energy_j=0.1)
    budgets = {node: 1.0 for node in graph}
    budgets["A"] = 0.5

    route = energy_constrained_route(
        graph,
        "A",
        "G",
        initial_energy_by_node_j=budgets,
        transmission_rate_bps=1_000_000,
        packet_bits=100,
    )

    assert route == ["A", "C", "G"]
    disconnected = nx.Graph()
    disconnected.add_node("A", compute_energy_j=0.0)
    disconnected.add_node("G", compute_energy_j=0.0)
    assert energy_constrained_route(
        disconnected,
        "A",
        "G",
        initial_energy_by_node_j={"A": 1.0, "G": 1.0},
        transmission_rate_bps=1_000_000,
    ) is None


def test_shortest_path_ties_are_lexicographic_across_edge_insertion_orders():
    edge_set = [
        ("A", "C", {"range_km": 1.0}),
        ("C", "G", {"range_km": 1.0}),
        ("A", "B", {"range_km": 1.0}),
        ("B", "G", {"range_km": 1.0}),
    ]
    first = nx.Graph()
    first.add_edges_from(edge_set)
    second = nx.Graph()
    second.add_edges_from(reversed(edge_set))

    first_route = shortest_path_route(first, "A", "G")
    second_route = shortest_path_route(second, "A", "G")

    assert first_route == second_route == ["A", "B", "G"]


def test_contact_graph_baseline_waits_for_future_contact():
    graph0 = nx.Graph()
    graph0.add_nodes_from(("sat:1", "gs:gateway"))
    graph1 = nx.Graph()
    graph1.add_edge("sat:1", "gs:gateway", range_km=100.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trace = TopologyTrace(
        times_utc=(start, start.replace(second=10)),
        graphs=(graph0, graph1),
        contact_windows=(),
    )

    route = contact_graph_route(
        trace,
        "sat:1",
        "gs:gateway",
        transmission_rate_bps=1_000_000,
    )
    assert route is not None
    assert route.states == (("sat:1", 0), ("sat:1", 1), ("gs:gateway", 1))
    hop_delay = 100.0 / 299_792.458 + 4096 / 1_000_000
    expected_arrival = start.replace(second=10) + timedelta(seconds=hop_delay)
    assert route.arrival_time_utc == expected_arrival
    assert route.elapsed_seconds == pytest.approx(10.0 + hop_delay)
    assert route.waiting_seconds == 10.0
    assert route.propagation_seconds == pytest.approx(100.0 / 299_792.458)
    assert route.transmission_seconds == pytest.approx(4096 / 1_000_000)
    assert route.total_delay_seconds == pytest.approx(10.0 + hop_delay)
    assert route.hop_count == 1


def test_contact_graph_planner_allows_only_one_hop_per_slot():
    graphs = []
    for _ in range(3):
        graph = nx.Graph()
        graph.add_nodes_from(("sat:a", "sat:b", "gs:gateway"))
        graph.add_edge("sat:a", "sat:b", range_km=100.0, remaining_contact_s=10.0)
        graph.add_edge("sat:b", "gs:gateway", range_km=100.0, remaining_contact_s=10.0)
        graphs.append(graph)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trace = TopologyTrace(
        times_utc=(start, start.replace(second=10), start.replace(second=20)),
        graphs=tuple(graphs),
        contact_windows=(),
    )

    route = contact_graph_route(
        trace,
        "sat:a",
        "gs:gateway",
        transmission_rate_bps=1_000_000,
        packet_bits=100,
    )

    assert route is not None
    assert route.states == (("sat:a", 0), ("sat:b", 1), ("gs:gateway", 1))
    assert route.hop_count == 2
    assert route.arrival_time_utc > start.replace(second=10)
    assert route.total_delay_seconds == pytest.approx(route.elapsed_seconds)
    assert route.waiting_seconds > 9.0


def test_contact_graph_planner_rejects_link_shorter_than_packet_delay():
    graph = nx.Graph()
    graph.add_edge(
        "sat:a",
        "gs:gateway",
        range_km=100.0,
        remaining_contact_s=0.001,
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trace = TopologyTrace((start,), (graph,), ())

    route = contact_graph_route(
        trace,
        "sat:a",
        "gs:gateway",
        transmission_rate_bps=1_000_000,
        packet_bits=4096,
    )

    assert route is None


def test_fuzzy_cluster_selector_prefers_high_capability_satellite():
    selector = FuzzyClusterSelector()
    score_a = selector.score(
        residual_energy=0.90,
        processing_capacity=0.95,
        traffic_load=0.10,
        contact_duration=0.85,
        link_quality=0.80,
        neighbor_count=0.90,
        distance_to_ground=0.20,
    )
    score_b = selector.score(
        residual_energy=0.30,
        processing_capacity=0.25,
        traffic_load=0.80,
        contact_duration=0.20,
        link_quality=0.30,
        neighbor_count=0.30,
        distance_to_ground=0.90,
    )
    assert score_a > score_b


def test_dqn_policy_outputs_action_dim():
    policy = DQNPolicy(input_dim=9, hidden_dim=16)
    two_candidates = np.ones((2, 9), dtype=np.float32)
    three_candidates = np.vstack((two_candidates, np.full((1, 9), 2.0, dtype=np.float32)))
    reordered_candidates = two_candidates[::-1].copy()
    two_candidate_values = policy(two_candidates)
    three_candidate_values = policy(three_candidates)
    reordered_values = policy(reordered_candidates)

    assert two_candidate_values.shape == (3,)
    assert three_candidate_values.shape == (4,)
    assert torch.allclose(two_candidate_values[:2], three_candidate_values[:2])
    assert torch.allclose(two_candidate_values[-1], three_candidate_values[-1])
    assert torch.allclose(reordered_values[:2], two_candidate_values[:2].flip(0))
    assert torch.allclose(reordered_values[-1], two_candidate_values[-1])


def test_dqn_action_mask_blocks_invalid_candidates_and_drop_is_explicit():
    policy = DQNPolicy(input_dim=9, hidden_dim=8)
    trainer = CentralizedDQNTrainer(policy)
    observation = np.zeros((3, 9), dtype=np.float32)

    with patch.object(policy, "forward", return_value=torch.tensor([100.0, 4.0, 2.0, 1.0])):
        assert trainer.choose_action(observation, np.array([0, 1, 1, 1])) == 1
        assert trainer.choose_action(observation, np.array([0, 0, 0, 1])) == 3


def test_dqn_td_target_uses_only_valid_next_actions():
    policy = DQNPolicy(input_dim=9, hidden_dim=1)
    with torch.no_grad():
        first, _, second, _, output = policy.candidate_scorer
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        second.weight.fill_(1.0)
        second.bias.zero_()
        output.weight.fill_(1.0)
        output.bias.zero_()
        policy.drop_action_value.fill_(0.0)
    trainer = CentralizedDQNTrainer(policy, gamma=0.5)
    next_state = np.zeros((3, 9), dtype=np.float32)
    next_state[:, 0] = [1.0, 9.0, 3.0]

    target = trainer.compute_td_target(
        reward=2.0,
        next_state=next_state,
        done=False,
        next_action_mask=np.array([1, 0, 0, 1]),
    )

    assert target.item() == pytest.approx(2.5)


def test_dqn_replay_buffer_copies_transitions_and_respects_capacity():
    buffer = DQNReplayBuffer(capacity=2)
    state = np.zeros((3, 9), dtype=np.float32)
    mask = np.array([1, 0, 0, 1], dtype=np.int8)
    for action in (0, 3, 3):
        buffer.add(
            DQNTransition(
                state=state,
                action=action,
                reward=1.0,
                next_state=state,
                terminated=action == 3,
                truncated=False,
                action_mask=mask,
                next_action_mask=mask,
            )
        )
    state.fill(9.0)

    assert len(buffer) == 2
    batch = buffer.sample(2, __import__("random").Random(1))
    assert all(np.all(item.state == 0.0) for item in batch)
    assert all(item.action in (3,) for item in batch)


def test_three_step_returns_aggregate_rewards_and_bootstrap_truncations():
    gamma = 0.5
    state = np.zeros((2, 9), dtype=np.float32)
    mask = np.array([1, 0, 1], dtype=np.int8)
    transitions = [
        DQNTransition(
            state=state,
            action=0,
            reward=reward,
            next_state=state,
            terminated=False,
            truncated=index == 2,
            action_mask=mask,
            next_action_mask=np.array([0, 0, 1], dtype=np.int8),
        )
        for index, reward in enumerate((1.0, 2.0, 3.0))
    ]
    aggregate = DQNTrainer._aggregate_n_step(transitions, gamma)
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=4),
        DQNTrainingConfig(
            seed=1,
            episodes=1,
            batch_size=1,
            replay_capacity=2,
            warmup_transitions=1,
            hidden_dim=4,
            gamma=gamma,
            n_step=3,
        ),
    )
    with torch.no_grad():
        for parameter in trainer.policy.parameters():
            parameter.zero_()
        for parameter in trainer.target_policy.parameters():
            parameter.zero_()
        trainer.target_policy.drop_action_value.fill_(4.0)

    loss = trainer._train_batch([aggregate])

    assert aggregate.reward == pytest.approx(2.75)
    assert aggregate.discount_steps == 3
    assert aggregate.truncated
    assert not aggregate.terminated
    assert loss == pytest.approx(3.25**2)


def test_dqn_bootstraps_from_time_limit_truncation():
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=4),
        DQNTrainingConfig(
            seed=1,
            episodes=1,
            batch_size=1,
            replay_capacity=2,
            warmup_transitions=1,
            hidden_dim=4,
            gamma=0.5,
        ),
    )
    with torch.no_grad():
        for parameter in trainer.policy.parameters():
            parameter.zero_()
        for parameter in trainer.target_policy.parameters():
            parameter.zero_()
        trainer.target_policy.drop_action_value.fill_(4.0)

    transition = DQNTransition(
        state=np.zeros((2, 9), dtype=np.float32),
        action=0,
        reward=1.0,
        next_state=np.zeros((2, 9), dtype=np.float32),
        terminated=False,
        truncated=True,
        action_mask=np.array([1, 0, 1], dtype=np.int8),
        next_action_mask=np.array([0, 0, 1], dtype=np.int8),
    )

    loss = trainer._train_batch([transition])

    assert loss == pytest.approx(9.0)


def test_dqn_trainer_records_step_limit_as_bootstrappable_truncation():
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            ("sat:source", {"kind": "satellite"}),
            ("sat:relay", {"kind": "satellite"}),
            ("gs:destination", {"kind": "ground_station"}),
        )
    )
    graph.add_edge("sat:source", "sat:relay", range_km=0.0)
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (datetime(2026, 1, 1, tzinfo=timezone.utc),),
            (graph,),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=2,
        ),
    )
    policy = DQNPolicy(input_dim=9, hidden_dim=4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.drop_action_value.fill_(-1.0)
    trainer = DQNTrainer(
        policy,
        DQNTrainingConfig(
            seed=1,
            episodes=1,
            max_steps_per_episode=1,
            batch_size=1,
            replay_capacity=2,
            warmup_transitions=2,
            epsilon_start=0.0,
            epsilon_end=0.0,
            hidden_dim=4,
        ),
    )
    logged_transitions = []

    history = trainer.train(
        env,
        "sat:source",
        "gs:destination",
        training_start_indices=(0,),
        transition_callback=logged_transitions.append,
    )

    transition = trainer.replay_buffer.transitions[0]
    assert history[0]["truncated"]
    assert logged_transitions[0]["truncated"]
    assert not logged_transitions[0]["terminated"]
    assert not transition.terminated
    assert transition.truncated


def test_dqn_hop_limit_is_a_terminal_drop_not_a_truncation():
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            ("sat:source", {"kind": "satellite"}),
            ("sat:relay", {"kind": "satellite"}),
            ("gs:destination", {"kind": "ground_station"}),
        )
    )
    graph.add_edge("sat:source", "sat:relay", range_km=0.0)
    graph.add_edge("sat:relay", "gs:destination", range_km=0.0)
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (datetime(2026, 1, 1, tzinfo=timezone.utc),),
            (graph,),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=2,
            max_hops=1,
        ),
    )
    _, info = env.reset(options={"source": "sat:source", "destination": "gs:destination"})

    _, _, terminated, truncated, next_info = env.step(0)

    assert terminated
    assert not truncated
    assert next_info["loss_cause"] == "hop_limit"


def test_dqn_trainer_uses_replay_masks_and_updates_target_network():
    config = RoutingEnvConfig(
        energy_capacity_j=100.0,
        tx_electronics_j_per_bit=1e-9,
        rx_electronics_j_per_bit=1e-9,
        amplifier_j_per_bit_km2=1e-12,
        transmission_rate_bps=1_000_000,
        max_neighbors=5,
    )
    env = SatelliteRoutingEnv(_make_trace(), config)
    trainer_config = DQNTrainingConfig(
        seed=41,
        episodes=4,
        max_steps_per_episode=4,
        batch_size=2,
        replay_capacity=32,
        warmup_transitions=2,
        target_update_interval=1,
        epsilon_start=1.0,
        epsilon_end=0.1,
        hidden_dim=8,
    )
    trainer = DQNTrainer(DQNPolicy(input_dim=9, hidden_dim=8), trainer_config)

    history = trainer.train(env, "sat:1", "gs:gateway", training_start_indices=(0,))

    assert len(history) == 4
    assert history[0]["epsilon"] > history[-1]["epsilon"]
    assert len(trainer.replay_buffer) >= 2
    assert trainer.gradient_steps > 0
    assert all(np.isfinite(item["mean_loss"]) for item in history if item["mean_loss"] is not None)
    for online, target in zip(trainer.policy.parameters(), trainer.target_policy.parameters()):
        assert torch.equal(online, target)


def test_dqn_learns_the_unambiguous_one_hop_route():
    graph = nx.Graph()
    graph.add_node("sat:source", kind="satellite")
    graph.add_node("gs:destination", kind="ground_station")
    graph.add_edge(
        "sat:source",
        "gs:destination",
        range_km=0.0,
        remaining_contact_s=1.0,
    )
    trace = TopologyTrace(
        (datetime(2026, 1, 1, tzinfo=timezone.utc),),
        (graph,),
        (),
    )
    env = SatelliteRoutingEnv(
        trace,
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=2,
        ),
    )
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=8),
        DQNTrainingConfig(
            seed=7,
            episodes=80,
            max_steps_per_episode=2,
            batch_size=1,
            replay_capacity=128,
            warmup_transitions=1,
            target_update_interval=1,
            learning_rate=0.01,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.05,
            hidden_dim=8,
        ),
    )

    history = trainer.train(
        env,
        "sat:source",
        "gs:destination",
        training_start_indices=(0,),
    )
    observation, info = env.reset(
        options={"source": "sat:source", "destination": "gs:destination"}
    )
    selected_action = trainer.select_greedy_action(observation, info["action_mask"])

    assert info["action_mask"].tolist() == [1, 0, 1]
    assert selected_action == 0
    _, reward, terminated, truncated, result = env.step(selected_action)
    assert terminated and not truncated
    assert result["delivered"]
    assert reward > 0.0
    assert result["reward_components"]["delivery"] == env.config.delivery_reward
    assert any(episode["delivered"] for episode in history)


def test_dqn_learns_a_static_three_hop_route():
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            ("sat:source", {"kind": "satellite"}),
            ("sat:relay-1", {"kind": "satellite"}),
            ("sat:relay-2", {"kind": "satellite"}),
            ("gs:destination", {"kind": "ground_station"}),
        )
    )
    graph.add_edges_from(
        (
            ("sat:source", "sat:relay-1", {"range_km": 100.0}),
            ("sat:relay-1", "sat:relay-2", {"range_km": 100.0}),
            ("sat:relay-2", "gs:destination", {"range_km": 100.0}),
        )
    )
    env = SatelliteRoutingEnv(
        TopologyTrace(
            (datetime(2026, 1, 1, tzinfo=timezone.utc),),
            (graph,),
            (),
        ),
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=4,
            max_hops=8,
        ),
    )
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=32),
        DQNTrainingConfig(
            seed=11,
            episodes=150,
            max_steps_per_episode=8,
            batch_size=32,
            replay_capacity=1000,
            warmup_transitions=32,
            target_update_interval=100,
            hidden_dim=32,
        ),
    )

    trainer.train(env, "sat:source", "gs:destination", training_start_indices=(0,))
    observation, info = env.reset(
        options={"source": "sat:source", "destination": "gs:destination"}
    )
    route = ["sat:source"]
    for _ in range(4):
        action = trainer.select_greedy_action(observation, info["action_mask"])
        if action == info["drop_action_index"]:
            break
        route.append(info["candidate_nodes"][action])
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert info["delivered"]
    assert route == [
        "sat:source",
        "sat:relay-1",
        "sat:relay-2",
        "gs:destination",
    ]


def test_dqn_learns_to_wait_for_a_delayed_contact_with_normalized_cost():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    contact_index = 20
    graphs = []
    for index in range(contact_index + 2):
        graph = nx.Graph()
        graph.add_node("sat:source", kind="satellite")
        graph.add_node("gs:destination", kind="ground_station")
        if index >= contact_index:
            graph.add_edge(
                "sat:source",
                "gs:destination",
                range_km=0.0,
                remaining_contact_s=10.0,
            )
        graphs.append(graph)
    trace = TopologyTrace(
        tuple(start + timedelta(seconds=10 * index) for index in range(len(graphs))),
        tuple(graphs),
        (),
    )
    env = SatelliteRoutingEnv(
        trace,
        RoutingEnvConfig(
            energy_capacity_j=10.0,
            tx_electronics_j_per_bit=0.01,
            rx_electronics_j_per_bit=0.002,
            amplifier_j_per_bit_km2=0.0,
            transmission_rate_bps=1000.0,
            packet_bits=100,
            max_neighbors=2,
            max_hops=4,
            wait_action_enabled=True,
            normalize_wait_penalty_by_horizon=True,
        ),
    )
    trainer = DQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=32),
        DQNTrainingConfig(
            seed=11,
            episodes=150,
            max_steps_per_episode=contact_index + 8,
            batch_size=32,
            replay_capacity=1000,
            warmup_transitions=16,
            target_update_interval=100,
            hidden_dim=32,
        ),
    )

    trainer.train(env, "sat:source", "gs:destination", training_start_indices=(0,))
    observation, info = env.reset(
        options={"source": "sat:source", "destination": "gs:destination"}
    )
    first_action = trainer.select_greedy_action(observation, info["action_mask"])
    assert first_action == info["wait_action_index"]

    for _ in range(contact_index + 2):
        observation, _, terminated, truncated, info = env.step(first_action)
        if terminated or truncated:
            break
        first_action = trainer.select_greedy_action(observation, info["action_mask"])

    assert info["delivered"]


def test_centralized_dqn_trainer_runs_minimal_training():
    env = SatelliteRoutingEnv(
        _make_trace(),
        RoutingEnvConfig(
            energy_capacity_j=100.0,
            tx_electronics_j_per_bit=1e-9,
            rx_electronics_j_per_bit=1e-9,
            amplifier_j_per_bit_km2=1e-12,
            transmission_rate_bps=1_000_000,
            max_neighbors=5,
        ),
    )
    losses = train_centralized_dqn(env, hidden_dim=16, episodes=2)
    assert len(losses) >= 1
    assert all(np.isfinite(np.asarray(losses)))


def test_cluster_selector_assigns_to_highest_ranked_head():
    graph = nx.Graph()
    graph.add_node("sat:1", kind="satellite", residual_energy=0.9, processing_capacity=0.9, traffic_load=0.1)
    graph.add_node("sat:2", kind="satellite", residual_energy=0.2, processing_capacity=0.2, traffic_load=0.9)
    graph.add_node("sat:3", kind="satellite", residual_energy=0.3, processing_capacity=0.3, traffic_load=0.7)
    graph.add_node("gs:gateway", kind="ground_station")
    graph.add_edge("sat:1", "sat:2", range_km=1000.0, contact_duration_s=4000.0, link_quality=0.9)
    graph.add_edge("sat:1", "gs:gateway", range_km=3000.0, link_quality=0.8)
    graph.add_edge("sat:2", "gs:gateway", range_km=5000.0, link_quality=0.4)

    assignment = select_cluster_heads(graph)
    assert assignment["sat:1"] == "sat:1"
    assert assignment["sat:2"] == "sat:1"
    assert assignment["sat:3"] == "sat:3"


def test_federated_aggregation_weights_client_states_by_transitions():
    tensor_states = [
        {"weight": torch.tensor([1.0])},
        {"weight": torch.tensor([3.0])},
    ]
    result = aggregate_state_dicts(tensor_states, [1, 3])
    assert result["weight"].item() == 2.5


def test_federated_dqn_runs_a_round_across_clients():
    config = RoutingEnvConfig(
        energy_capacity_j=100.0,
        tx_electronics_j_per_bit=1e-9,
        rx_electronics_j_per_bit=1e-9,
        amplifier_j_per_bit_km2=1e-12,
        transmission_rate_bps=1_000_000,
        max_neighbors=5,
    )
    trainer = FederatedDQNTrainer(
        DQNPolicy(input_dim=9, hidden_dim=16),
        energy_config=FederatedEnergyConfig(
            training_j_per_transition=0.01,
            communication_j_per_byte=1e-6,
        ),
        client_energy_budgets_j={"client-a": 10.0, "client-b": 10.0},
    )
    result = trainer.run_round(
        [SatelliteRoutingEnv(_make_trace(), config), SatelliteRoutingEnv(_make_trace(), config)],
        episodes_per_client=1,
        client_ids=("client-a", "client-b"),
    )
    assert result.total_transitions > 0
    assert len(result.client_losses) == 2
    assert result.model_payload_bytes > 0
    assert result.communication_bytes == 2 * result.model_payload_bytes * 2
    energy_costs = dict(result.client_energy_costs_j)
    remaining = dict(result.remaining_energy_j)
    for client_id, losses in zip(("client-a", "client-b"), result.client_losses):
        assert np.isclose(energy_costs[client_id], len(losses) * 0.01 + 2 * result.model_payload_bytes * 1e-6)
        assert np.isclose(remaining[client_id], 10.0 - energy_costs[client_id])
