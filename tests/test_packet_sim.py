from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest
import torch

from fedroute.packet_sim import PacketFlow, PacketSimulationConfig, PacketSimulator
from fedroute.dqn import DQNPolicy
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace


def _energy_config(packet_bits=100, capacity_j=10.0):
    return RoutingEnvConfig(
        energy_capacity_j=capacity_j,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1000.0,
        packet_bits=packet_bits,
        max_neighbors=4,
    )


def _trace(graphs, step_seconds=1.0):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = tuple(start + timedelta(seconds=index * step_seconds) for index in range(len(graphs)))
    return TopologyTrace(times, tuple(graphs), ())


def _single_hop_graph(distance_km=299.792458, remaining_contact_s=3.0):
    graph = nx.Graph()
    graph.add_node("sat:a", kind="satellite")
    graph.add_node("gs:g", kind="ground_station")
    graph.add_edge(
        "sat:a",
        "gs:g",
        range_km=distance_km,
        remaining_contact_s=remaining_contact_s,
    )
    return graph


def test_packet_events_separate_transmission_completion_and_arrival():
    trace = _trace([_single_hop_graph(), _single_hop_graph()], step_seconds=5.0)
    simulator = PacketSimulator(trace, _energy_config(), PacketSimulationConfig(5.0))
    result = simulator.run((PacketFlow("sat:a", "gs:g", (0.0,), 100),), "delay")

    packet = result.packets[0]
    event_times = {event["event"]: event["time_seconds"] for event in result.event_log}
    assert packet.status == "delivered"
    assert result.packet_delivery_ratio == 1.0
    assert packet.transmission_delay_seconds == pytest.approx(0.1)
    assert packet.propagation_delay_seconds == pytest.approx(0.001)
    assert packet.accumulated_delay_seconds == pytest.approx(0.101)
    assert result.average_propagation_delay_seconds == pytest.approx(0.001)
    assert result.average_transmission_delay_seconds == pytest.approx(0.1)
    assert result.average_processing_delay_seconds == 0.0
    assert result.average_waiting_delay_seconds == 0.0
    assert event_times["transmission_completed"] == pytest.approx(0.1)
    assert event_times["packet_arrived"] == pytest.approx(0.101)
    assert result.total_communication_energy_j == pytest.approx(1.0)
    assert dict(result.communication_energy_by_satellite_j) == {
        "sat:a": pytest.approx(1.0),
    }
    assert dict(result.residual_energy_by_satellite_j)["sat:a"] == pytest.approx(9.0)


def test_processing_precedes_transmission_and_is_reported_separately():
    graph = _single_hop_graph(distance_km=299.792458, remaining_contact_s=2.0)
    graph["sat:a"]["gs:g"]["processing_s"] = 0.05
    trace = _trace([graph, graph.copy()], step_seconds=2.0)
    simulator = PacketSimulator(
        trace,
        _energy_config(),
        PacketSimulationConfig(
            horizon_seconds=2.0,
            processing_delay_seconds=0.0,
            processing_energy_j_per_hop=0.03,
        ),
    )

    result = simulator.run((PacketFlow("sat:a", "gs:g", (0.0,), 100),), "delay")

    packet = result.packets[0]
    event_times = {event["event"]: event["time_seconds"] for event in result.event_log}
    assert packet.status == "delivered"
    assert packet.processing_delay_seconds == pytest.approx(0.05)
    assert packet.transmission_delay_seconds == pytest.approx(0.1)
    assert packet.propagation_delay_seconds == pytest.approx(0.001)
    assert packet.accumulated_delay_seconds == pytest.approx(0.151)
    assert event_times["transmission_started"] == pytest.approx(0.05)
    assert event_times["transmission_completed"] == pytest.approx(0.15)
    assert event_times["packet_arrived"] == pytest.approx(0.151)
    assert result.total_compute_energy_j == pytest.approx(0.03)


def test_packet_waits_for_future_contact_before_transmitting():
    graph_without_link = _single_hop_graph(remaining_contact_s=0.0)
    graph_without_link.remove_edge("sat:a", "gs:g")
    graph_with_link = _single_hop_graph(distance_km=0.0, remaining_contact_s=0.5)
    trace = _trace([graph_without_link, graph_with_link, graph_with_link])
    simulator = PacketSimulator(trace, _energy_config(), PacketSimulationConfig(2.0))
    result = simulator.run((PacketFlow("sat:a", "gs:g", (0.0,), 100),), "contact")

    packet = result.packets[0]
    assert packet.status == "delivered"
    assert packet.waiting_delay_seconds == pytest.approx(1.0)
    assert packet.transmission_delay_seconds == pytest.approx(0.1)
    assert packet.accumulated_delay_seconds == pytest.approx(1.1)
    events = [event["event"] for event in result.event_log]
    assert events.index("packet_waiting") < events.index("transmission_started")


def test_packet_drops_when_contact_expires_before_transmission_completes():
    active = _single_hop_graph(distance_km=0.0, remaining_contact_s=0.05)
    inactive = _single_hop_graph(remaining_contact_s=0.0)
    inactive.remove_edge("sat:a", "gs:g")
    simulator = PacketSimulator(
        _trace([active, inactive]),
        _energy_config(),
        PacketSimulationConfig(1.0),
    )
    result = simulator.run((PacketFlow("sat:a", "gs:g", (0.0,), 100),), "delay")

    packet = result.packets[0]
    assert packet.status == "dropped"
    assert packet.drop_reason == "contact_expired"
    assert result.total_communication_energy_j == 0.0


def test_packets_contend_for_half_duplex_radio_and_energy_is_charged_once():
    trace = _trace([_single_hop_graph(distance_km=0.0, remaining_contact_s=2.0)] * 3)
    simulator = PacketSimulator(trace, _energy_config(), PacketSimulationConfig(2.0))
    flow = PacketFlow("sat:a", "gs:g", (0.0, 0.0), 100)
    result = simulator.run((flow,), "delay")

    assert result.delivered_packets == 2
    assert [packet.accumulated_delay_seconds for packet in result.packets] == pytest.approx(
        [0.1, 0.2]
    )
    assert result.packets[1].waiting_delay_seconds == pytest.approx(0.1)
    assert dict(result.communication_energy_by_satellite_j) == {
        "sat:a": pytest.approx(2.0),
    }
    assert result.total_communication_energy_j == pytest.approx(2.0)


def test_satellite_relay_charges_tx_and_rx_once_each():
    graphs = []
    for _ in range(3):
        graph = nx.Graph()
        graph.add_node("sat:a", kind="satellite")
        graph.add_node("sat:b", kind="satellite")
        graph.add_node("gs:g", kind="ground_station")
        graph.add_edge("sat:a", "sat:b", range_km=0.0, remaining_contact_s=1.0)
        graph.add_edge("sat:b", "gs:g", range_km=0.0, remaining_contact_s=1.0)
        graphs.append(graph)
    trace = _trace(graphs)
    energy = RoutingEnvConfig(
        energy_capacity_j=3.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1000.0,
        packet_bits=100,
    )
    result = PacketSimulator(trace, energy, PacketSimulationConfig(2.0)).run(
        (PacketFlow("sat:a", "gs:g", (0.0,), 100),),
        "delay",
    )

    assert result.delivered_packets == 1
    assert dict(result.communication_energy_by_satellite_j) == {
        "sat:a": pytest.approx(1.0),
        "sat:b": pytest.approx(1.2),
    }
    assert result.total_communication_energy_j == pytest.approx(2.2)
    assert dict(result.residual_energy_by_satellite_j) == {
        "sat:a": pytest.approx(2.0),
        "sat:b": pytest.approx(1.8),
    }


def test_packet_transmission_exactly_consumes_sender_energy():
    trace = _trace([_single_hop_graph(distance_km=0.0), _single_hop_graph(distance_km=0.0)])
    energy = RoutingEnvConfig(
        energy_capacity_j=1.0,
        tx_electronics_j_per_bit=0.01,
        rx_electronics_j_per_bit=0.002,
        amplifier_j_per_bit_km2=0.0,
        transmission_rate_bps=1000.0,
        packet_bits=100,
    )
    result = PacketSimulator(trace, energy, PacketSimulationConfig(1.0)).run(
        (PacketFlow("sat:a", "gs:g", (0.0,), 100),),
        "delay",
    )

    assert result.delivered_packets == 1
    assert dict(result.residual_energy_by_satellite_j)["sat:a"] == pytest.approx(0.0)


def test_energy_policy_uses_affordable_relay_when_direct_link_is_unaffordable():
    graphs = []
    for _ in range(3):
        graph = nx.Graph()
        graph.add_node("sat:a", kind="satellite")
        graph.add_node("sat:b", kind="satellite")
        graph.add_node("gs:g", kind="ground_station")
        graph.add_edge("sat:a", "gs:g", range_km=500.0, remaining_contact_s=2.0)
        graph.add_edge("sat:a", "sat:b", range_km=10.0, remaining_contact_s=2.0)
        graph.add_edge("sat:b", "gs:g", range_km=10.0, remaining_contact_s=2.0)
        graphs.append(graph)
    energy = RoutingEnvConfig(
        energy_capacity_j=0.3,
        tx_electronics_j_per_bit=0.0001,
        rx_electronics_j_per_bit=0.0005,
        amplifier_j_per_bit_km2=2e-8,
        transmission_rate_bps=1000.0,
        packet_bits=100,
    )

    result = PacketSimulator(
        _trace(graphs),
        energy,
        PacketSimulationConfig(2.0),
    ).run((PacketFlow("sat:a", "gs:g", (0.0,), 100),), "energy")

    assert result.delivered_packets == 1
    assert result.packets[0].route == ["sat:a", "sat:b", "gs:g"]
    assert result.packets[0].hop_count == 2


def test_packet_metrics_are_none_when_denominator_is_zero():
    graph = nx.Graph()
    graph.add_node("sat:a", kind="satellite")
    graph.add_node("gs:g", kind="ground_station")
    trace = _trace([graph, graph.copy()])
    simulator = PacketSimulator(trace, _energy_config(), PacketSimulationConfig(1.0))

    no_packets = simulator.run((), "delay")
    assert no_packets.generated_packets == 0
    assert no_packets.packet_delivery_ratio is None
    assert no_packets.average_end_to_end_delay_seconds is None
    assert no_packets.average_waiting_delay_seconds is None
    assert no_packets.average_hop_count is None

    no_deliveries = simulator.run(
        (PacketFlow("sat:a", "gs:g", (0.0,), 100),),
        "hops",
    )
    assert no_deliveries.packet_delivery_ratio == 0.0
    assert no_deliveries.average_end_to_end_delay_seconds is None
    assert no_deliveries.average_waiting_delay_seconds is None
    assert no_deliveries.average_hop_count is None
    assert dict(no_deliveries.dropped_by_reason) == {"no_feasible_route": 1}


def test_packet_simulation_is_deterministic_for_identical_inputs():
    trace = _trace([_single_hop_graph(), _single_hop_graph()], step_seconds=5.0)
    flow = PacketFlow("sat:a", "gs:g", (0.0,), 100)
    config = PacketSimulationConfig(5.0)
    first = PacketSimulator(trace, _energy_config(), config).run((flow,), "delay")
    second = PacketSimulator(trace, _energy_config(), config).run((flow,), "delay")

    assert first.packets == second.packets
    assert first.event_log == second.event_log
    assert first.communication_energy_by_satellite_j == second.communication_energy_by_satellite_j


def test_packet_simulator_requires_dqn_policy_and_uses_candidate_scores():
    graph = nx.Graph()
    graph.add_node("sat:a", kind="satellite")
    graph.add_node("sat:b", kind="satellite")
    graph.add_node("gs:g", kind="ground_station")
    graph.add_edge("sat:a", "sat:b", range_km=100.0, remaining_contact_s=1.0)
    graph.add_edge("sat:a", "gs:g", range_km=100.0, remaining_contact_s=1.0)
    trace = _trace([graph, graph.copy()], step_seconds=1.0)
    flow = (PacketFlow("sat:a", "gs:g", (0.0,), 100),)
    config = PacketSimulationConfig(1.0)
    energy = _energy_config()

    with pytest.raises(ValueError, match="requires a trained DQNPolicy"):
        PacketSimulator(trace, energy, config).run(flow, "dqn")

    policy = DQNPolicy(input_dim=9, hidden_dim=4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.drop_action_value.fill_(-1.0)
        policy.candidate_scorer[0].weight[0, 7] = 1.0
        policy.candidate_scorer[2].weight[0, 0] = 1.0
        policy.candidate_scorer[4].weight[0, 0] = 1.0
    result = PacketSimulator(trace, energy, config, dqn_policy=policy).run(flow, "dqn")

    assert result.delivered_packets == 1
    assert result.packets[0].route == ["sat:a", "gs:g"]
