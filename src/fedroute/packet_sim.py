"""Deterministic discrete-event packet simulation over sampled topology traces."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Literal

import networkx as nx
import numpy as np
import torch

from fedroute.baselines import (
    contact_graph_route,
    energy_constrained_route,
    shortest_path_route,
)
from fedroute.dqn import DQNPolicy
from fedroute.routing import RoutingEnvConfig
from fedroute.topology import TopologyTrace

RoutingPolicy = Literal["hops", "delay", "energy", "contact", "dqn"]


@dataclass(frozen=True)
class PacketFlow:
    source_satellite: str
    destination: str
    generation_times_seconds: tuple[float, ...]
    packet_size_bits: int
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.source_satellite or not self.destination:
            raise ValueError("Packet flow endpoints cannot be empty")
        if self.source_satellite == self.destination:
            raise ValueError("Packet flow endpoints must differ")
        if self.packet_size_bits <= 0:
            raise ValueError("packet_size_bits must be positive")
        if not self.generation_times_seconds:
            raise ValueError("A packet flow requires at least one generation time")
        if any(not isfinite(time) or time < 0.0 for time in self.generation_times_seconds):
            raise ValueError("Packet generation times must be finite and non-negative")
        if tuple(sorted(self.generation_times_seconds)) != self.generation_times_seconds:
            raise ValueError("Packet generation times must be sorted")
        if self.deadline_seconds is not None and (
            not isfinite(self.deadline_seconds) or self.deadline_seconds <= 0.0
        ):
            raise ValueError("deadline_seconds must be finite and positive")


@dataclass(frozen=True)
class PacketSimulationConfig:
    horizon_seconds: float
    max_hops: int = 32
    processing_delay_seconds: float = 0.0
    processing_energy_j_per_hop: float = 0.0
    dqn_wait_action_enabled: bool = False
    cycle_avoidance_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.horizon_seconds) or self.horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be finite and positive")
        if self.max_hops <= 0:
            raise ValueError("max_hops must be positive")
        if not isfinite(self.processing_delay_seconds) or self.processing_delay_seconds < 0.0:
            raise ValueError("processing_delay_seconds must be finite and non-negative")
        if not isfinite(self.processing_energy_j_per_hop) or self.processing_energy_j_per_hop < 0.0:
            raise ValueError("processing_energy_j_per_hop must be finite and non-negative")
        if (
            not isfinite(self.cycle_avoidance_penalty)
            or not 0.0 <= self.cycle_avoidance_penalty <= 2.0
        ):
            raise ValueError("cycle_avoidance_penalty must be finite and between 0 and 2")


@dataclass
class Packet:
    packet_id: str
    source_satellite: str
    destination: str
    generation_time: datetime
    packet_size_bits: int
    current_satellite: str
    hop_count: int = 0
    accumulated_delay_seconds: float = 0.0
    propagation_delay_seconds: float = 0.0
    transmission_delay_seconds: float = 0.0
    processing_delay_seconds: float = 0.0
    waiting_delay_seconds: float = 0.0
    status: str = "generated"
    drop_reason: str | None = None
    deadline_seconds: float | None = None
    route: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PacketSimulationResult:
    routing_policy: str
    generated_packets: int
    delivered_packets: int
    dropped_packets: int
    packet_delivery_ratio: float | None
    average_end_to_end_delay_seconds: float | None
    average_propagation_delay_seconds: float | None
    average_transmission_delay_seconds: float | None
    average_processing_delay_seconds: float | None
    average_waiting_delay_seconds: float | None
    average_hop_count: float | None
    total_communication_energy_j: float
    total_compute_energy_j: float
    communication_energy_by_satellite_j: tuple[tuple[str, float], ...]
    compute_energy_by_satellite_j: tuple[tuple[str, float], ...]
    residual_energy_by_satellite_j: tuple[tuple[str, float], ...]
    dropped_by_reason: tuple[tuple[str, int], ...]
    packets: tuple[Packet, ...]
    event_log: tuple[dict, ...]


class PacketSimulator:
    """Single-radio, half-duplex packet simulation with stable event ordering.

    A satellite radio is reserved from processing start until packet arrival at
    the next node. Competing packets wait and retry in event insertion order.
    TX energy is charged at transmission start; RX energy at arrival. Both
    endpoints are preflighted before TX begins. Propagation occurs after TX
    completion, and packet forwarding cannot happen before the arrival event.
    """

    def __init__(
        self,
        trace: TopologyTrace,
        energy_config: RoutingEnvConfig,
        config: PacketSimulationConfig,
        *,
        dqn_policy: DQNPolicy | None = None,
        record_dqn_decisions: bool = False,
    ):
        if not trace.graphs or len(trace.graphs) != len(trace.times_utc):
            raise ValueError("Trace must contain one graph per sample time")
        if any(right <= left for left, right in zip(trace.times_utc, trace.times_utc[1:])):
            raise ValueError("Topology trace times must be strictly increasing")
        self.trace = trace
        self.energy_config = energy_config
        self.config = config
        self.dqn_policy = dqn_policy
        self.record_dqn_decisions = record_dqn_decisions
        self.start_time = trace.times_utc[0]
        self.sample_offsets = tuple((time - self.start_time).total_seconds() for time in trace.times_utc)
        self.satellite_nodes = tuple(
            sorted(
                node
                for node, attributes in trace.graphs[0].nodes(data=True)
                if attributes.get("kind") == "satellite"
            )
        )
        if not self.satellite_nodes:
            raise ValueError("Packet simulation requires satellite nodes")
        if config.horizon_seconds > self.sample_offsets[-1] + 1e-9:
            raise ValueError("Simulation horizon cannot exceed the topology trace horizon")

    def run(self, flows: tuple[PacketFlow, ...], routing_policy: RoutingPolicy) -> PacketSimulationResult:
        if routing_policy not in {"hops", "delay", "energy", "contact", "dqn"}:
            raise ValueError(f"Unsupported routing policy: {routing_policy}")
        if routing_policy == "dqn" and self.dqn_policy is None:
            raise ValueError("DQN routing requires a trained DQNPolicy")
        self._reset()
        sequence = count()
        for flow_index, flow in enumerate(flows):
            for packet_index, generation_time in enumerate(flow.generation_times_seconds):
                if generation_time <= self.config.horizon_seconds:
                    packet_id = f"f{flow_index}-p{packet_index}"
                    self._push(
                        generation_time,
                        next(sequence),
                        "generate",
                        packet_id,
                        (flow.source_satellite, flow.destination, flow.packet_size_bits, flow.deadline_seconds),
                    )

        while self.event_queue:
            event_time, _, event_type, packet_id, payload = heappop(self.event_queue)
            if event_time > self.config.horizon_seconds + 1e-9:
                break
            if event_type == "generate":
                self._generate(packet_id, payload, event_time, sequence)
            elif event_type == "route":
                self._route(packet_id, event_time, routing_policy, sequence)
            elif event_type == "transmission_start":
                self._transmission_start(packet_id, event_time, payload, sequence)
            elif event_type == "transmission_complete":
                self._transmission_complete(packet_id, event_time, payload, sequence)
            elif event_type == "arrival":
                self._arrival(packet_id, event_time, payload, routing_policy, sequence)
            else:
                raise RuntimeError(f"Unknown packet event {event_type!r}")

        for packet in self.packets.values():
            if packet.status not in {"delivered", "dropped"}:
                self._drop(packet, "simulation_horizon", self.config.horizon_seconds)
        return self._result(routing_policy)

    def _reset(self) -> None:
        self.initial_energy_j = {
            node: self.energy_config.energy_capacity_j for node in self.satellite_nodes
        }
        self.communication_energy_j = {node: 0.0 for node in self.satellite_nodes}
        self.compute_energy_j = {node: 0.0 for node in self.satellite_nodes}
        self.radio_available_at_seconds = {
            node: 0.0
            for node in set(self.satellite_nodes).union(
                self.trace.graphs[0].nodes
            )
        }
        self.event_queue: list[tuple[float, int, str, str, object]] = []
        self.packets: dict[str, Packet] = {}
        self.event_log: list[dict] = []
        self.dqn_decision_trace: list[dict] = []
        self.contact_plan_cache = {}
        self.contact_graph_cache = {}
        self.dqn_future_route_cache = {}

    @staticmethod
    def _push(event_queue, event_time, order, event_type, packet_id, payload=None) -> None:
        heappush(event_queue, (event_time, order, event_type, packet_id, payload))

    def _push(self, event_time, order, event_type, packet_id, payload=None) -> None:
        heappush(self.event_queue, (event_time, order, event_type, packet_id, payload))

    def _timestamp(self, seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=seconds)

    def _log(self, time_seconds: float, packet: Packet, event: str, **details) -> None:
        self.event_log.append(
            {
                "time_utc": self._timestamp(time_seconds).isoformat(),
                "time_seconds": time_seconds,
                "packet_id": packet.packet_id,
                "event": event,
                "node": packet.current_satellite,
                **details,
            }
        )

    def _generate(self, packet_id, payload, time_seconds, sequence) -> None:
        source, destination, packet_bits, deadline = payload
        if source not in self.satellite_nodes:
            raise ValueError(f"Unknown source satellite {source!r}")
        packet = Packet(
            packet_id=packet_id,
            source_satellite=source,
            destination=destination,
            generation_time=self._timestamp(time_seconds),
            packet_size_bits=packet_bits,
            current_satellite=source,
            deadline_seconds=deadline,
            route=[source],
        )
        self.packets[packet_id] = packet
        self._log(time_seconds, packet, "packet_generated", destination=destination, packet_bits=packet_bits)
        self._push(time_seconds, next(sequence), "route", packet_id)

    def _sample_index(self, time_seconds: float) -> int:
        return min(max(bisect_right(self.sample_offsets, time_seconds) - 1, 0), len(self.sample_offsets) - 1)

    def _edge_delay(self, edge: dict, packet_bits: int) -> tuple[float, float, float]:
        distance_km = float(edge.get("range_km", 0.0))
        propagation = float(edge.get("propagation_s", distance_km / 299_792.458))
        transmission = float(edge.get("transmission_s", packet_bits / self.energy_config.transmission_rate_bps))
        processing = float(edge.get("processing_s", self.config.processing_delay_seconds))
        if any(not isfinite(value) or value < 0.0 for value in (propagation, transmission, processing)):
            raise ValueError("Packet delay components must be finite and non-negative")
        return propagation, transmission, processing

    def _edge_energy(self, edge: dict, packet_bits: int) -> tuple[float, float]:
        distance_km = float(edge.get("range_km", 0.0))
        transmit = packet_bits * (
            self.energy_config.tx_electronics_j_per_bit
            + self.energy_config.amplifier_j_per_bit_km2 * distance_km**2
        )
        receive = packet_bits * self.energy_config.rx_electronics_j_per_bit
        return transmit, receive

    def _choose_next_hop(self, packet: Packet, time_seconds: float, policy: RoutingPolicy):
        index = self._sample_index(time_seconds)
        graph = self.trace.graphs[index]
        current = packet.current_satellite
        if current == packet.destination:
            return "delivered", None
        if policy == "contact":
            cache_key = (current, packet.destination, index, packet.packet_size_bits)
            if cache_key not in self.contact_plan_cache:
                self.contact_plan_cache[cache_key] = contact_graph_route(
                    self.trace,
                    current,
                    packet.destination,
                    start_index=index,
                    transmission_rate_bps=self.energy_config.transmission_rate_bps,
                    packet_bits=packet.packet_size_bits,
                    initial_energy_by_node_j=self.energies,
                    _graph_cache=self.contact_graph_cache,
                    _route_cache=self.contact_plan_cache,
                )
            plan = self.contact_plan_cache[cache_key]
            if plan is None or len(plan.states) < 2:
                return None, None
            next_node, next_index = plan.states[1]
            if next_node == current:
                target_index = max(index + 1, next_index)
                if target_index < len(self.sample_offsets):
                    return "wait", self.sample_offsets[target_index]
                return None, None
            return next_node, None
        if policy == "dqn":
            return self._select_dqn_next_hop(packet, graph, time_seconds)
        if policy == "hops":
            path = shortest_path_route(graph, current, packet.destination, metric="hops")
        elif policy == "delay":
            path = shortest_path_route(
                graph,
                current,
                packet.destination,
                metric="delay",
                transmission_rate_bps=self.energy_config.transmission_rate_bps,
                packet_bits=packet.packet_size_bits,
            )
        else:
            energy_graph = graph.copy()
            for node in energy_graph:
                energy_graph.nodes[node]["compute_energy_j"] = self.compute_energy_j.get(node, 0.0)
            for node_a, node_b, edge in energy_graph.edges(data=True):
                edge["tx_energy_j"], edge["rx_energy_j"] = self._edge_energy(edge, packet.packet_size_bits)
            budgets = dict(self.energies)
            budgets.update({node: 1e100 for node in energy_graph if node not in budgets})
            path = energy_constrained_route(
                energy_graph,
                current,
                packet.destination,
                initial_energy_by_node_j=budgets,
                transmission_rate_bps=self.energy_config.transmission_rate_bps,
                packet_bits=packet.packet_size_bits,
            )
        if path is None or len(path) < 2:
            return None, None
        return path[1], None

    def _dqn_future_route(self, packet: Packet, next_index: int):
        next_time = self.sample_offsets[next_index]
        packet_age = next_time - (packet.generation_time - self.start_time).total_seconds()
        if packet.deadline_seconds is not None and packet_age > packet.deadline_seconds:
            return None
        key = (packet.current_satellite, packet.destination, next_index, packet.packet_size_bits)
        if key not in self.dqn_future_route_cache:
            self.dqn_future_route_cache[key] = contact_graph_route(
                self.trace,
                packet.current_satellite,
                packet.destination,
                start_index=next_index,
                transmission_rate_bps=self.energy_config.transmission_rate_bps,
                packet_bits=packet.packet_size_bits,
                _graph_cache=self.contact_graph_cache,
                _route_cache=self.contact_plan_cache,
            )
        route = self.dqn_future_route_cache[key]
        if (
            route is not None
            and packet.deadline_seconds is not None
            and packet_age + route.elapsed_seconds > packet.deadline_seconds
        ):
            return None
        return route

    def _select_dqn_next_hop(
        self,
        packet: Packet,
        graph: nx.Graph,
        time_seconds: float,
    ) -> tuple[str | None, float | None]:
        candidates = sorted(
            neighbor
            for neighbor in graph.neighbors(packet.current_satellite)
            if neighbor in graph
        )[: self.energy_config.max_neighbors]
        graph_index = self._sample_index(time_seconds)
        next_index = graph_index + 1
        next_sample_time = (
            self.sample_offsets[next_index]
            if next_index < len(self.sample_offsets)
            else None
        )
        future_route = (
            self._dqn_future_route(packet, next_index)
            if self.config.dqn_wait_action_enabled
            and next_sample_time is not None
            and next_sample_time <= self.config.horizon_seconds
            else None
        )
        wait_enabled = future_route is not None
        max_range_km = max(
            (float(edge.get("range_km", 0.0)) for _, _, edge in graph.edges(data=True)),
            default=1.0,
        )
        sample_index = graph_index
        interval = (
            self.sample_offsets[sample_index + 1] - self.sample_offsets[sample_index]
            if sample_index + 1 < len(self.sample_offsets)
            else self.sample_offsets[sample_index] - self.sample_offsets[sample_index - 1]
            if sample_index > 0
            else 1.0
        )
        local_energy = self.energies[packet.current_satellite] / self.energy_config.energy_capacity_j
        candidate_features: list[list[float]] = []
        for neighbor in candidates:
            edge = graph[packet.current_satellite][neighbor]
            distance_km = float(edge.get("range_km", 0.0))
            neighbor_energy = (
                self.energies.get(neighbor, 0.0) / self.energy_config.energy_capacity_j
            )
            if "link_quality" in edge:
                link_quality = float(np.clip(edge["link_quality"], 0.0, 1.0))
            else:
                link_quality = float(np.clip(1.0 - distance_km / max(max_range_km, 1.0), 0.0, 1.0))
            remaining_contact = float(edge.get("remaining_contact_s", interval))
            try:
                hops_to_destination = nx.shortest_path_length(
                    graph,
                    neighbor,
                    packet.destination,
                )
                destination_reachable = 1.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                hops_to_destination = self.config.max_hops
                destination_reachable = 0.0
            candidate_features.append(
                [
                    local_energy,
                    0.0,
                    neighbor_energy,
                    link_quality,
                    float(np.clip(remaining_contact / max(interval, 1e-9), 0.0, 1.0)),
                    float(np.clip(distance_km / max(max_range_km, 1.0), 0.0, 1.0)),
                    float(np.clip(hops_to_destination / self.config.max_hops, 0.0, 1.0)),
                    float(graph.nodes[neighbor].get("kind") == "ground_station"),
                    destination_reachable,
                    *(
                        [float(neighbor in packet.route)]
                        if self.energy_config.include_path_history
                        else []
                    ),
                ]
            )

        if self.config.dqn_wait_action_enabled:
            feature_count = 9 + int(self.energy_config.include_path_history)
            padded_candidate_features = candidate_features + [
                [0.0] * feature_count
                for _ in range(self.energy_config.max_neighbors - len(candidates))
            ]
            wait_features = [0.0] * feature_count
            if wait_enabled:
                local_energy = self.energies[packet.current_satellite] / self.energy_config.energy_capacity_j
                wait_features = [
                    local_energy,
                    0.0,
                    local_energy,
                    0.0,
                    float(np.clip((next_sample_time - time_seconds) / max(interval, 1e-9), 0.0, 1.0)),
                    float(
                        np.clip(
                            future_route.waiting_seconds
                            / max(self.config.horizon_seconds, 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    float(np.clip(future_route.hop_count / self.config.max_hops, 0.0, 1.0)),
                    0.0,
                    1.0,
                ]
                if self.energy_config.include_path_history:
                    wait_features.append(0.0)
            candidate_features = padded_candidate_features + [wait_features]
        else:
            feature_count = 9 + int(self.energy_config.include_path_history)
            candidate_features += [
                [0.0] * feature_count
                for _ in range(self.energy_config.max_neighbors - len(candidates))
            ]

        action_mask = np.zeros(len(candidate_features) + 1, dtype=np.int8)
        action_mask[: len(candidates)] = 1
        wait_action_index = self.energy_config.max_neighbors if self.config.dqn_wait_action_enabled else None
        if wait_enabled and wait_action_index is not None:
            action_mask[wait_action_index] = 1
        action_mask[-1] = 1

        policy = self.dqn_policy
        policy.eval()
        with torch.no_grad():
            action_values = policy(np.asarray(candidate_features, dtype=np.float32))
        if action_values.shape != (len(candidate_features) + 1,):
            raise ValueError("DQN output must contain one Q-value per action row and one DROP value")
        masked_action_values = action_values.clone()
        masked_action_values[
            torch.as_tensor(action_mask == 0, device=masked_action_values.device)
        ] = -torch.inf
        selection_action_values = action_values.clone()
        if self.config.cycle_avoidance_penalty:
            for action_index, neighbor in enumerate(candidates):
                if neighbor in packet.route:
                    selection_action_values[action_index] -= (
                        self.config.cycle_avoidance_penalty
                    )
        selection_action_values[
            torch.as_tensor(action_mask == 0, device=selection_action_values.device)
        ] = -torch.inf
        action = int(torch.argmax(selection_action_values).item())
        if action == len(action_mask) - 1:
            selected: str | None = "policy_drop"
            action_name = "DROP"
        elif wait_action_index is not None and action == wait_action_index:
            selected = "wait"
            action_name = "WAIT"
        else:
            selected = candidates[action]
            action_name = selected

        if self.record_dqn_decisions:
            self.dqn_decision_trace.append(
                {
                    "packet_id": packet.packet_id,
                    "decision_index": sum(
                        decision["packet_id"] == packet.packet_id
                        for decision in self.dqn_decision_trace
                    ),
                    "time_seconds": time_seconds,
                    "time_utc": self._timestamp(time_seconds).isoformat(),
                    "graph_index": graph_index,
                    "current_satellite": packet.current_satellite,
                    "destination": packet.destination,
                    "packet_age_seconds": packet.accumulated_delay_seconds,
                    "hop_count": packet.hop_count,
                    "deadline_seconds": packet.deadline_seconds,
                    "candidate_next_hops": candidates,
                    "candidate_features": candidate_features,
                    "action_mask": action_mask.tolist(),
                    "q_values": action_values.detach().cpu().tolist(),
                    "masked_q_values": [
                        float(masked_action_values[index].item())
                        if action_mask[index]
                        else None
                        for index in range(len(action_mask))
                    ],
                    "selection_q_values": [
                        float(selection_action_values[index].item())
                        if action_mask[index]
                        else None
                        for index in range(len(action_mask))
                    ],
                    "selected_action_index": action,
                    "selected_action": action_name,
                    "epsilon": None,
                    "reward": None,
                    "reward_components": None,
                    "next_state": None,
                    "future_route_available": wait_enabled,
                    "future_route_states": (
                        [list(state) for state in future_route.states]
                        if future_route is not None
                        else None
                    ),
                }
            )
        if selected == "wait":
            return selected, next_sample_time
        return selected, None

    def _route(self, packet_id, time_seconds, policy, sequence) -> None:
        packet = self.packets[packet_id]
        if packet.status in {"delivered", "dropped"}:
            return
        packet.accumulated_delay_seconds = time_seconds - (
            packet.generation_time - self.start_time
        ).total_seconds()
        if packet.deadline_seconds is not None and packet.accumulated_delay_seconds > packet.deadline_seconds:
            self._drop(packet, "deadline_expired", time_seconds)
            return
        selected, wait_until = self._choose_next_hop(packet, time_seconds, policy)
        if selected == "delivered":
            self._deliver(packet, time_seconds)
            return
        if selected == "wait":
            self._wait(packet, time_seconds, wait_until, "future_contact", sequence)
            return
        if selected == "policy_drop":
            self._drop(packet, "policy_drop", time_seconds)
            return
        if selected is None:
            reason = "contact_expired" if packet.status == "waiting_for_contact" else "no_feasible_route"
            self._drop(packet, reason, time_seconds)
            return

        graph_index = self._sample_index(time_seconds)
        graph = self.trace.graphs[graph_index]
        if selected not in graph or not graph.has_edge(packet.current_satellite, selected):
            self._drop(packet, "link_unavailable", time_seconds)
            return
        edge = graph[packet.current_satellite][selected]
        propagation, transmission, processing = self._edge_delay(edge, packet.packet_size_bits)
        transmit_energy, receive_energy = self._edge_energy(edge, packet.packet_size_bits)
        slot_end = (
            self.sample_offsets[graph_index + 1]
            if graph_index + 1 < len(self.sample_offsets)
            else self.config.horizon_seconds
        )
        window_end = time_seconds + max(
            0.0,
            float(edge.get("remaining_contact_s", slot_end - self.sample_offsets[graph_index]))
            - (time_seconds - self.sample_offsets[graph_index]),
        )
        if time_seconds + processing + transmission > window_end + 1e-9:
            if graph_index + 1 < len(self.sample_offsets) and self.sample_offsets[graph_index + 1] <= self.config.horizon_seconds:
                packet.status = "waiting_for_contact"
                self._wait(
                    packet,
                    time_seconds,
                    self.sample_offsets[graph_index + 1],
                    "contact_too_short",
                    sequence,
                )
            else:
                self._drop(packet, "contact_too_short", time_seconds)
            return

        radio_ready = max(
            self.radio_available_at_seconds.get(packet.current_satellite, 0.0),
            self.radio_available_at_seconds.get(selected, 0.0),
        )
        if radio_ready > time_seconds + 1e-9:
            self._wait(packet, time_seconds, radio_ready, "radio_contention", sequence)
            return
        if processing > 0.0 or self.config.processing_energy_j_per_hop > 0.0:
            if self.energies[packet.current_satellite] < self.config.processing_energy_j_per_hop:
                self._drop(packet, "compute_energy_depletion", time_seconds)
                return
            self.compute_energy_j[packet.current_satellite] += self.config.processing_energy_j_per_hop
            packet.processing_delay_seconds += processing
        if self.energies[packet.current_satellite] < transmit_energy:
            self._drop(packet, "sender_energy_depletion", time_seconds)
            return
        if selected in self.energies and self.energies[selected] < receive_energy:
            self._drop(packet, "receiver_energy_depletion", time_seconds)
            return

        arrival_time = time_seconds + processing + transmission + propagation
        self.radio_available_at_seconds[packet.current_satellite] = arrival_time
        self.radio_available_at_seconds[selected] = arrival_time
        packet.status = "processing" if processing > 0.0 else "transmitting"
        self._log(
            time_seconds,
            packet,
            "route_selected",
            next_hop=selected,
            policy=policy,
        )
        self._log(
            time_seconds,
            packet,
            "processing_started",
            processing_seconds=processing,
            processing_energy_j=self.config.processing_energy_j_per_hop,
        ) if processing > 0.0 or self.config.processing_energy_j_per_hop > 0.0 else None
        self._push(
            time_seconds + processing,
            next(sequence),
            "transmission_start",
            packet_id,
            (selected, transmission, propagation, transmit_energy, receive_energy),
        )

    def _transmission_start(self, packet_id, time_seconds, payload, sequence) -> None:
        packet = self.packets[packet_id]
        next_node, transmission, propagation, transmit_energy, receive_energy = payload
        if self.energies[packet.current_satellite] < transmit_energy:
            self._release_radios(packet, next_node, time_seconds)
            self._drop(packet, "sender_energy_depletion", time_seconds)
            return
        if next_node in self.energies and self.energies[next_node] < receive_energy:
            self._release_radios(packet, next_node, time_seconds)
            self._drop(packet, "receiver_energy_depletion", time_seconds)
            return
        self.communication_energy_j[packet.current_satellite] += transmit_energy
        packet.transmission_delay_seconds += transmission
        packet.status = "transmitting"
        self._log(
            time_seconds,
            packet,
            "transmission_started",
            next_hop=next_node,
            transmission_seconds=transmission,
            propagation_seconds=propagation,
            transmit_energy_j=transmit_energy,
        )
        self._push(
            time_seconds + transmission,
            next(sequence),
            "transmission_complete",
            packet_id,
            (next_node, propagation, receive_energy),
        )

    def _transmission_complete(self, packet_id, time_seconds, payload, sequence) -> None:
        packet = self.packets[packet_id]
        next_node, propagation, receive_energy = payload
        packet.status = "propagating"
        self._log(time_seconds, packet, "transmission_completed", next_hop=next_node)
        self._push(
            time_seconds + propagation,
            next(sequence),
            "arrival",
            packet_id,
            (next_node, receive_energy, propagation),
        )

    def _arrival(self, packet_id, time_seconds, payload, policy, sequence) -> None:
        packet = self.packets[packet_id]
        next_node, receive_energy, propagation = payload
        if next_node in self.communication_energy_j:
            self.communication_energy_j[next_node] += receive_energy
        packet.propagation_delay_seconds += propagation
        packet.accumulated_delay_seconds = time_seconds - (
            packet.generation_time - self.start_time
        ).total_seconds()
        packet.current_satellite = next_node
        packet.hop_count += 1
        packet.route.append(next_node)
        packet.status = "arrived"
        self._log(time_seconds, packet, "packet_arrived", receive_energy_j=receive_energy)
        if time_seconds - (packet.generation_time - self.start_time).total_seconds() > (
            packet.deadline_seconds if packet.deadline_seconds is not None else float("inf")
        ):
            self._drop(packet, "deadline_expired", time_seconds)
        elif next_node == packet.destination:
            self._deliver(packet, time_seconds)
        elif packet.hop_count >= self.config.max_hops:
            self._drop(packet, "hop_limit", time_seconds)
        else:
            self._push(time_seconds, next(sequence), "route", packet_id)

    def _wait(self, packet, now, until, reason, sequence) -> None:
        if until <= now + 1e-9:
            self._drop(packet, "no_feasible_route", now)
            return
        if until > self.config.horizon_seconds + 1e-9:
            self._drop(packet, "simulation_horizon", self.config.horizon_seconds)
            return
        waiting = until - now
        packet.waiting_delay_seconds += waiting
        packet.status = "waiting_for_contact" if reason != "radio_contention" else "waiting_for_radio"
        self._log(now, packet, "packet_waiting", until_seconds=until, reason=reason)
        self._push(until, next(sequence), "route", packet.packet_id)

    def _release_radios(self, packet, next_node, time_seconds) -> None:
        self.radio_available_at_seconds[packet.current_satellite] = time_seconds
        self.radio_available_at_seconds[next_node] = time_seconds

    def _deliver(self, packet: Packet, time_seconds: float) -> None:
        packet.status = "delivered"
        packet.accumulated_delay_seconds = time_seconds - (
            packet.generation_time - self.start_time
        ).total_seconds()
        self._log(time_seconds, packet, "packet_delivered")

    def _drop(self, packet: Packet, reason: str, time_seconds: float) -> None:
        if packet.status in {"delivered", "dropped"}:
            return
        packet.status = "dropped"
        packet.drop_reason = reason
        packet.accumulated_delay_seconds = time_seconds - (
            packet.generation_time - self.start_time
        ).total_seconds()
        self._log(time_seconds, packet, "packet_dropped", reason=reason)

    @property
    def energies(self) -> dict[str, float]:
        return {
            node: self.initial_energy_j[node]
            - self.communication_energy_j[node]
            - self.compute_energy_j[node]
            for node in self.satellite_nodes
        }

    def _result(self, routing_policy: RoutingPolicy) -> PacketSimulationResult:
        packets = tuple(self.packets[key] for key in sorted(self.packets))
        delivered = [packet for packet in packets if packet.status == "delivered"]
        dropped = [packet for packet in packets if packet.status == "dropped"]
        drop_counts: dict[str, int] = {}
        for packet in dropped:
            reason = packet.drop_reason or "unknown"
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
        generated_count = len(packets)
        delivered_count = len(delivered)
        return PacketSimulationResult(
            routing_policy=routing_policy,
            generated_packets=generated_count,
            delivered_packets=delivered_count,
            dropped_packets=len(dropped),
            packet_delivery_ratio=(
                delivered_count / generated_count if generated_count else None
            ),
            average_end_to_end_delay_seconds=(
                sum(packet.accumulated_delay_seconds for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            average_propagation_delay_seconds=(
                sum(packet.propagation_delay_seconds for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            average_transmission_delay_seconds=(
                sum(packet.transmission_delay_seconds for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            average_processing_delay_seconds=(
                sum(packet.processing_delay_seconds for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            average_waiting_delay_seconds=(
                sum(packet.waiting_delay_seconds for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            average_hop_count=(
                sum(packet.hop_count for packet in delivered) / delivered_count
                if delivered_count
                else None
            ),
            total_communication_energy_j=sum(self.communication_energy_j.values()),
            total_compute_energy_j=sum(self.compute_energy_j.values()),
            communication_energy_by_satellite_j=tuple(sorted(self.communication_energy_j.items())),
            compute_energy_by_satellite_j=tuple(sorted(self.compute_energy_j.items())),
            residual_energy_by_satellite_j=tuple(sorted(self.energies.items())),
            dropped_by_reason=tuple(sorted(drop_counts.items())),
            packets=packets,
            event_log=tuple(self.event_log),
        )
