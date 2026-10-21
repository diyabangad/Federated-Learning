"""Deterministic routing references for sampled contact graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappop, heappush
from math import ceil, isfinite

import networkx as nx

from fedroute.topology import TopologyTrace


@dataclass(frozen=True)
class ContactGraphRoute:
    states: tuple[tuple[str, int], ...]
    arrival_time_utc: datetime
    elapsed_seconds: float
    hop_count: int
    waiting_seconds: float
    propagation_seconds: float
    transmission_seconds: float
    processing_seconds: float
    communication_energy_j: float | None
    compute_energy_j: float | None
    residual_energy_j: tuple[tuple[str, float], ...] | None

    @property
    def total_delay_seconds(self) -> float:
        return (
            self.propagation_seconds
            + self.transmission_seconds
            + self.processing_seconds
            + self.waiting_seconds
        )


@dataclass(frozen=True)
class RouteMetrics:
    hop_count: int
    propagation_seconds: float
    transmission_seconds: float
    total_delay_seconds: float
    waiting_seconds: float = 0.0
    processing_seconds: float = 0.0
    communication_energy_j: float | None = None
    compute_energy_j: float | None = None
    residual_energy_j: tuple[tuple[str, float], ...] | None = None


def _validate_link_cost_inputs(transmission_rate_bps: float, packet_bits: int) -> None:
    if not isfinite(transmission_rate_bps) or transmission_rate_bps <= 0:
        raise ValueError("transmission_rate_bps must be finite and positive")
    if packet_bits <= 0:
        raise ValueError("packet_bits must be positive")


def _delay_components(
    edge: dict,
    transmission_rate_bps: float,
    packet_bits: int,
) -> tuple[float, float, float]:
    distance_km = float(edge.get("range_km", 0.0))
    if not isfinite(distance_km) or distance_km < 0.0:
        raise ValueError("Edge range_km must be finite and non-negative")
    propagation = float(edge.get("propagation_s", distance_km / 299_792.458))
    transmission = float(edge.get("transmission_s", packet_bits / transmission_rate_bps))
    processing = float(edge.get("processing_s", 0.0))
    if any(not isfinite(value) or value < 0.0 for value in (propagation, transmission, processing)):
        raise ValueError("Edge delay components must be finite and non-negative")
    return propagation, transmission, processing


def _delay_cost(edge: dict, transmission_rate_bps: float, packet_bits: int) -> float:
    return sum(_delay_components(edge, transmission_rate_bps, packet_bits))


def _deterministic_shortest_path(graph: nx.Graph, source: str, destination: str, weight) -> list[str] | None:
    queue = [(0.0, (source,), source)]
    best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, (source,))}
    while queue:
        total_cost, path, node = heappop(queue)
        if best.get(node) != (total_cost, path):
            continue
        if node == destination:
            return list(path)
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor in path:
                continue
            edge_cost = 1.0 if weight is None else float(weight(node, neighbor, graph[node][neighbor]))
            if not isfinite(edge_cost) or edge_cost < 0.0:
                raise ValueError("Route edge costs must be finite and non-negative")
            candidate_cost = total_cost + edge_cost
            candidate_path = (*path, neighbor)
            previous = best.get(neighbor)
            if previous is None or (candidate_cost, candidate_path) < previous:
                best[neighbor] = (candidate_cost, candidate_path)
                heappush(queue, (candidate_cost, candidate_path, neighbor))
    return None


def shortest_path_route(
    graph: nx.Graph,
    source: str,
    destination: str,
    *,
    metric: str = "hops",
    transmission_rate_bps: float | None = None,
    packet_bits: int = 4096,
) -> list[str] | None:
    """Find a route in one graph snapshot by hop count or propagation delay."""
    if metric not in {"hops", "delay"}:
        raise ValueError("metric must be 'hops' or 'delay'")
    if source not in graph or destination not in graph:
        return None
    if metric == "hops":
        weight = None
    else:
        if transmission_rate_bps is None:
            raise ValueError("transmission_rate_bps is required for delay routing")
        _validate_link_cost_inputs(transmission_rate_bps, packet_bits)
        weight = lambda _source, _destination, edge: _delay_cost(
            edge, transmission_rate_bps, packet_bits
        )
    return _deterministic_shortest_path(graph, source, destination, weight)


def calculate_route_metrics(
    graph: nx.Graph,
    route: list[str] | tuple[str, ...],
    *,
    transmission_rate_bps: float,
    packet_bits: int = 4096,
    waiting_seconds: float = 0.0,
    initial_energy_by_node_j: dict[str, float] | None = None,
) -> RouteMetrics:
    """Calculate hand-checkable propagation and transmission delay totals."""
    _validate_link_cost_inputs(transmission_rate_bps, packet_bits)
    if not isfinite(waiting_seconds) or waiting_seconds < 0.0:
        raise ValueError("waiting_seconds must be finite and non-negative")
    propagation_seconds = 0.0
    transmission_seconds = 0.0
    processing_seconds = 0.0
    communication_energy_available = True
    communication_energy_j = 0.0
    communication_by_node: dict[str, float] = {}
    for node_a, node_b in zip(route, route[1:]):
        if not graph.has_edge(node_a, node_b):
            raise ValueError(f"Route uses unavailable edge {node_a!r}-{node_b!r}")
        edge = graph[node_a][node_b]
        propagation, transmission, processing = _delay_components(
            edge,
            transmission_rate_bps,
            packet_bits,
        )
        propagation_seconds += propagation
        transmission_seconds += transmission
        processing_seconds += processing
        tx_energy = edge.get("tx_energy_j")
        rx_energy = edge.get("rx_energy_j")
        if tx_energy is None or rx_energy is None:
            communication_energy_available = False
        else:
            tx_energy = float(tx_energy)
            rx_energy = float(rx_energy)
            if any(not isfinite(value) or value < 0.0 for value in (tx_energy, rx_energy)):
                raise ValueError("Edge TX/RX energy must be finite and non-negative")
            communication_energy_j += tx_energy + rx_energy
            communication_by_node[node_a] = communication_by_node.get(node_a, 0.0) + tx_energy
            communication_by_node[node_b] = communication_by_node.get(node_b, 0.0) + rx_energy
    route_nodes = tuple(dict.fromkeys(route))
    compute_energy_j = 0.0
    compute_energy_available = True
    for node in route_nodes:
        if "compute_energy_j" not in graph.nodes[node]:
            compute_energy_available = False
            continue
        energy = float(graph.nodes[node]["compute_energy_j"])
        if not isfinite(energy) or energy < 0.0:
            raise ValueError("Node compute_energy_j must be finite and non-negative")
        compute_energy_j += energy
    residual_energy = None
    if initial_energy_by_node_j is not None and communication_energy_available and compute_energy_available:
        residual_energy = []
        for node in route_nodes:
            if node not in initial_energy_by_node_j:
                raise ValueError(f"Missing initial energy for route node {node!r}")
            initial = float(initial_energy_by_node_j[node])
            if not isfinite(initial) or initial < 0.0:
                raise ValueError("Initial energy must be finite and non-negative")
            comm = communication_by_node.get(node, 0.0)
            compute = float(graph.nodes[node].get("compute_energy_j", 0.0))
            remaining = initial - comm - compute
            if remaining < 0.0:
                raise ValueError(f"Route exceeds initial energy for node {node!r}")
            residual_energy.append((node, remaining))
    return RouteMetrics(
        hop_count=max(0, len(route) - 1),
        propagation_seconds=propagation_seconds,
        transmission_seconds=transmission_seconds,
        total_delay_seconds=(
            propagation_seconds
            + transmission_seconds
            + processing_seconds
            + waiting_seconds
        ),
        waiting_seconds=waiting_seconds,
        processing_seconds=processing_seconds,
        communication_energy_j=(
            communication_energy_j if communication_energy_available else None
        ),
        compute_energy_j=compute_energy_j if compute_energy_available else None,
        residual_energy_j=tuple(residual_energy) if residual_energy is not None else None,
    )


def energy_constrained_route(
    graph: nx.Graph,
    source: str,
    destination: str,
    *,
    initial_energy_by_node_j: dict[str, float],
    transmission_rate_bps: float,
    packet_bits: int = 4096,
    metric: str = "hops",
) -> list[str] | None:
    """Select a deterministic simple route whose annotated node budgets suffice.

    Intended for small baseline graphs: all simple paths are enumerated. Each
    route requires per-edge ``tx_energy_j``/``rx_energy_j`` and per-node
    ``compute_energy_j`` annotations so feasibility is never inferred from
    missing data.
    """
    if metric not in {"hops", "delay"}:
        raise ValueError("metric must be 'hops' or 'delay'")
    _validate_link_cost_inputs(transmission_rate_bps, packet_bits)
    if source not in graph or destination not in graph:
        return None

    ordered_graph = nx.Graph()
    for node, attributes in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        ordered_graph.add_node(node, **attributes)
    for node_a, node_b, attributes in sorted(
        graph.edges(data=True),
        key=lambda edge: tuple(sorted((str(edge[0]), str(edge[1])))),
    ):
        ordered_graph.add_edge(node_a, node_b, **attributes)
    if metric == "hops":
        weight = None
    else:
        weight = lambda node_a, node_b, edge: _delay_cost(
            edge,
            transmission_rate_bps,
            packet_bits,
        )
    try:
        candidate_paths = nx.shortest_simple_paths(
            ordered_graph,
            source,
            destination,
            weight=weight,
        )
        for path in candidate_paths:
            try:
                metrics = calculate_route_metrics(
                    graph,
                    path,
                    transmission_rate_bps=transmission_rate_bps,
                    packet_bits=packet_bits,
                    initial_energy_by_node_j=initial_energy_by_node_j,
                )
            except ValueError as error:
                if str(error).startswith("Route exceeds initial energy"):
                    continue
                raise
            if (
                metrics.communication_energy_j is None
                or metrics.compute_energy_j is None
                or metrics.residual_energy_j is None
            ):
                raise ValueError(
                    "Energy-constrained routes require complete edge and node energy annotations"
                )
            return path
    except nx.NetworkXNoPath:
        return None
    return None


def _build_time_expanded_contact_graph(
    trace: TopologyTrace,
    transmission_rate_bps: float,
    packet_bits: int,
) -> tuple[nx.DiGraph, dict[str, object]]:
    time_expanded = nx.DiGraph()
    destination_sinks: dict[str, object] = {}
    for index, graph in enumerate(trace.graphs):
        for node in graph.nodes:
            state = (node, index)
            time_expanded.add_node(state)
            sink = destination_sinks.setdefault(node, object())
            time_expanded.add_edge(state, sink, weight=0.0, kind="finish")
            if index + 1 < len(trace.graphs) and node in trace.graphs[index + 1]:
                wait_seconds = (
                    trace.times_utc[index + 1] - trace.times_utc[index]
                ).total_seconds()
                if wait_seconds <= 0.0:
                    raise ValueError("Trace sample times must be strictly increasing")
                time_expanded.add_edge(
                    state,
                    (node, index + 1),
                    weight=wait_seconds,
                    kind="wait",
                )
        for node_a, node_b, edge in graph.edges(data=True):
            propagation, transmission, processing = _delay_components(
                edge,
                transmission_rate_bps,
                packet_bits,
            )
            delay = propagation + transmission + processing
            interval = (
                (trace.times_utc[index + 1] - trace.times_utc[index]).total_seconds()
                if index + 1 < len(trace.times_utc)
                else (trace.times_utc[index] - trace.times_utc[index - 1]).total_seconds()
                if index > 0
                else 1.0
            )
            remaining_contact = float(edge.get("remaining_contact_s", interval))
            if not isfinite(remaining_contact) or remaining_contact < 0.0 or delay > remaining_contact:
                continue
            slot_count = max(1, ceil(delay / interval))
            next_index = index + slot_count
            for source_node, target_node in ((node_a, node_b), (node_b, node_a)):
                if target_node.startswith("gs:"):
                    time_expanded.add_edge(
                        (source_node, index),
                        (target_node, index),
                        weight=delay,
                        kind="hop",
                        propagation_seconds=propagation,
                        transmission_seconds=transmission,
                        processing_seconds=processing,
                        tx_energy_j=edge.get("tx_energy_j"),
                        rx_energy_j=edge.get("rx_energy_j"),
                    )
                elif next_index < len(trace.graphs) and target_node in trace.graphs[next_index]:
                    elapsed_to_next_decision = (
                        trace.times_utc[next_index] - trace.times_utc[index]
                    ).total_seconds()
                    time_expanded.add_edge(
                        (source_node, index),
                        (target_node, next_index),
                        weight=elapsed_to_next_decision,
                        kind="hop",
                        propagation_seconds=propagation,
                        transmission_seconds=transmission,
                        processing_seconds=processing,
                        slot_wait_seconds=max(0.0, elapsed_to_next_decision - delay),
                        tx_energy_j=edge.get("tx_energy_j"),
                        rx_energy_j=edge.get("rx_energy_j"),
                    )
    return time_expanded, destination_sinks


def contact_graph_route(
    trace: TopologyTrace,
    source: str,
    destination: str,
    *,
    start_index: int = 0,
    transmission_rate_bps: float,
    packet_bits: int = 4096,
    initial_energy_by_node_j: dict[str, float] | None = None,
    _graph_cache: dict | None = None,
    _route_cache: dict | None = None,
) -> ContactGraphRoute | None:
    """Find a route over sampled contacts, with at most one in-flight hop.

    A link is usable only if it is active at the departure sample, its remaining
    contact covers the packet's propagation plus transmission delay, and its
    endpoints exist at the modeled arrival sample. Intermediate forwarding
    decisions occur only at sample boundaries; an in-flight hop occupies
    ceil(delay / interval) sample intervals. The final delivery uses its exact
    propagation plus transmission delay. Links lacking explicit remaining
    contact metadata are conservatively assumed to last one sample interval.
    """
    _validate_link_cost_inputs(transmission_rate_bps, packet_bits)
    if len(trace.graphs) != len(trace.times_utc) or not trace.graphs:
        raise ValueError("Topology trace must have a graph for each sample time")
    if not 0 <= start_index < len(trace.graphs):
        raise ValueError("start_index is outside the topology trace")
    if source not in trace.graphs[start_index] or destination not in set().union(
        *(set(graph.nodes) for graph in trace.graphs[start_index:])
    ):
        return None

    cache_key = (id(trace), transmission_rate_bps, packet_bits)
    cached_graph = _graph_cache.get(cache_key) if _graph_cache is not None else None
    if cached_graph is None:
        time_expanded, destination_sinks = _build_time_expanded_contact_graph(
            trace,
            transmission_rate_bps,
            packet_bits,
        )
        if _graph_cache is not None:
            _graph_cache[cache_key] = (time_expanded, destination_sinks)
    else:
        time_expanded, destination_sinks = cached_graph

    start = (source, start_index)
    if start not in time_expanded:
        return None
    sink = destination_sinks.get(destination)
    if sink is None:
        return None
    route_cache_key = (cache_key, destination)
    reverse_paths = _route_cache.get(route_cache_key) if _route_cache is not None else None
    if reverse_paths is None:
        reverse_graph = time_expanded.reverse(copy=False)
        _, reverse_paths = nx.single_source_dijkstra(
            reverse_graph,
            sink,
            weight="weight",
        )
        if _route_cache is not None:
            _route_cache[route_cache_key] = reverse_paths
    reverse_path = reverse_paths.get(start)
    if reverse_path is None:
        return None
    path = list(reversed(reverse_path))
    states = tuple(path[:-1])
    elapsed_seconds = sum(
        time_expanded[source_state][target_state]["weight"]
        for source_state, target_state in zip(path, path[1:])
    )
    waiting_seconds = sum(
        time_expanded[source_state][target_state]["weight"]
        for source_state, target_state in zip(path, path[1:])
        if time_expanded[source_state][target_state]["kind"] == "wait"
    ) + sum(
        time_expanded[source_state][target_state].get("slot_wait_seconds", 0.0)
        for source_state, target_state in zip(path, path[1:])
    )
    propagation_seconds = sum(
        time_expanded[source_state][target_state].get("propagation_seconds", 0.0)
        for source_state, target_state in zip(path, path[1:])
    )
    transmission_seconds = sum(
        time_expanded[source_state][target_state].get("transmission_seconds", 0.0)
        for source_state, target_state in zip(path, path[1:])
    )
    processing_seconds = sum(
        time_expanded[source_state][target_state].get("processing_seconds", 0.0)
        for source_state, target_state in zip(path, path[1:])
    )
    hop_count = sum(
        time_expanded[source_state][target_state]["kind"] == "hop"
        for source_state, target_state in zip(path, path[1:])
    )
    communication_energy_available = True
    communication_energy_j = 0.0
    communication_by_node: dict[str, float] = {}
    for source_state, target_state in zip(path, path[1:]):
        edge = time_expanded[source_state][target_state]
        if edge.get("kind") != "hop":
            continue
        tx_energy = edge.get("tx_energy_j")
        rx_energy = edge.get("rx_energy_j")
        if tx_energy is None or rx_energy is None:
            communication_energy_available = False
            continue
        tx_energy = float(tx_energy)
        rx_energy = float(rx_energy)
        if any(not isfinite(value) or value < 0.0 for value in (tx_energy, rx_energy)):
            raise ValueError("Edge TX/RX energy must be finite and non-negative")
        communication_energy_j += tx_energy + rx_energy
        source_node = source_state[0]
        target_node = target_state[0]
        communication_by_node[source_node] = communication_by_node.get(source_node, 0.0) + tx_energy
        communication_by_node[target_node] = communication_by_node.get(target_node, 0.0) + rx_energy

    route_nodes = tuple(dict.fromkeys(state[0] for state in states))
    compute_energy_available = all(
        "compute_energy_j" in trace.graphs[index].nodes[node]
        for node, index in states
    )
    compute_energy_by_node = {
        node: max(
            float(graph.nodes[node].get("compute_energy_j", 0.0))
            for graph in trace.graphs
            if node in graph
        )
        for node in route_nodes
    }
    if any(not isfinite(value) or value < 0.0 for value in compute_energy_by_node.values()):
        raise ValueError("Node compute_energy_j must be finite and non-negative")
    compute_energy_j = (
        sum(compute_energy_by_node.values()) if compute_energy_available else None
    )
    residual_energy = None
    if (
        initial_energy_by_node_j is not None
        and communication_energy_available
        and compute_energy_available
    ):
        residual_energy = []
        for node in route_nodes:
            if node not in initial_energy_by_node_j:
                raise ValueError(f"Missing initial energy for route node {node!r}")
            initial = float(initial_energy_by_node_j[node])
            if not isfinite(initial) or initial < 0.0:
                raise ValueError("Initial energy must be finite and non-negative")
            remaining = (
                initial
                - communication_by_node.get(node, 0.0)
                - compute_energy_by_node[node]
            )
            if remaining < 0.0:
                raise ValueError(f"Route exceeds initial energy for node {node!r}")
            residual_energy.append((node, remaining))
    return ContactGraphRoute(
        states=states,
        arrival_time_utc=trace.times_utc[start_index] + timedelta(seconds=elapsed_seconds),
        elapsed_seconds=elapsed_seconds,
        hop_count=hop_count,
        waiting_seconds=waiting_seconds,
        propagation_seconds=propagation_seconds,
        transmission_seconds=transmission_seconds,
        processing_seconds=processing_seconds,
        communication_energy_j=(
            communication_energy_j if communication_energy_available else None
        ),
        compute_energy_j=compute_energy_j,
        residual_energy_j=tuple(residual_energy) if residual_energy is not None else None,
    )