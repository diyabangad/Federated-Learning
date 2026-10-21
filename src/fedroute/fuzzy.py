"""Fuzzy cluster-head selection for dynamic satellite networks."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import skfuzzy as fuzz


@dataclass(frozen=True)
class FuzzyClusterInput:
    residual_energy: float
    processing_capacity: float
    traffic_load: float
    contact_duration: float
    link_quality: float
    neighbor_count: float
    distance_to_ground: float


class FuzzyClusterSelector:
    """Two-stage Mamdani-style selector consistent with the proposal."""

    def __init__(self):
        self.value_levels = np.linspace(0.0, 1.0, 101)

    def _membership(self, value: float, low: tuple[float, float, float], mid: tuple[float, float, float], high: tuple[float, float, float]):
        low_term = fuzz.trimf(self.value_levels, low)
        mid_term = fuzz.trimf(self.value_levels, mid)
        high_term = fuzz.trimf(self.value_levels, high)
        return {
            "low": fuzz.interp_membership(self.value_levels, low_term, value),
            "medium": fuzz.interp_membership(self.value_levels, mid_term, value),
            "high": fuzz.interp_membership(self.value_levels, high_term, value),
        }

    def score(
        self,
        residual_energy: float,
        processing_capacity: float,
        traffic_load: float,
        contact_duration: float,
        link_quality: float,
        neighbor_count: float,
        distance_to_ground: float,
    ) -> float:
        capability = self._capability_score(
            residual_energy=residual_energy,
            processing_capacity=processing_capacity,
            traffic_load=traffic_load,
        )
        suitability = self._suitability_score(
            contact_duration=contact_duration,
            link_quality=link_quality,
            neighbor_count=neighbor_count,
        )
        proximity = 1.0 - min(max(distance_to_ground, 0.0), 1.0)
        return float(0.5 * capability + 0.35 * suitability + 0.15 * proximity)

    def _capability_score(self, residual_energy: float, processing_capacity: float, traffic_load: float) -> float:
        energy = self._membership(residual_energy, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))
        capacity = self._membership(processing_capacity, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))
        traffic = self._membership(1.0 - traffic_load, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))

        rules = [
            min(energy["high"], capacity["high"], traffic["high"]),
            min(energy["high"], capacity["medium"], traffic["high"]),
            min(energy["medium"], capacity["high"], traffic["medium"]),
            min(energy["medium"], capacity["medium"], traffic["high"]),
        ]
        return float(max(rules) if rules else 0.0)

    def _suitability_score(self, contact_duration: float, link_quality: float, neighbor_count: float) -> float:
        contact = self._membership(contact_duration, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))
        quality = self._membership(link_quality, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))
        neighbors = self._membership(neighbor_count, (0.0, 0.0, 0.5), (0.2, 0.5, 0.8), (0.5, 1.0, 1.0))

        rules = [
            min(contact["high"], quality["high"], neighbors["high"]),
            min(contact["high"], quality["medium"], neighbors["high"]),
            min(contact["medium"], quality["high"], neighbors["medium"]),
            min(contact["medium"], quality["medium"], neighbors["high"]),
        ]
        return float(max(rules) if rules else 0.0)


def select_cluster_heads(graph: nx.Graph, selector: FuzzyClusterSelector | None = None) -> dict[str, str]:
    """Return a satellite-to-head mapping for the current graph snapshot."""
    if selector is None:
        selector = FuzzyClusterSelector()

    satellites = [
        node for node, data in graph.nodes(data=True) if data.get("kind") == "satellite"
    ]
    if not satellites:
        return {}

    satellite_scores: dict[str, float] = {}
    for node in satellites:
        neighbors = list(graph.neighbors(node))
        contact_duration = max(
            (float(graph[node][neighbor].get("contact_duration_s", 0.0)) for neighbor in neighbors if neighbor in satellites),
            default=0.0,
        )
        link_quality = max(
            (float(graph[node][neighbor].get("link_quality", 0.5)) for neighbor in neighbors if neighbor in satellites),
            default=0.5,
        )
        distance_to_ground = min(
            (
                float(graph[node][neighbor].get("range_km", 1.0))
                for neighbor in neighbors
                if graph.nodes[neighbor].get("kind") == "ground_station"
            ),
            default=1.0,
        )
        satellite_scores[node] = selector.score(
            residual_energy=float(graph.nodes[node].get("residual_energy", 0.7)),
            processing_capacity=float(graph.nodes[node].get("processing_capacity", 0.7)),
            traffic_load=float(graph.nodes[node].get("traffic_load", 0.3)),
            contact_duration=min(max(contact_duration / 10000.0, 0.0), 1.0),
            link_quality=min(max(link_quality, 0.0), 1.0),
            neighbor_count=min(len(neighbors) / 10.0, 1.0),
            distance_to_ground=min(distance_to_ground / 10000.0, 1.0),
        )

    assigned: dict[str, str] = {}
    satellite_graph = graph.subgraph(satellites)
    unassigned = set(satellites)
    while unassigned:
        head = min(unassigned, key=lambda node: (-satellite_scores[node], node))
        members = {head} | (set(satellite_graph.neighbors(head)) & unassigned)
        assigned.update({member: head for member in members})
        unassigned.difference_update(members)
    return assigned
