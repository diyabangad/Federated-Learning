"""Masked next-hop routing environment for dynamic satellite networks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from fedroute.fuzzy import FuzzyClusterSelector, select_cluster_heads
from fedroute.topology import TopologyTrace


@dataclass(frozen=True)
class RoutingEnvConfig:
    energy_capacity_j: float
    tx_electronics_j_per_bit: float
    rx_electronics_j_per_bit: float
    amplifier_j_per_bit_km2: float
    transmission_rate_bps: float
    packet_bits: int = 4096
    max_neighbors: int = 8
    max_hops: int = 16
    delivery_reward: float = 10.0
    drop_penalty: float = 5.0
    energy_penalty: float = 0.5
    delay_penalty: float = 0.2

    def __post_init__(self) -> None:
        for value in (
            self.energy_capacity_j,
            self.transmission_rate_bps,
            self.packet_bits,
            self.max_neighbors,
            self.max_hops,
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError("Energy capacity, rate, and count parameters must be finite and positive")
        for value in (
            self.tx_electronics_j_per_bit,
            self.rx_electronics_j_per_bit,
            self.amplifier_j_per_bit_km2,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError("Energy coefficients must be finite and non-negative")


class SatelliteRoutingEnv(gym.Env):
    """Route a single packet across a dynamic orbital graph.

    The environment intentionally keeps the state compact and uses a masked action
    space, matching the proposal's control logic for next-hop selection under
    time-varying satellite connectivity.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: TopologyTrace,
        config: RoutingEnvConfig,
        *,
        cluster_aware: bool = False,
        cluster_selector: FuzzyClusterSelector | None = None,
    ):
        super().__init__()
        self.trace = trace
        self.config = config
        self.cluster_aware = cluster_aware
        self.cluster_assignments = tuple(
            select_cluster_heads(graph, cluster_selector)
            if cluster_aware
            else {}
            for graph in trace.graphs
        )
        self.satellite_nodes = tuple(
            sorted(
                node
                for node, data in trace.graphs[0].nodes(data=True)
                if data.get("kind") == "satellite"
            )
        )
        self.ground_station_nodes = tuple(
            sorted(
                node
                for node, data in trace.graphs[0].nodes(data=True)
                if data.get("kind") == "ground_station"
            )
        )
        if not self.satellite_nodes:
            raise ValueError("Routing environment requires at least one satellite node")
        if not self.ground_station_nodes:
            raise ValueError("Routing environment requires at least one ground station node")

        self.action_space = spaces.Discrete(config.max_neighbors + 1)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.max_neighbors, 9),
            dtype=np.float32,
        )

        self.source_node = self.satellite_nodes[0]
        self.destination_node = self.ground_station_nodes[0]
        self.current_node = self.source_node
        self.current_graph_index = 0
        self.hop_count = 0
        self.initial_energy_j = {
            node: config.energy_capacity_j for node in self.satellite_nodes
        }
        self.communication_energy_j = {node: 0.0 for node in self.satellite_nodes}
        self.compute_energy_j = {node: 0.0 for node in self.satellite_nodes}
        self._candidate_nodes: list[str] = []
        self._action_mask = np.zeros(config.max_neighbors + 1, dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        source = options.get("source", self.source_node)
        destination = options.get("destination", self.destination_node)
        graph_index = int(options.get("start_index", 0))

        if source not in self.satellite_nodes:
            raise ValueError("source must be a satellite node")
        if destination not in self.ground_station_nodes:
            raise ValueError("destination must be a ground-station node")
        if not 0 <= graph_index < len(self.trace.graphs):
            raise ValueError("start_index is outside the topology trace")

        self.current_node = source
        self.destination_node = destination
        self.current_graph_index = graph_index
        self.hop_count = 0
        self.communication_energy_j = {node: 0.0 for node in self.satellite_nodes}
        self.compute_energy_j = {node: 0.0 for node in self.satellite_nodes}
        obs = self._observation()
        return obs, self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space")

        action = int(action)
        if self._action_mask[action] == 0:
            raise ValueError(f"Action {action} is not feasible in the current state")

        if action == self.config.max_neighbors:
            reward = -self.config.drop_penalty
            terminated = True
            truncated = False
            loss_cause = (
                "voluntary_drop"
                if np.any(self._action_mask[:-1])
                else "no_feasible_next_hop"
            )
            info = self._info(loss_cause=loss_cause)
            return self._empty_observation(), reward, terminated, truncated, info

        next_node = self._candidate_nodes[action]
        graph = self.trace.graphs[self.current_graph_index]
        edge = graph[self.current_node][next_node]
        distance_km = float(edge.get("range_km", 1.0))
        transmission_cost = self.config.packet_bits * (
            self.config.tx_electronics_j_per_bit
            + self.config.amplifier_j_per_bit_km2 * distance_km**2
        )
        receiver_cost = self.config.packet_bits * self.config.rx_electronics_j_per_bit

        if self.energies[self.current_node] < transmission_cost:
            terminated = True
            reward = -self.config.drop_penalty
            info = self._info(loss_cause="energy_depletion")
            return self._empty_observation(), reward, terminated, False, info
        if (
            next_node in self.satellite_nodes
            and self.energies[next_node] < receiver_cost
        ):
            terminated = True
            reward = -self.config.drop_penalty
            info = self._info(loss_cause="receiver_energy_depletion")
            return self._empty_observation(), reward, terminated, False, info

        self.communication_energy_j[self.current_node] += transmission_cost
        if next_node in self.satellite_nodes:
            self.communication_energy_j[next_node] += receiver_cost

        self.hop_count += 1
        hop_delay = self._hop_delay(distance_km)
        link_quality = self._link_quality(edge)
        reward = (
            -self.config.energy_penalty * (transmission_cost / self.config.energy_capacity_j)
            - self.config.delay_penalty * (hop_delay / max(self._time_step_seconds(), 1.0))
            - self.config.delay_penalty * (1.0 - link_quality)
        )

        if next_node == self.destination_node:
            reward += self.config.delivery_reward
            terminated = True
            truncated = False
            info = self._info(delivered=True)
            return self._empty_observation(), reward, terminated, truncated, info

        self.current_node = next_node
        self.current_graph_index = min(self.current_graph_index + 1, len(self.trace.graphs) - 1)
        terminated = False
        truncated = self.hop_count >= self.config.max_hops
        if truncated:
            reward -= self.config.drop_penalty
        observation = self._observation()
        info = self._info()
        return observation, reward, terminated, truncated, info

    def _observation(self):
        graph = self.trace.graphs[self.current_graph_index]
        candidates: list[tuple[str, dict]] = []
        for neighbor in graph.neighbors(self.current_node):
            if neighbor in self.ground_station_nodes or neighbor in self.satellite_nodes:
                edge = graph[self.current_node][neighbor]
                candidates.append((neighbor, edge))
        candidates = sorted(candidates, key=lambda item: item[0])

        if len(candidates) > self.config.max_neighbors:
            candidates = candidates[: self.config.max_neighbors]

        self._candidate_nodes = [node for node, _ in candidates]
        self._action_mask = np.zeros(self.config.max_neighbors + 1, dtype=np.int8)
        observation = self._empty_observation()
        for index, (neighbor, edge) in enumerate(candidates):
            self._action_mask[index] = 1
            local_energy = self.energies[self.current_node] / self.config.energy_capacity_j
            neighbor_energy = self.energies.get(neighbor, 0.0) / self.config.energy_capacity_j
            link_quality = self._link_quality(edge)
            delay_norm = min(self._hop_delay(float(edge.get("range_km", 1.0))) / max(self._time_step_seconds(), 1.0), 1.0)
            remaining_contact = float(edge.get("remaining_contact_s", self._time_step_seconds()))
            distance_norm = min(float(edge.get("range_km", 1.0)) / max(self._max_range_km(), 1.0), 1.0)
            try:
                hop_count = nx.shortest_path_length(
                    graph,
                    neighbor,
                    self.destination_node,
                )
                destination_reachable = True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                hop_count = self.config.max_hops
                destination_reachable = False
            observation[index] = np.array(
                [
                    local_energy,
                    float(
                        self.cluster_aware
                        and neighbor in self.satellite_nodes
                        and self.cluster_assignments[self.current_graph_index].get(neighbor)
                        == self.cluster_assignments[self.current_graph_index].get(self.current_node)
                    ),
                    neighbor_energy,
                    link_quality,
                    min(remaining_contact / max(self._time_step_seconds(), 1.0), 1.0),
                    distance_norm,
                    min(hop_count / self.config.max_hops, 1.0),
                    float(neighbor in self.ground_station_nodes),
                    float(destination_reachable),
                ],
                dtype=np.float32,
            )

        self._action_mask[-1] = 1
        return observation

    def _empty_observation(self):
        return np.zeros((self.config.max_neighbors, 9), dtype=np.float32)

    @property
    def energies(self) -> dict[str, float]:
        return {
            node: self.initial_energy_j[node]
            - self.communication_energy_j[node]
            - self.compute_energy_j[node]
            for node in self.satellite_nodes
        }

    def charge_compute_energy(self, satellite: str, energy_j: float) -> None:
        if satellite not in self.initial_energy_j:
            raise ValueError(f"Unknown satellite {satellite!r}")
        if not isfinite(energy_j) or energy_j < 0.0:
            raise ValueError("Compute energy must be finite and non-negative")
        if energy_j > self.energies[satellite]:
            raise ValueError(f"Satellite {satellite!r} has insufficient energy for computation")
        self.compute_energy_j[satellite] += energy_j

    def _hop_delay(self, distance_km: float) -> float:
        propagation_s = distance_km / 299_792.458
        transmission_s = self.config.packet_bits / self.config.transmission_rate_bps
        return propagation_s + transmission_s

    def _link_quality(self, edge: dict) -> float:
        distance_km = float(edge.get("range_km", 1.0))
        if "link_quality" in edge:
            return float(np.clip(edge["link_quality"], 0.0, 1.0))
        return float(np.clip(1.0 - (distance_km / max(self._max_range_km(), 1.0)), 0.0, 1.0))

    def _max_range_km(self) -> float:
        graphs = self.trace.graphs
        max_distance = 1.0
        for graph in graphs:
            for _, _, edge in graph.edges(data=True):
                candidate = float(edge.get("range_km", 0.0))
                if candidate > max_distance:
                    max_distance = candidate
        return max_distance

    def _time_step_seconds(self) -> float:
        if len(self.trace.times_utc) < 2:
            return 1.0
        return (self.trace.times_utc[1] - self.trace.times_utc[0]).total_seconds()

    def _info(self, *, delivered=False, loss_cause=None):
        return {
            "current_node": self.current_node,
            "destination": self.destination_node,
            "graph_index": self.current_graph_index,
            "hop_count": self.hop_count,
            "delivered": delivered,
            "loss_cause": loss_cause,
            "action_mask": self._action_mask.copy(),
            "initial_energy_j": dict(self.initial_energy_j),
            "communication_energy_j": dict(self.communication_energy_j),
            "compute_energy_j": dict(self.compute_energy_j),
            "residual_energy_j": dict(self.energies),
        }
