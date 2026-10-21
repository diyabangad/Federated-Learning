"""Federated averaging for local DQN routing prototypes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import torch
from torch import Tensor

from fedroute.dqn import CentralizedDQNTrainer, DQNPolicy


def aggregate_state_dicts(
    client_states: Sequence[Mapping[str, Tensor]],
    sample_counts: Sequence[int],
) -> dict[str, Tensor]:
    """Aggregate client model states using sample-count-weighted FedAvg."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Client states and sample counts must be non-empty and aligned")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Every client sample count must be positive")

    reference = client_states[0]
    reference_keys = set(reference)
    for state in client_states[1:]:
        if set(state) != reference_keys:
            raise ValueError("Client model states must contain identical keys")
        for key in reference_keys:
            if state[key].shape != reference[key].shape or state[key].dtype != reference[key].dtype:
                raise ValueError(f"Client model state mismatch for {key!r}")

    total_samples = sum(sample_counts)
    aggregated: dict[str, Tensor] = {}
    for key, reference_value in reference.items():
        if not (torch.is_floating_point(reference_value) or torch.is_complex(reference_value)):
            aggregated[key] = reference_value.detach().clone()
            continue
        average = torch.zeros_like(reference_value)
        for state, count in zip(client_states, sample_counts):
            average.add_(state[key].to(device=average.device), alpha=count / total_samples)
        aggregated[key] = average
    return aggregated


@dataclass(frozen=True)
class FederatedEnergyConfig:
    training_j_per_transition: float
    communication_j_per_byte: float

    def __post_init__(self) -> None:
        if not isfinite(self.training_j_per_transition) or self.training_j_per_transition < 0:
            raise ValueError("training_j_per_transition must be finite and non-negative")
        if not isfinite(self.communication_j_per_byte) or self.communication_j_per_byte < 0:
            raise ValueError("communication_j_per_byte must be finite and non-negative")


@dataclass(frozen=True)
class FederatedRoundResult:
    client_losses: tuple[tuple[float, ...], ...]
    total_transitions: int
    model_payload_bytes: int
    communication_bytes: int
    client_energy_costs_j: tuple[tuple[str, float], ...]
    remaining_energy_j: tuple[tuple[str, float], ...]


class FederatedDQNTrainer:
    """Train local DQN copies and aggregate them into the shared policy."""

    def __init__(
        self,
        policy: DQNPolicy,
        energy_config: FederatedEnergyConfig,
        client_energy_budgets_j: Mapping[str, float],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ):
        if not client_energy_budgets_j:
            raise ValueError("Client energy budgets are required")
        if any(not isfinite(value) or value < 0 for value in client_energy_budgets_j.values()):
            raise ValueError("Client energy budgets must be finite and non-negative")
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.energy_config = energy_config
        self.remaining_energy_j = dict(client_energy_budgets_j)

    def run_round(
        self,
        client_envs: Sequence,
        episodes_per_client: int = 1,
        max_steps_per_episode: int = 10,
        client_ids: Sequence[str] | None = None,
    ) -> FederatedRoundResult:
        if not client_envs:
            raise ValueError("At least one client environment is required")
        if episodes_per_client <= 0 or max_steps_per_episode <= 0:
            raise ValueError("Local episode and step counts must be positive")
        if client_ids is None:
            client_ids = tuple(f"client:{index}" for index in range(len(client_envs)))
        if len(client_ids) != len(client_envs) or len(set(client_ids)) != len(client_ids):
            raise ValueError("Client identifiers must be unique and aligned with environments")
        unknown_clients = set(client_ids) - self.remaining_energy_j.keys()
        if unknown_clients:
            raise ValueError(f"Missing energy budgets for clients: {sorted(unknown_clients)}")

        client_states: list[Mapping[str, Tensor]] = []
        sample_counts: list[int] = []
        participating_client_ids: list[str] = []
        all_client_losses: list[tuple[float, ...]] = []
        for client_id, env in zip(client_ids, client_envs):
            local_policy = deepcopy(self.policy)
            local_trainer = CentralizedDQNTrainer(
                local_policy,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )
            losses = tuple(
                local_trainer.train(
                    env,
                    episodes=episodes_per_client,
                    max_steps_per_episode=max_steps_per_episode,
                )
            )
            all_client_losses.append(losses)
            if losses:
                client_states.append(local_policy.state_dict())
                sample_counts.append(len(losses))
                participating_client_ids.append(client_id)

        if not sample_counts:
            raise ValueError("No client produced a training transition")
        model_payload_bytes = sum(
            value.numel() * value.element_size()
            for value in self.policy.state_dict().values()
        )
        client_energy_costs = {}
        for client_id, transitions in zip(participating_client_ids, sample_counts):
            communication_bytes = 2 * model_payload_bytes
            cost_j = (
                transitions * self.energy_config.training_j_per_transition
                + communication_bytes * self.energy_config.communication_j_per_byte
            )
            if cost_j > self.remaining_energy_j[client_id]:
                raise ValueError(
                    f"Client {client_id!r} has insufficient energy for this round: "
                    f"requires {cost_j:.6g} J, has {self.remaining_energy_j[client_id]:.6g} J"
                )
            client_energy_costs[client_id] = cost_j

        self.policy.load_state_dict(aggregate_state_dicts(client_states, sample_counts))
        for client_id, cost_j in client_energy_costs.items():
            self.remaining_energy_j[client_id] -= cost_j
        return FederatedRoundResult(
            client_losses=tuple(all_client_losses),
            total_transitions=sum(sample_counts),
            model_payload_bytes=model_payload_bytes,
            communication_bytes=2 * model_payload_bytes * len(sample_counts),
            client_energy_costs_j=tuple(client_energy_costs.items()),
            remaining_energy_j=tuple(
                (client_id, self.remaining_energy_j[client_id])
                for client_id in client_energy_costs
            ),
        )