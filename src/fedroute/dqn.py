"""Minimal deep-Q training and policy utilities for centralized routing validation."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
import torch
from torch import nn


class DQNPolicy(nn.Module):
    """Score each candidate row once, with one separate value for dropping."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate_scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.drop_action_value = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        if features.ndim < 2:
            raise ValueError("Candidate features must have candidate and feature dimensions")
        candidate_values = self.candidate_scorer(features).squeeze(-1)
        drop_values = self.drop_action_value.expand(*candidate_values.shape[:-1], 1)
        return torch.cat((candidate_values, drop_values), dim=-1)


class CentralizedDQNTrainer:
    """Lightweight Q-learning trainer for a centralized validation loop."""

    def __init__(self, policy: DQNPolicy, learning_rate: float = 1e-3, gamma: float = 0.99):
        self.policy = policy
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        q_values = self.policy(observation)
        if q_values.ndim != 1 or q_values.shape[0] != action_mask.shape[0]:
            raise ValueError("Action mask must have one entry per candidate plus the drop action")
        if not np.any(action_mask):
            raise ValueError("At least one action must be valid")
        masked = q_values.clone()
        masked[torch.as_tensor(action_mask == 0, device=masked.device)] = -torch.inf
        return int(torch.argmax(masked).item())

    def compute_td_target(
        self,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> torch.Tensor:
        target = torch.tensor(reward, dtype=torch.float32)
        if done:
            return target
        with torch.no_grad():
            next_q_values = self.policy(next_state)
            if next_q_values.shape[0] != next_action_mask.shape[0]:
                raise ValueError("Next action mask must align with next-state Q-values")
            if not np.any(next_action_mask):
                raise ValueError("At least one next-state action must be valid")
            masked_next = next_q_values.clone()
            masked_next[
                torch.as_tensor(next_action_mask == 0, device=masked_next.device)
            ] = -torch.inf
            return target + self.gamma * masked_next.max()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> float:
        q_values = self.policy(state)
        if q_values.ndim != 1 or q_values.shape[0] != action_mask.shape[0]:
            raise ValueError("Current action mask must align with candidate Q-values")
        if not 0 <= action < q_values.shape[0] or action_mask[action] == 0:
            raise ValueError("The selected action must be valid in the current state")

        target = self.compute_td_target(
            reward,
            next_state,
            done,
            next_action_mask,
        ).to(device=q_values.device, dtype=q_values.dtype)

        self.optimizer.zero_grad()
        loss = nn.functional.mse_loss(q_values[action], target)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train(self, env, episodes: int = 5, max_steps_per_episode: int = 10):
        losses: list[float] = []
        for _ in range(episodes):
            observation, info = env.reset()
            for _ in range(max_steps_per_episode):
                action_mask = info["action_mask"]
                action = self.choose_action(observation, action_mask)
                next_observation, reward, terminated, truncated, info = env.step(action)
                loss = self.train_step(
                    state=observation,
                    action=action,
                    reward=reward,
                    next_state=next_observation,
                    done=terminated or truncated,
                    action_mask=action_mask,
                    next_action_mask=info["action_mask"],
                )
                losses.append(loss)
                observation = next_observation
                if terminated or truncated:
                    break
        return losses


def train_centralized_dqn(env, hidden_dim: int = 32, episodes: int = 5):
    input_dim = env.observation_space.shape[-1]
    policy = DQNPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
    trainer = CentralizedDQNTrainer(policy=policy)
    return trainer.train(env, episodes=episodes)


@dataclass(frozen=True)
class DQNTrainingConfig:
    seed: int
    episodes: int = 100
    max_steps_per_episode: int = 16
    batch_size: int = 32
    replay_capacity: int = 10_000
    warmup_transitions: int = 64
    target_update_interval: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("DQN seed must be non-negative")
        if min(
            self.episodes,
            self.max_steps_per_episode,
            self.batch_size,
            self.replay_capacity,
            self.warmup_transitions,
            self.target_update_interval,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("DQN counts and dimensions must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least the batch size")
        if self.learning_rate <= 0.0 or not 0.0 < self.gamma <= 1.0:
            raise ValueError("DQN learning rate and discount factor are invalid")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("DQN epsilon values must satisfy 0 <= end <= start <= 1")


@dataclass(frozen=True)
class DQNTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class DQNReplayBuffer:
    """Bounded replay storage for masked candidate-action transitions."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.transitions: deque[DQNTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: DQNTransition) -> None:
        self.transitions.append(
            DQNTransition(
                state=np.array(transition.state, dtype=np.float32, copy=True),
                action=int(transition.action),
                reward=float(transition.reward),
                next_state=np.array(transition.next_state, dtype=np.float32, copy=True),
                done=bool(transition.done),
                action_mask=np.array(transition.action_mask, dtype=np.int8, copy=True),
                next_action_mask=np.array(
                    transition.next_action_mask,
                    dtype=np.int8,
                    copy=True,
                ),
            )
        )

    def sample(self, batch_size: int, rng: random.Random) -> list[DQNTransition]:
        if not 0 < batch_size <= len(self.transitions):
            raise ValueError("Batch size must be within replay-buffer length")
        return rng.sample(list(self.transitions), batch_size)


class DQNTrainer:
    """Replay-buffer DQN with a target network and candidate-aligned Q-values."""

    def __init__(self, policy: DQNPolicy, config: DQNTrainingConfig):
        self.policy = policy
        self.target_policy = deepcopy(policy)
        self.target_policy.eval()
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.replay_buffer = DQNReplayBuffer(config.replay_capacity)
        self.rng = random.Random(config.seed)
        self.gradient_steps = 0

    def _copy_target_network(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())

    @staticmethod
    def _masked_q_values(policy: DQNPolicy, state: np.ndarray, action_mask: np.ndarray) -> torch.Tensor:
        q_values = policy(state)
        if q_values.ndim != 1 or q_values.shape[0] != action_mask.shape[0]:
            raise ValueError("Action mask must have one entry per candidate plus drop")
        if not np.any(action_mask):
            raise ValueError("At least one action must be valid")
        masked = q_values.clone()
        masked[torch.as_tensor(action_mask == 0, device=masked.device)] = -torch.inf
        return masked

    def select_greedy_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        self.policy.eval()
        with torch.no_grad():
            masked = self._masked_q_values(self.policy, state, action_mask)
        return int(torch.argmax(masked).item())

    def _select_training_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            raise ValueError("At least one action must be valid")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions.tolist()))
        return self.select_greedy_action(state, action_mask)

    def _train_batch(self, batch: list[DQNTransition]) -> float:
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
        )
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32)
        action_masks = torch.as_tensor(
            np.stack([item.action_mask for item in batch]),
            dtype=torch.bool,
        )
        next_action_masks = torch.as_tensor(
            np.stack([item.next_action_mask for item in batch]),
            dtype=torch.bool,
        )
        q_values = self.policy(states)
        if q_values.shape != action_masks.shape or not torch.all(action_masks.any(dim=1)):
            raise ValueError("Replay action masks must match Q-values and include a valid action")
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_policy(next_states)
            if next_q_values.shape != next_action_masks.shape:
                raise ValueError("Next-state replay masks must match target Q-values")
            if not torch.all(next_action_masks.any(dim=1)):
                raise ValueError("Every nonterminal next state requires a valid action")
            next_q_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
            next_values = next_q_values.max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_values

        loss = nn.functional.mse_loss(selected_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self._copy_target_network()
        return float(loss.item())

    def train(
        self,
        env,
        source: str,
        destination: str,
        *,
        training_start_indices: tuple[int, ...] | None = None,
    ) -> list[dict[str, float | int | bool | None]]:
        torch.manual_seed(self.config.seed)
        self.policy.train()
        history: list[dict[str, float | int | bool | None]] = []
        if training_start_indices is None:
            training_start_indices = tuple(range(len(env.trace.graphs)))
        if not training_start_indices:
            raise ValueError("At least one training start index is required")

        for episode in range(self.config.episodes):
            fraction = episode / max(1, self.config.episodes - 1)
            epsilon = self.config.epsilon_start + fraction * (
                self.config.epsilon_end - self.config.epsilon_start
            )
            start_index = self.rng.choice(training_start_indices)
            state, info = env.reset(
                seed=self.config.seed + episode,
                options={
                    "source": source,
                    "destination": destination,
                    "start_index": start_index,
                },
            )
            episode_return = 0.0
            episode_losses: list[float] = []
            transitions = 0
            for _ in range(self.config.max_steps_per_episode):
                action_mask = info["action_mask"]
                action = self._select_training_action(state, action_mask, epsilon)
                next_state, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                transition = DQNTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    action_mask=action_mask,
                    next_action_mask=next_info["action_mask"],
                )
                self.replay_buffer.add(transition)
                episode_return += reward
                transitions += 1
                if len(self.replay_buffer) >= max(
                    self.config.batch_size,
                    self.config.warmup_transitions,
                ):
                    episode_losses.append(
                        self._train_batch(
                            self.replay_buffer.sample(self.config.batch_size, self.rng)
                        )
                    )
                state, info = next_state, next_info
                if done:
                    break
            history.append(
                {
                    "episode": episode,
                    "epsilon": epsilon,
                    "return": episode_return,
                    "mean_loss": (
                        sum(episode_losses) / len(episode_losses) if episode_losses else None
                    ),
                    "transitions": transitions,
                    "delivered": bool(info.get("delivered", False)),
                    "start_index": start_index,
                }
            )
        self._copy_target_network()
        self.policy.eval()
        return history
