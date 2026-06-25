from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np


@dataclass
class MCTSNode:
    """One node in a root-perspective Monte Carlo search tree."""

    state: Any
    parent: "MCTSNode | None" = None
    prior: float = 0.0
    player_to_move: str | None = None
    root_player: str | None = None
    action_from_parent: Any = None
    children: dict[Any, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        """Mean value from the root player's perspective."""
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def puct_score(self, child: "MCTSNode", c_puct: float) -> float:
        """
        PUCT score for selecting ``child`` from this node.

        ``q_value`` is stored from the root player's perspective.  Therefore,
        when the opponent is to move we invert the exploitation term so the
        opponent selects lines that are bad for the root player.
        """
        parent_visits = max(1, self.visit_count)
        exploration = c_puct * child.prior * sqrt(parent_visits) / (1 + child.visit_count)
        direction = 1.0 if self.player_to_move == self.root_player else -1.0
        return direction * child.q_value + exploration

    def select_child(self, c_puct: float) -> tuple[Any, "MCTSNode"]:
        if not self.children:
            raise RuntimeError("Cannot select a child from an unexpanded MCTS node.")

        return max(
            self.children.items(),
            key=lambda item: self.puct_score(item[1], c_puct),
        )

    def add_child(self, action: Any, state: Any, prior: float, player_to_move: str | None) -> "MCTSNode":
        child = MCTSNode(
            state=state,
            parent=self,
            prior=float(prior),
            player_to_move=player_to_move,
            root_player=self.root_player,
            action_from_parent=action,
        )
        self.children[action] = child
        return child

    def backup(self, value: float) -> None:
        """Back up a root-perspective value through this node's ancestry."""
        node = self
        value = float(value)
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def visit_counts(self, max_actions: int | None = None) -> np.ndarray:
        if max_actions is None:
            if not self.children:
                return np.zeros(0, dtype=np.float32)
            numeric_actions = [a for a in self.children if isinstance(a, (int, np.integer))]
            max_actions = int(max(numeric_actions)) + 1 if numeric_actions else len(self.children)

        counts = np.zeros(max_actions, dtype=np.float32)
        fallback_index = 0

        for action, child in self.children.items():
            if isinstance(action, (int, np.integer)) and 0 <= int(action) < max_actions:
                counts[int(action)] = child.visit_count
            elif fallback_index < max_actions:
                counts[fallback_index] = child.visit_count
                fallback_index += 1

        return counts

    def visit_distribution(self, max_actions: int | None = None, temperature: float = 1.0) -> np.ndarray:
        counts = self.visit_counts(max_actions=max_actions)

        if counts.size == 0:
            return counts

        if temperature <= 0:
            policy = np.zeros_like(counts, dtype=np.float32)
            policy[int(np.argmax(counts))] = 1.0
            return policy

        counts = np.power(counts, 1.0 / temperature)
        total = float(counts.sum())

        if total <= 0:
            legal = counts > 0
            if legal.any():
                counts[legal] = 1.0
                total = float(counts.sum())
            else:
                return np.ones_like(counts, dtype=np.float32) / max(1, counts.size)

        return (counts / total).astype(np.float32)

    def best_action(self) -> Any:
        if not self.children:
            raise RuntimeError("Cannot choose an action from an unexpanded MCTS node.")
        return max(self.children.items(), key=lambda item: item[1].visit_count)[0]

    def sample_action(self, temperature: float = 1.0, rng=None) -> Any:
        if not self.children:
            raise RuntimeError("Cannot sample an action from an unexpanded MCTS node.")

        if temperature <= 0:
            return self.best_action()

        rng = np.random.default_rng() if rng is None else rng
        actions = list(self.children.keys())
        counts = np.asarray([self.children[a].visit_count for a in actions], dtype=np.float64)
        counts = np.power(counts, 1.0 / temperature)

        if counts.sum() <= 0:
            probs = np.ones(len(actions), dtype=np.float64) / len(actions)
        else:
            probs = counts / counts.sum()

        return actions[int(rng.choice(len(actions), p=probs))]
