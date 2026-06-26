"""A tiny two-player AEC Nim environment for AutoMARL.

Actions are encoded as categorical indices:
    0 -> take 1 stone
    1 -> take 2 stones
    2 -> take 3 stones
    ...

The observation is intentionally small and dense so it can be used with the
existing ToTorchTranslator + FullyConnectedModelSchema stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium
import numpy as np
import torch

from automarl.component import ParameterSignature
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
from automarl.components.rl.environment.environment_type import EnvironmentType


@dataclass(frozen=True)
class NimSearchState:
    """Immutable state used by MCTS/search code.

    previous_player is the player that made the last move. It is needed at a
    terminal node because the player who removed the final stone wins.
    """

    stones: int
    current_player: str
    previous_player: Optional[str] = None


class NimAECEnvironment(AECEnvironmentComponent):
    """A minimal AEC environment for single-pile take-away Nim.

    Game rules:
    - Start with `initial_stones` stones.
    - Players alternate turns.
    - A move removes 1..max_take stones.
    - The player who removes the last stone receives +1. The other receives -1.

    This class also exposes small search hooks (`get_search_state`,
    `legal_actions`, `next_search_state`, `terminal_value`, ...) so the same
    toy problem can be used to sanity-check the MCTS package.
    """

    parameters_signature = {
        "initial_stones": ParameterSignature(default_value=15),
        "max_take": ParameterSignature(default_value=3),
        "start_player": ParameterSignature(default_value="player_0"),
        "invalid_action_penalty": ParameterSignature(default_value=1.0),
    }

    _AGENTS = ("player_0", "player_1")

    def _process_input_internal(self):
        super()._process_input_internal()

        self.initial_stones = int(self.get_input_value("initial_stones"))
        self.max_take = int(self.get_input_value("max_take"))
        self.start_player = self.get_input_value("start_player")
        self.invalid_action_penalty = float(self.get_input_value("invalid_action_penalty"))

        if self.initial_stones < 1:
            raise ValueError("initial_stones must be >= 1")
        
        if self.max_take < 1:
            raise ValueError("max_take must be >= 1")
        
        if self.start_player not in self._AGENTS:
            raise ValueError(f"start_player must be one of {self._AGENTS}")

        # Dense observation used by the existing FCN + ToTorchTranslator path.
        # [stones/max_stones, stones_mod_period/period, player_sign]
        self._observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._action_mask_space = gymnasium.spaces.MultiBinary(self.max_take)
        self._action_space = gymnasium.spaces.Discrete(self.max_take)

        self.total_reset()

    # ---------------------------------------------------------------------
    # AutoMARL environment contract
    # ---------------------------------------------------------------------

    def get_environment_type(self) -> EnvironmentType:
        return EnvironmentType.AEC

    def get_env_name(self):
        return "nim_aec"

    def agents(self):
        return list(self._AGENTS)

    def get_active_agents(self):
        if self.terminated or self.truncated:
            return []
        return [self.current_agent]

    def get_agent_action_space(self, agent):
        self._assert_agent(agent)
        return self._action_space

    def get_agent_state_space(self, agent):
        self._assert_agent(agent)
        return {
            "observation": self._observation_space,
            "action_mask": self._action_mask_space,
        }

    def get_whole_state_shape(self):
        return {"observation": self._observation_space}

    def get_current_whole_state(self):
        return {"observation": self._make_observation(self.current_agent)}

    def reset(self):
        self.stones = self.initial_stones
        self.current_agent = self.start_player
        self.previous_agent = None
        self.winner = None

        self.terminated = False
        self.truncated = False

        self._rewards = {agent: 0.0 for agent in self._AGENTS}
        self._terminal_queue = []

        return self.observe(self.current_agent)

    def total_reset(self):
        return self.reset()

    def close(self):
        # No external resources to free. Kept for framework compatibility.
        return None

    def observe(self, agent_name: str):
        self._assert_agent(agent_name)
        return {
            "observation": self._make_observation(agent_name),
            "action_mask": self._make_action_mask(),
        }

    def last(self):
        """Return the AEC tuple for the current agent."""
        info = {
            "stones": self.stones,
            "current_agent": self.current_agent,
            "previous_agent": self.previous_agent,
            "winner": self.winner,
        }
        return (
            self.observe(self.current_agent),
            float(self._rewards.get(self.current_agent, 0.0)),
            bool(self.terminated),
            bool(self.truncated),
            info,
        )

    def agent_iter(self):
        """Yield agents in AEC order.

        After a terminal move, both agents are yielded once with done=True so
        each agent trainer can flush its pending transition with the correct
        terminal reward.
        """
        while not self.terminated and not self.truncated:
            yield self.current_agent

        while self._terminal_queue:
            self.current_agent = self._terminal_queue.pop(0)
            yield self.current_agent

    def step(self, action):
        """Apply an action for the current agent.

        The framework calls step(None) for done agents. In that case there is
        nothing to apply.
        """
        if action is None:
            return None

        if self.terminated or self.truncated:
            return None

        action_index = self._action_to_int(action)
        actor = self.current_agent
        opponent = self._other_agent(actor)

        if not self._is_legal_action_index(action_index):
            # Invalid moves should be impossible when using MaskedCategorical,
            # but this keeps the toy env debuggable with arbitrary policies.
            self._rewards[actor] = -self.invalid_action_penalty
            self._rewards[opponent] = self.invalid_action_penalty
            self.winner = opponent
            self.terminated = True
            self._terminal_queue = [actor, opponent]
            return self.last()

        take = action_index + 1
        self.stones -= take
        self.previous_agent = actor

        self._rewards = {agent: 0.0 for agent in self._AGENTS}

        if self.stones <= 0:
            self.stones = 0
            self.winner = actor
            self.terminated = True
            self._rewards[actor] = 1.0
            self._rewards[opponent] = -1.0
            self._terminal_queue = [actor, opponent]
        else:
            self.current_agent = opponent

        return self.last()

    def rewards(self):
        return dict(self._rewards)

    def render(self):
        winner = f", winner={self.winner}" if self.winner is not None else ""
        return f"Nim(stones={self.stones}, current_agent={self.current_agent}{winner})"

    # ---------------------------------------------------------------------
    # Search/MCTS hooks
    # ---------------------------------------------------------------------

    def get_search_state(self):
        return NimSearchState(
            stones=self.stones,
            current_player=self.current_agent,
            previous_player=self.previous_agent,
        )

    def current_player(self, state: Optional[NimSearchState] = None):
        return self.current_agent if state is None else state.current_player

    def legal_actions(self, state: Optional[NimSearchState] = None, player=None):
        stones = self.stones if state is None else state.stones
        return list(range(min(self.max_take, stones)))

    def action_mask(self, state: Optional[NimSearchState] = None, player=None):
        stones = self.stones if state is None else state.stones
        return self._make_action_mask(stones=stones)

    def next_search_state(self, state: NimSearchState, action):
        action_index = self._action_to_int(action)
        take = action_index + 1
        next_stones = max(0, state.stones - take)
        next_player = self._other_agent(state.current_player)
        return NimSearchState(
            stones=next_stones,
            current_player=next_player,
            previous_player=state.current_player,
        )

    def step_search(self, state: NimSearchState, action):
        return self.next_search_state(state, action)

    def is_terminal(self, state: Optional[NimSearchState] = None):
        if state is None:
            return self.terminated or self.stones <= 0
        return state.stones <= 0

    def terminal_value(self, state: NimSearchState, root_player: str):
        if not self.is_terminal(state):
            return 0.0
        return 1.0 if state.previous_player == root_player else -1.0

    # ---------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------

    def _assert_agent(self, agent):
        if agent not in self._AGENTS:
            raise ValueError(f"Unknown Nim agent: {agent}")

    def _other_agent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"

    def _make_observation(self, agent_name: str):
        period = self.max_take + 1
        player_sign = 1.0 if agent_name == "player_0" else -1.0
        return np.array(
            [
                self.stones / float(self.initial_stones),
                (self.stones % period) / float(period),
                player_sign,
            ],
            dtype=np.float32,
        )

    def _make_action_mask(self):
        mask = np.zeros(self.max_take, dtype=np.int8)
    
        for action in range(self.max_take):
            take = action + 1
            if take <= self.stones:
                mask[action] = 1
    
        return mask
    
    def _is_legal_action_index(self, action_index: int):
        return 0 <= action_index < min(self.max_take, self.stones)

    def _action_to_int(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu()
            if action.numel() != 1:
                raise ValueError(f"Nim expects a scalar action, received tensor {action}")
            return int(action.item())

        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Nim expects a scalar action, received array {action}")
            return int(action.item())

        return int(action)
