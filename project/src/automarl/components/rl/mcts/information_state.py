from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InformationState:
    """
    Search-facing view of the world for one agent.

    For perfect-information games, ``public_state`` can simply be the exact
    search state.  For imperfect-information games, keep hidden/private data in
    ``private_observation`` and put any belief/determinization payload in
    ``belief_state``.
    """

    agent_name: str
    public_state: Any = None
    private_observation: Any = None
    action_mask: Any = None
    history: list[Any] = field(default_factory=list)
    belief_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env, agent_name: str, public_state: Any = None, history=None, belief_state=None):
        observation = env.observe(agent_name)

        if isinstance(observation, dict):
            private_observation = observation.get("observation", observation)
            action_mask = observation.get("action_mask")
        else:
            private_observation = observation
            action_mask = None

        if public_state is None:
            for method_name in ("get_search_state", "clone_state", "get_current_whole_state"):
                method = getattr(env, method_name, None)
                if method is not None:
                    try:
                        public_state = method()
                        break
                    except TypeError:
                        pass

        return cls(
            agent_name=agent_name,
            public_state=public_state,
            private_observation=private_observation,
            action_mask=action_mask,
            history=list(history or []),
            belief_state=belief_state,
        )

    def with_public_state(self, public_state: Any) -> "InformationState":
        return InformationState(
            agent_name=self.agent_name,
            public_state=public_state,
            private_observation=self.private_observation,
            action_mask=self.action_mask,
            history=list(self.history),
            belief_state=self.belief_state,
            metadata=dict(self.metadata),
        )

    def append_action(self, agent_name: str, action: Any) -> "InformationState":
        copied = self.with_public_state(self.public_state)
        copied.history.append((agent_name, action))
        return copied
