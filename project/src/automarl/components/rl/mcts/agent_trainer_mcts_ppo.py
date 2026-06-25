from __future__ import annotations

import numpy as np
import torch

from automarl.component import ParameterSignature
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.ml.memory.memory_utils import interpret_values
from automarl.components.rl.learners.ppo_learner import PPOLearner
from automarl.components.rl.mcts.information_state import InformationState
from automarl.components.rl.mcts.mcts_planner import ImperfectInformationSearchPlanner
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO


class MCTSPPOLearner(PPOLearner):
    """PPO learner with an auxiliary loss toward the MCTS visit policy."""

    parameters_signature = {
        "mcts_policy_coef": ParameterSignature(default_value=0.2),
    }

    def _process_input_internal(self):
        super()._process_input_internal()
        self.mcts_policy_coef = float(self.get_input_value("mcts_policy_coef"))

    def interpret_trajectory(self, trajectory):
        interpreted = super().interpret_trajectory(trajectory)
        interpreted["mcts_policy"] = interpret_values(trajectory["mcts_policy"], self.device).detach()
        return interpreted

    def _normalize_policy_target(self, target, action_mask=None):
        target = target.float()
        if action_mask is not None:
            mask = action_mask.to(device=target.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            target = target.masked_fill(~mask, 0.0)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _mask_logits(self, logits, action_mask=None):
        if action_mask is None:
            return logits
        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, device=logits.device)
        else:
            action_mask = action_mask.to(device=logits.device)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask > 0
        while action_mask.dim() < logits.dim():
            action_mask = action_mask.unsqueeze(0)
        return logits.masked_fill(~action_mask, -1e9)

    def _mcts_policy_loss(self, interpreted_trajectory):
        model_output = interpreted_trajectory["model_output"]
        action_mask = interpreted_trajectory.get("action_mask")
        target = self._normalize_policy_target(interpreted_trajectory["mcts_policy"], action_mask)
        logits = self._mask_logits(model_output, action_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    def _compute_losses(self, interpreted_trajectory):
        ratio, policy_loss, value_loss, loss = super()._compute_losses(interpreted_trajectory)
        mcts_loss = self._mcts_policy_loss(interpreted_trajectory)
        policy_loss = policy_loss + self.mcts_policy_coef * mcts_loss

        if self.critic_optimizer is None:
            loss = policy_loss + value_loss

        interpreted_trajectory["mcts_policy_loss"] = mcts_loss
        return ratio, policy_loss, value_loss, loss

    def _learn(self, trajectory: dict):
        values = super()._learn(trajectory)
        return values


class AgentTrainerMCTSPPO(AgentTrainerPPO):
    """
    PPO trainer where action selection is performed by MCTS.

    The PPO behavior log-prob is the log-probability under the MCTS visit
    distribution, not under the raw neural policy.  This keeps the PPO ratio
    honest while still training the neural policy toward the search policy.
    """

    parameters_signature = {
        "learner": ComponentParameterSignature(default_component_definition=(MCTSPPOLearner, {})),
        "search_planner": ComponentParameterSignature(
            default_component_definition=(ImperfectInformationSearchPlanner, {})
        ),
        "opponent_policy": ParameterSignature(mandatory=False, ignore_at_serialization=True),
        "train_search": ParameterSignature(default_value=True),
    }

    def _process_input_internal(self):
        super()._process_input_internal()
        self.search_planner = self.get_input_value("search_planner")
        self.search_planner.process_input_if_not_processed()
        self.opponent_policy = self.get_input_value("opponent_policy")
        self.train_search = bool(self.get_input_value("train_search"))

    def _policy_vector_size(self):
        action_shape = self.agent_policy.get_policy_output_shape()
        if hasattr(action_shape, "n"):
            return int(action_shape.n)
        if isinstance(action_shape, int):
            return int(action_shape)
        if isinstance(action_shape, torch.Size):
            return int(np.prod(tuple(action_shape)))
        if isinstance(action_shape, (tuple, list)):
            return int(np.prod(action_shape))
        raise NotImplementedError(f"Cannot infer MCTS policy target size from {action_shape}")

    def initialize_memory(self):
        super().initialize_memory()

        self.memory_fields_shapes = [
            *self.memory_fields_shapes,
            ("mcts_policy", self._policy_vector_size()),
        ]

        self.memory.pass_input({"transition_data": self.memory_fields_shapes})

    def _action_to_scalar_tensor(self, action):
        if torch.is_tensor(action):
            return action.to(self.device).reshape(-1)[0].long()
        if isinstance(action, np.ndarray):
            return torch.tensor(int(action.reshape(-1)[0]), dtype=torch.long, device=self.device)
        if isinstance(action, (tuple, list)):
            if len(action) != 1:
                raise RuntimeError(
                    "AgentTrainerMCTSPPO currently expects scalar candidate-action ids. "
                    "For multi-select games, enumerate candidates in the environment and return the candidate id."
                )
            return torch.tensor(int(action[0]), dtype=torch.long, device=self.device)
        return torch.tensor(int(action), dtype=torch.long, device=self.device)

    def _log_prob_from_search_policy(self, search_policy, action):
        action_index = int(self._action_to_scalar_tensor(action).detach().cpu().item())
        prob = float(search_policy[action_index]) if action_index < len(search_policy) else 0.0
        return torch.log(torch.tensor([max(prob, 1e-8)], dtype=torch.float32, device=self.device))

    def _run_search_with_memory(self):
        information_state = InformationState(
            agent_name=self.agent.name,
            public_state=None,
            private_observation=self.agent.state_memory["observation"],
            action_mask=self.agent.state_memory.get("action_mask"),
        )

        env = getattr(self, "_current_env_for_search", None)
        if env is None:
            raise RuntimeError(
                "AgentTrainerMCTSPPO needs the current env. Use do_training_step, "
                "or set _current_env_for_search before selecting an action."
            )

        return self.search_planner.search(
            env=env,
            agent_name=self.agent.name,
            information_state=information_state,
            evaluator=self.learner,
            opponent_policy=self.opponent_policy,
            training=self.train_search and self.values["is_training"],
        )

    def select_action_with_memory(self):
        search_result = self._run_search_with_memory()

        action_val = self._action_to_scalar_tensor(search_result.action)
        self.last_action_val = action_val
        self.last_log_prob = self._log_prob_from_search_policy(search_result.search_policy, action_val)
        self.last_mcts_policy = torch.as_tensor(
            search_result.search_policy,
            dtype=torch.float32,
            device=self.device,
        )

        return action_val

    def do_training_step(self, i_episode, env):
        self._current_env_for_search = env
        try:
            return super().do_training_step(i_episode, env)
        finally:
            self._current_env_for_search = None

    def _gen_to_push(self, prev_state, next_state, action, reward, done, truncated):
        to_push = super()._gen_to_push(prev_state, next_state, action, reward, done, truncated)
        to_push["mcts_policy"] = self.last_mcts_policy
        return to_push
