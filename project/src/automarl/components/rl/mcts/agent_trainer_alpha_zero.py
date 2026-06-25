from __future__ import annotations

import numpy as np
import torch

from automarl.component import ParameterSignature, requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.mcts.alpha_zero_learner import AlphaZeroLearner
from automarl.components.rl.mcts.information_state import InformationState
from automarl.components.rl.mcts.mcts_planner import ImperfectInformationSearchPlanner


class AgentTrainerAlphaZero(AgentTrainer):
    """
    Per-agent trainer that uses MCTS to act and stores AlphaZero samples.

    Samples are pushed at episode end because the value target is the final
    outcome, not the immediate reward.
    """

    parameters_signature = {
        "learner": ComponentParameterSignature(default_component_definition=(AlphaZeroLearner, {})),
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

    def initialize_memory(self):
        super().initialize_memory()

        observation_shape = self.agent.processed_state_shape["observation"]
        policy_size = self._policy_vector_size()

        self.memory_fields_shapes = [
            *self.memory_fields_shapes,
            ("observation", observation_shape),
            ("mcts_policy", policy_size),
            ("value_target", 1),
        ]

        if "action_mask" in self.agent.processed_state_shape:
            self.memory_fields_shapes.append(("action_mask", self.agent.processed_state_shape["action_mask"]))

        self.memory.pass_input(
            {
                "device": self.device,
                "transition_data": self.memory_fields_shapes,
            }
        )

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
        raise NotImplementedError(f"Cannot infer AlphaZero policy target size from {action_shape}")

    @requires_input_process
    def setup_episode(self, env):
        super().setup_episode(env)
        self._episode_samples = []

    def _clone_tensor_or_value(self, value):
        if torch.is_tensor(value):
            return value.detach().clone()
        return value

    def _current_action_mask(self):
        if "action_mask" not in self.agent.state_memory:
            return None
        return self._clone_tensor_or_value(self.agent.state_memory["action_mask"])

    def _store_search_sample(self, search_result):
        sample = {
            "observation": self.agent.state_memory["observation"].detach().clone(),
            "mcts_policy": torch.as_tensor(search_result.search_policy, dtype=torch.float32, device=self.device),
        }

        action_mask = self._current_action_mask()
        if action_mask is not None:
            sample["action_mask"] = action_mask

        self._episode_samples.append(sample)

    def _episode_value_target(self, env):
        for method_name in ("terminal_value", "result_value", "get_result_value"):
            method = getattr(env, method_name, None)
            if method is not None:
                for args in ((self.agent.name,), (None, self.agent.name), (self.agent.name, None)):
                    try:
                        value = method(*args)
                        if value is not None:
                            return float(value)
                    except TypeError:
                        continue

        rewards = getattr(env, "rewards", None)
        if rewards is not None:
            try:
                reward_dict = rewards()
                if isinstance(reward_dict, dict) and self.agent.name in reward_dict:
                    return float(reward_dict[self.agent.name])
            except Exception:
                pass

        return float(self.values["episode_score"])

    def _push_episode_samples(self, final_value):
        for sample in self._episode_samples:
            to_push = {
                "observation": sample["observation"],
                "mcts_policy": sample["mcts_policy"],
                "value_target": torch.tensor([final_value], dtype=torch.float32, device=self.device),
            }
            if "action_mask" in sample:
                to_push["action_mask"] = sample["action_mask"]
            self.push_to_memory(to_push)

    @requires_input_process
    def do_training_step(self, i_episode, env):
        observation, reward, done, truncated, info = env.last()
        self.observe_new_state(env, observation)

        self.values["episode_score"] += reward

        if done or truncated:
            env.step(None)
            return reward, done, truncated

        information_state = InformationState.from_env(env, self.agent.name)

        with torch.no_grad():
            search_result = self.search_planner.search(
                env=env,
                agent_name=self.agent.name,
                information_state=information_state,
                evaluator=self.learner,
                opponent_policy=self.opponent_policy,
                training=self.train_search and self.values["is_training"],
            )

        if self.values["is_saving_in_memory"]:
            self._store_search_sample(search_result)

        action = search_result.action
        env.step(action)

        self.values["episode_steps"] += 1
        self.values["total_steps"] += 1

        if self.values["is_training"]:
            self.values["steps_done_in_session"] += 1
            if self._check_if_to_end_training():
                self.end_training()

        self._learn_if_needed()
        return reward, done, truncated

    @requires_input_process
    def end_episode(self, i_episode=None, env=None):
        final_value = self._episode_value_target(env) if env is not None else float(self.values["episode_score"])

        if self.values["is_saving_in_memory"]:
            self._push_episode_samples(final_value)

        self._episode_samples = []
        self.values["episodes_done"] += 1
        self.calculate_and_log_results()

        if self.values["is_training"]:
            for acessory in self.agent_trainer_acessories:
                acessory.as_fun()

        self._learn_if_needed()
