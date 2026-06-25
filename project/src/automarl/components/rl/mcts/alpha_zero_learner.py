from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from automarl.component import ParameterSignature, requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.ml.memory.memory_utils import interpret_values
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.rl.policy.stochastic_policy import StochasticPolicy


class AlphaZeroLearner(LearnerSchema, ComponentWithLogging):
    """
    Learner for AlphaZero-style self-play samples.

    Expected batch fields:
    - observation: model input observation
    - mcts_policy: visit-count policy target, shape [B, action_count]
    - value_target: final game outcome from this agent's perspective, shape [B, 1]
    - action_mask: optional legal-action mask, shape [B, action_count]
    """

    parameters_signature = {
        "device": ParameterSignature(ignore_at_serialization=True, get_from_parent=True),
        "critic_model": ComponentParameterSignature(
            default_component_definition=(
                FullyConnectedModelSchema,
                {"hidden_layers": 1, "hidden_size": 64, "output_shape": 1},
            )
        ),
        "critic_model_input": ParameterSignature(mandatory=False, ignore_at_serialization=True),
        "optimizer": ComponentParameterSignature(default_component_definition=(AdamOptimizer, {})),
        "policy_loss_coef": ParameterSignature(default_value=1.0),
        "value_loss_coef": ParameterSignature(default_value=1.0),
        "entropy_coef": ParameterSignature(default_value=0.0),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.device = self.get_input_value("device")
        self.policy = self.agent.get_policy()
        self.model: TorchModelComponent = self.policy.model

        self._initialize_critic_model()
        self._initialize_optimizer()

        self.policy_loss_coef = float(self.get_input_value("policy_loss_coef"))
        self.value_loss_coef = float(self.get_input_value("value_loss_coef"))
        self.entropy_coef = float(self.get_input_value("entropy_coef"))
        self.number_of_times_optimized = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_critic_model(self):
        self.critic: TorchModelComponent = self.get_input_value("critic_model")

        if not self.critic.has_custom_name_passed():
            self.critic.pass_input({"name": "alpha_zero_critic"})

        critic_model_passed_input = self.get_input_value("critic_model_input")
        if critic_model_passed_input is not None:
            self.critic.pass_input(critic_model_passed_input)

        self.critic.pass_input(
            {
                "input_shape": self.agent.processed_state_shape["observation"],
                "output_shape": 1,
                "device": self.device,
            }
        )
        self.critic.process_input_if_not_processed()

    def _initialize_optimizer(self):
        self.optimizer: OptimizerSchema = self.get_input_value("optimizer")

        if not self.optimizer.has_custom_name_passed():
            self.optimizer.pass_input({"name": "AlphaZeroOptimizer"})

        params = list(self.model.get_model_params()) + list(self.critic.get_model_params())
        self.optimizer.pass_input({"params": params})

    # ------------------------------------------------------------------
    # Search evaluator API used by mcts_planner.py
    # ------------------------------------------------------------------

    @requires_input_process
    def evaluate_search_state(self, state, action_mask=None, agent_name=None, information_state=None, env=None):
        policy_input = self._state_to_policy_input(state, action_mask, information_state)

        with torch.no_grad():
            logits = self.policy.predict_model_output(policy_input)
            logits = self._mask_logits(logits, policy_input.get("action_mask"))
            priors = torch.softmax(logits, dim=-1).detach().cpu().reshape(-1).numpy()

            value = self.critic.predict(policy_input["observation"])
            value = float(torch.tanh(value).detach().cpu().reshape(-1)[0])

        return {"priors": priors, "value": value}

    def _state_to_policy_input(self, state, action_mask=None, information_state=None):
        if isinstance(state, dict) and "observation" in state:
            obs = state["observation"]
        elif information_state is not None:
            obs = information_state.private_observation
        else:
            obs = state

        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        policy_input = {"observation": obs}

        if action_mask is not None:
            if not torch.is_tensor(action_mask):
                action_mask = torch.as_tensor(action_mask, device=self.device)
            else:
                action_mask = action_mask.to(self.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            policy_input["action_mask"] = action_mask

        return policy_input

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def interpret_trajectory(self, trajectory):
        observation = interpret_values(trajectory["observation"], self.device).detach()
        mcts_policy = interpret_values(trajectory["mcts_policy"], self.device).detach()
        value_target = interpret_values(trajectory["value_target"], self.device).detach()

        interpreted = {
            "observation": observation,
            "mcts_policy": mcts_policy,
            "value_target": value_target,
        }

        if "action_mask" in trajectory:
            interpreted["action_mask"] = interpret_values(trajectory["action_mask"], self.device).detach()

        return interpreted

    def _normalize_policy_target(self, target, action_mask=None):
        target = target.float()

        if action_mask is not None:
            mask = action_mask.to(device=target.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            target = target.masked_fill(~mask, 0.0)

        denom = target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return target / denom

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

    def _learn(self, trajectory: dict):
        self.number_of_times_optimized += 1

        batch = self.interpret_trajectory(trajectory)
        observation = batch["observation"]
        action_mask = batch.get("action_mask")
        policy_target = self._normalize_policy_target(batch["mcts_policy"], action_mask)
        value_target = batch["value_target"].float().reshape(-1, 1)

        policy_input = {"observation": observation}
        if action_mask is not None:
            policy_input["action_mask"] = action_mask

        logits = self.policy.predict_model_output(policy_input)
        logits = self._mask_logits(logits, action_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        value_pred = torch.tanh(self.critic.predict(observation)).reshape(-1, 1)
        value_loss = F.mse_loss(value_pred, value_target)

        loss = (
            self.policy_loss_coef * policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.clear_optimizer_gradients()
        loss.backward()
        self.optimizer.optimize_with_backward_pass_done()

        return {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
        }
