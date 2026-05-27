from automarl.components.basic_components.dynamic_value import get_value_or_dynamic_value
from automarl.component import ParameterSignature, requires_input_process
from automarlcore.advanced_input_management import ComponentParameterSignature
from automarlcore.advanced_input_utils import get_value_of_type_or_component
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.ml.memory.memory_utils import interpret_values, interpret_values
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automarl.components.rl.learners.learner_component import LearnerSchema, NoAgentLearner
from automarl.components.rl.policy.stochastic_policy import StochasticPolicy
from automarl.components.rl.rl_agorithms_utils import ppo

from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.models.torch_model_components import TorchModelComponent
import torch

from automarl.utils.class_util import get_class_from


class PPOLearnerNoCritic(LearnerSchema, ComponentWithLogging):
    '''
    PPO learner without critic optimization.

    This learner assumes critic-related quantities are already precomputed and
    stored in the trajectory, namely the advantages. It only optimizes the policy
    using PPO clipped objective.

    Expected trajectory keys:
        - observation
        - action_val
        - log_prob
        - advantages

    Optional extra state keys:
        - any key present in policy.input_state_shape besides "observation"
    '''

    parameters_signature = {
        "device": ParameterSignature(ignore_at_serialization=True, get_from_parent=True),

        "optimizer": ComponentParameterSignature(
            default_component_definition=(
                AdamOptimizer,
                {}
            )
        ),

        "clip_epsilon": ParameterSignature(
            default_value=0.2,
            description="The clip range",
            custom_dict={"hyperparameter_suggestion": ["float", {"low": 0.1, "high": 0.3}]}
        ),

        "entropy_coef": ParameterSignature(
            default_value=0.01,
            description="How much weight entropy has",
            custom_dict={"hyperparameter_suggestion": ["float", {"low": 0.0, "high": 0.3}]}
        ),
        "lambda_gae" : ParameterSignature(default_value=0.95, description="Controls trade-off between bias and variance, higher means more variance and less bias",
                                     custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.9, "high": 0.999 }]}),

        "discount_factor" : ParameterSignature(get_from_parent=True)
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.device = self.get_input_value("device")

        self.policy: StochasticPolicy = self.agent.get_policy()

        if isinstance(self.policy, StochasticPolicy) is False:
            raise Exception(
                "PPOLearnerNoCritic requires a Stochastic Policy, but got {}".format(
                    get_class_from(self.policy)
                )
            )

        self.custom_data_beyond_obs = [
            key for key in self.policy.input_state_shape.keys()
            if key != "observation"
        ]

        if len(self.custom_data_beyond_obs) > 0:
            self.lg.writeLine(
                f"Noticed that policy also receives in its shape: {self.custom_data_beyond_obs}"
            )

        self.model = self.policy.model

        self.initialize_optimizer()

        self.clip_epsilon = self.get_input_value("clip_epsilon")
        self.entropy_coef = get_value_of_type_or_component(self, "entropy_coef", float)
        self.lambda_gae = self.get_input_value("lambda_gae")
        self.discount_factor = self.get_input_value("discount_factor")
        

        self.number_of_times_optimized = 0

    def initialize_optimizer(self):
        self.actor_optimizer: OptimizerSchema = self.get_input_value("optimizer")
        self.actor_optimizer.pass_input({"model": self.model})

        if not self.actor_optimizer.has_custom_name_passed():
            self.actor_optimizer.pass_input({"name": "ActorOptimizer"})

    def interpret_trajectory(self, trajectory):
        interpreted_trajectory = super().interpret_trajectory(trajectory)

        interpreted_trajectory["truncation"] = interpret_values(
            trajectory["truncation"], self.device
        ).detach()


        interpreted_trajectory["log_prob"] = interpret_values(
            trajectory["log_prob"], self.device
        ).detach()

        interpreted_trajectory["action_val"] = interpret_values(
            trajectory["action_val"], self.device
        ).detach()

        for key in self.custom_data_beyond_obs:
            interpreted_trajectory[key] = interpret_values(
                trajectory[key], self.device
            ).detach()

        return interpreted_trajectory

    def _evaluate_actions(self, interpreted_trajectory):
        """
        Evaluate given actions under the current policy.
        Computes log probabilities of actions and entropy of the policy distribution.
        """

        observation_batch = interpreted_trajectory["observation"]
        action_vals_batch = interpreted_trajectory["action_val"]

        state_batch = {"observation": observation_batch}

        for custom_data_key in self.custom_data_beyond_obs:
            state_batch[custom_data_key] = interpreted_trajectory[custom_data_key]

        model_output, new_log_probs, entropy = ppo.evaluate_actions(
            self.policy,
            state_batch,
            action_vals_batch
        )

        interpreted_trajectory["model_output"] = model_output
        interpreted_trajectory["new_log_probs"] = new_log_probs
        interpreted_trajectory["entropy"] = entropy

    def _compute_policy_loss(self, new_log_probs, log_prob_batch, advantages, entropy):
        return ppo.compute_policy_loss(
            new_log_probs,
            log_prob_batch,
            advantages,
            entropy,
            self.clip_epsilon,
            self.entropy_coef
        )
    
    @requires_input_process
    def compute_error_and_advantage(self, interpreted_trajectory, observation_critic_values = None, next_obs_critic_values = None):

        reward_batch = interpreted_trajectory["reward"]
        next_obs_critic_values = interpreted_trajectory["next_obs_critic_values"] if next_obs_critic_values is None else next_obs_critic_values
        observation_critic_values = interpreted_trajectory["observation_critic_values"] if observation_critic_values is None else observation_critic_values
        done_batch = interpreted_trajectory["done"]
        truncated_batch = interpreted_trajectory["truncation"]

        return ppo.compute_gae_and_returns(reward_batch, observation_critic_values, next_obs_critic_values, done_batch, truncated_batch, self.discount_factor, self.lambda_gae)
    


    def _compute_losses(self, interpreted_trajectory):
        new_log_probs = interpreted_trajectory["new_log_probs"]
        entropy = interpreted_trajectory["entropy"]
        log_prob_batch = interpreted_trajectory["log_prob"]
        advantages = interpreted_trajectory["advantages"]

        ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss = \
            self._compute_policy_loss(
                new_log_probs,
                log_prob_batch,
                advantages,
                entropy
            )

        return ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss

    def _optimize_using_loss(self, policy_loss):
        self.actor_optimizer.clear_optimizer_gradients()
        policy_loss.backward()
        self.actor_optimizer.optimize_with_backward_pass_done()

    def _learn(self, trajectory: dict):
        super()._learn(trajectory)

        self.number_of_times_optimized += 1

        interpreted_trajectory = self.interpret_trajectory(trajectory)

        # NOTE: PPO REQUIRES ADVANTAGES TO BE PRE-COMPUTED.
        # This learner assumes they are already present in trajectory["advantages"].

        self._evaluate_actions(interpreted_trajectory)

        ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss = \
            self._compute_losses(interpreted_trajectory)

        self._optimize_using_loss(policy_loss)

        return {
            "log_prob": interpreted_trajectory["log_prob"],
            "new_log_probs": interpreted_trajectory["new_log_probs"],
            "ratio": ratio.detach(),
            "policy_loss": policy_loss.detach(),
            "mean_policy_loss": mean_policy_loss.detach(),
        }
    


class PPOLearnerOnlyCritic(NoAgentLearner, ComponentWithLogging):
    '''
    PPO learner that only optimizes the critic.

    This learner assumes critic-side targets are already precomputed and stored
    in the trajectory.

    Expected trajectory keys:
        - observation
        - next_observation
        - done
        - returns
        - observation_old_critic_values
    '''

    parameters_signature = {
        "device": ParameterSignature(ignore_at_serialization=True, get_from_parent=True),

        "critic_model": ComponentParameterSignature(
            default_component_definition=(
                FullyConnectedModelSchema,
                {"hidden_layers": 1, "hidden_size": 64, "output_shape": 1}
            )
        ),

        "critic_model_input": ParameterSignature(
            mandatory=False,
            ignore_at_serialization=True
        ),

        "optimizer": ComponentParameterSignature(
            default_component_definition=(
                AdamOptimizer,
                {}
            )
        ),

        "clip_epsilon": ParameterSignature(
            default_value=0.2,
            description="The clip range for value clipping",
            custom_dict={"hyperparameter_suggestion": ["float", {"low": 0.1, "high": 0.3}]}
        ),

        "value_loss_coef": ParameterSignature(
            default_value=0.5,
            description="The weight given to the critic value loss",
            custom_dict={"hyperparameter_suggestion": ["float", {"low": 0.3, "high": 0.7}]},
        ),

        "lambda_gae": ParameterSignature(
            default_value=0.95,
            description="Controls trade-off between bias and variance, higher means more variance and less bias",
            custom_dict={"hyperparameter_suggestion": ["float", {"low": 0.9, "high": 0.999}]}
        ),

        "discount_factor": ParameterSignature(get_from_parent=True),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.device = self.get_input_value("device")

        self.initialize_critic_model()
        self.initialize_optimizer()

        self.clip_epsilon = self.get_input_value("clip_epsilon")
        self.value_loss_coef = self.get_input_value("value_loss_coef")
        self.lambda_gae = self.get_input_value("lambda_gae")
        self.discount_factor = self.get_input_value("discount_factor")

        self.number_of_times_optimized = 0

    @requires_input_process
    def critic_pred(self, observation):
        return self.critic.predict(observation)

    def initialize_critic_model(self):
        self.critic: TorchModelComponent = self.get_input_value("critic_model")

        if not self.critic.has_custom_name_passed():
            self.critic.pass_input({"name": "critic"})

        critic_model_passed_input = self.get_input_value("critic_model_input")
        if critic_model_passed_input is not None:
            self.critic.pass_input(critic_model_passed_input)

        critic_output_shape = self.critic.get_input_value("output_shape")
        if critic_output_shape is None:
            self.critic.pass_input({"output_shape": 1})
        else:
            self.lg.writeLine(f"Critic model already has output shape defined: {critic_output_shape}")

        self.critic.process_input_if_not_processed()

    def initialize_optimizer(self):
        self.critic_optimizer: OptimizerSchema = self.get_input_value("optimizer")
        self.critic_optimizer.pass_input({"model": self.critic})

        if not self.critic_optimizer.has_custom_name_passed():
            self.critic_optimizer.pass_input({"name": "CriticOptimizer"})

    def interpret_trajectory(self, trajectory):
        interpreted_trajectory = {**trajectory}

        interpreted_trajectory["observation"] = interpret_values(
            trajectory["observation"], self.device
        ).detach()

        interpreted_trajectory["next_observation"] = interpret_values(
            trajectory["next_observation"], self.device
        ).detach()

        interpreted_trajectory["reward"] = interpret_values(
            trajectory["reward"], self.device
        ).detach()

        interpreted_trajectory["done"] = interpret_values(
            trajectory["done"], self.device
        ).detach()

        interpreted_trajectory["truncation"] = interpret_values(
            trajectory["truncation"], self.device
        ).detach()

        if "alive_agents" in trajectory:
            interpreted_trajectory["alive_agents"] = interpret_values(
                trajectory["alive_agents"], self.device
            ).detach()

        if "observation_old_critic_value" in trajectory:
            interpreted_trajectory["observation_old_critic_value"] = interpret_values(
                trajectory["observation_old_critic_value"], self.device
            ).detach()

        if "next_obs_old_critic_value" in trajectory:
            interpreted_trajectory["next_obs_old_critic_value"] = interpret_values(
                trajectory["next_obs_old_critic_value"], self.device
            ).detach()

        if "returns" in trajectory:
            interpreted_trajectory["returns"] = interpret_values(
                trajectory["returns"], self.device
            ).detach()

        if "advantages" in trajectory:
            interpreted_trajectory["advantages"] = interpret_values(
                trajectory["advantages"], self.device
            ).detach()

        if "critic_obs_pred_error" in trajectory:
            interpreted_trajectory["critic_obs_pred_error"] = interpret_values(
                trajectory["critic_obs_pred_error"], self.device
            ).detach()

        if "non_normalized_advantages" in trajectory:
            interpreted_trajectory["non_normalized_advantages"] = interpret_values(
                trajectory["non_normalized_advantages"], self.device
            ).detach()

        return interpreted_trajectory

    def _masked_mean(self, tensor, mask, eps=1e-8):
        masked_tensor = tensor * mask
        denom = mask.sum().clamp_min(eps)
        return masked_tensor.sum() / denom

    def compute_values_estimates(self, interpreted_trajectory):
        '''
        Computes value estimates using the critic.
        Note that this may differ from the critic values computed before learning
        starts, as learning may be called multiple times.
        '''

        observation_batch = interpreted_trajectory["observation"]
        next_observation_batch = interpreted_trajectory["next_observation"]
        done_batch = interpreted_trajectory["done"]

        observation_critic_values, next_obs_critic_values = ppo.compute_values_estimates(
            self.critic,
            observation_batch,
            next_observation_batch,
            done_batch
        )

        if "alive_agents" in interpreted_trajectory:
            alive_agents = interpreted_trajectory["alive_agents"]

            observation_critic_values = observation_critic_values * alive_agents
            next_obs_critic_values = next_obs_critic_values * alive_agents

        return observation_critic_values, next_obs_critic_values

    def _compute_critic_loss(self, interpreted_trajectory):
        '''
        Computes the clipped critic loss.
        '''

        observation_critic_values = interpreted_trajectory["observation_critic_values"]
        returns = interpreted_trajectory["returns"]
        obs_old_critic_values = interpreted_trajectory["observation_old_critic_value"]

        value_loss_unclipped, value_loss_clipped, value_loss_batch, _, _ = ppo.compute_critic_loss(
            observation_critic_values,
            returns,
            obs_old_critic_values,
            self.clip_epsilon,
            self.value_loss_coef
        )

        if "alive_agents" in interpreted_trajectory:
            alive_agents = interpreted_trajectory["alive_agents"]

            value_loss_mean = self._masked_mean(value_loss_batch, alive_agents)
            value_loss = value_loss_mean * self.value_loss_coef

            return (
                value_loss_unclipped,
                value_loss_clipped,
                value_loss_batch,
                value_loss_mean,
                value_loss
            )

        value_loss_mean = value_loss_batch.mean()
        value_loss = value_loss_mean * self.value_loss_coef

        return (
            value_loss_unclipped,
            value_loss_clipped,
            value_loss_batch,
            value_loss_mean,
            value_loss
        )

    @requires_input_process
    def compute_error_and_advantage(
        self,
        interpreted_trajectory,
        observation_critic_values=None,
        next_obs_critic_values=None
    ):
        reward_batch = interpreted_trajectory["reward"]
        next_obs_critic_values = (
            interpreted_trajectory["next_obs_critic_values"]
            if next_obs_critic_values is None else next_obs_critic_values
        )
        observation_critic_values = (
            interpreted_trajectory["observation_critic_values"]
            if observation_critic_values is None else observation_critic_values
        )
        done_batch = interpreted_trajectory["done"]
        truncated_batch = interpreted_trajectory["truncation"]

        if "alive_agents" in interpreted_trajectory:
            alive_agents = interpreted_trajectory["alive_agents"]

            reward_batch = reward_batch * alive_agents
            observation_critic_values = observation_critic_values * alive_agents
            next_obs_critic_values = next_obs_critic_values * alive_agents

            # Dead agents should stop GAE recursion immediately
            done_batch = torch.maximum(done_batch, 1.0 - alive_agents)

        return ppo.compute_gae_and_returns(
            reward_batch,
            observation_critic_values,
            next_obs_critic_values,
            done_batch,
            truncated_batch,
            self.discount_factor,
            self.lambda_gae
        )

    def _compute_losses(self, interpreted_trajectory):
        observation_critic_values, next_obs_critic_values = self.compute_values_estimates(
            interpreted_trajectory
        )

        interpreted_trajectory["observation_critic_values"] = observation_critic_values
        interpreted_trajectory["next_obs_critic_values"] = next_obs_critic_values

        value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss = \
            self._compute_critic_loss(interpreted_trajectory)

        return (
            value_loss_unclipped,
            value_loss_clipped,
            value_loss_batch,
            value_loss_mean,
            value_loss,
        )

    def _optimize_using_loss(self, value_loss):
        self.critic_optimizer.clear_optimizer_gradients()
        value_loss.backward()
        self.critic_optimizer.optimize_with_backward_pass_done()

    def _learn(self, trajectory: dict):
        super()._learn(trajectory)

        self.number_of_times_optimized += 1

        interpreted_trajectory = self.interpret_trajectory(trajectory)

        value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss = \
            self._compute_losses(interpreted_trajectory)

        self._optimize_using_loss(value_loss)

        return {
            "observation_critic_values": interpreted_trajectory["observation_critic_values"].detach(),
            "next_obs_critic_values": interpreted_trajectory["next_obs_critic_values"].detach(),
            "returns": interpreted_trajectory["returns"],
            "value_loss": value_loss.detach(),
            "value_loss_mean": value_loss_mean.detach(),
        }