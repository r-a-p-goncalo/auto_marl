import torch
from automarl.components.rl.learners.ppo_learner import PPOLearner
from automarl.components.rl.learners.debug.learner_debug import LearnerDebug
from automarlcore.input_management import ParameterSignature
from automarl.components.ml.models.torch_model_components import TorchModelComponent

import torch.nn.functional as F

from automarl.components.basic_components.dynamic_value import get_value_or_dynamic_value


class PPOLearnerDebug(LearnerDebug, PPOLearner):

    is_debug_schema = True

    parameters_signature = {
        "interval_between_debug_writes": ParameterSignature(default_value=50),
        "compare_old_and_new_critic_predictions_interval": ParameterSignature(default_value=-1),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.interval_between_debug_writes = self.get_input_value("interval_between_debug_writes")

        self.compare_old_and_new_critic_predictions_interval = self.get_input_value("compare_old_and_new_critic_predictions_interval")

        self.__current_learning_step = 0
        self.__last_interval_checked_ppo_learning = -1

        self.lg.open_or_create_relative_folder("_ppo_debug")

        self.__compare_old_and_new_critic_predictions = self.compare_old_and_new_critic_predictions_interval >= 1

        if self.__compare_old_and_new_critic_predictions:
            self.lg.writeLine(f"Will compare old and new critic predictions")
            self.__old_critic_model: TorchModelComponent = self.critic.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_critic", "create_new_directory" : False}, is_deep_clone=True)

        else:
            self.lg.writeLine(f"Will not compare old and new critic predictions")


    def _split_actor_critic_params(self):
        shared_params, actor_only, critic_only = super()._split_actor_critic_params()

        self.lg.writeLine(f"Split actor and critic params: Shared: {len(shared_params)}, Actor only: {len(actor_only)}, Critic only: {len(critic_only)}")

        return shared_params, actor_only, critic_only


    def _should_log(self):

        # if we stil have not checked in the current interval
        if self.__last_interval_checked_ppo_learning < self.__current_learning_step:
    
            if not self.__current_learning_step % self.interval_between_debug_writes == 0:
                self.__debug_path = None
            
            else:
                self.__debug_path = self.lg.new_relative_path_if_exists(
                    "ppo_step.txt", dir="_ppo_debug"
                )

            self.__last_interval_checked_ppo_learning = self.__current_learning_step

        
        return self.__debug_path is not None
        

            

    
    def compute_values_estimates(self, interpreted_trajectory):
        values, next_values = super().compute_values_estimates(interpreted_trajectory)

        if self._should_log():

            action_batch = interpreted_trajectory.get("action", None)
            done_batch = interpreted_trajectory["done"]


            self.lg.writeLine(
                    "\nCritic for state, action, Critic for next state, done batch",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )
            
            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: state_value {values[i]} and action {action_batch[i]} -> next state value {next_values[i]}, done: {done_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        return values, next_values
    
    def compute_error_and_advantage(self, interpreted_trajectory):
        critic_obs_pred_error, non_normalized_advantages, advantages, returns = super().compute_error_and_advantage(
            interpreted_trajectory
        )

        if self._should_log():

            reward_batch = interpreted_trajectory["reward"]
            next_values = interpreted_trajectory["values"]
            values = interpreted_trajectory["old_values"]
            done_batch = interpreted_trajectory["done"]

            self.lg.writeLine(f"\nValue error calculation: (reward + discount_factor * next_values - values)",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {critic_obs_pred_error[i]} = {reward_batch[i]} + {self.discount_factor} * {next_values[i]} - {values[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nAdvantage calculation:  critic_obs_pred_error + ( discount_factor * self.lamda * prev_advantage if not done) -> normalized value",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            prev_advantage = 0

            for i in reversed(range(len(non_normalized_advantages))):
                
                self.lg.writeLine(
                    f"{i}: {non_normalized_advantages[i]} = {critic_obs_pred_error[i]} + {self.discount_factor} * {self.lambda_gae} * {prev_advantage} * (1 - {done_batch[i]}) -> {advantages[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

                prev_advantage = non_normalized_advantages[i]

            self.lg.writeLine(
                "\nAdvantage statistics:",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            self.lg.writeLine(
                f"mean={advantages.mean():.6f}, std={advantages.std():.6f}, "
                f"min={advantages.min():.6f}, max={advantages.max():.6f}",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {critic_obs_pred_error[i]} = {reward_batch[i]} + {discount_factor} * {next_values[i]} - {values[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nReturns (for critic) = values critic predicted + advantages",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {returns[i]} = {values[i]} + {non_normalized_advantages[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        return critic_obs_pred_error, non_normalized_advantages, advantages, returns
    
    def _evaluate_actions(self, interpreted_trajectory):
        super()._evaluate_actions(interpreted_trajectory)

        if self._should_log():

            model_output = interpreted_trajectory["model_output"]
            actions = interpreted_trajectory["action"]
            new_log_probs = interpreted_trajectory["new_log_probs"]

            self.lg.writeLine(
                "\nEvaluation of actions:\nmodel_output selected by action -> log_probs of chosen actions",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(model_output)):

                self.lg.writeLine(
                    f" | output {model_output[i]} | -> | action {actions[i]} | -> | log {new_log_probs[i]} |",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

    

    def _compute_policy_loss(self, new_log_probs, log_prob_batch, advantages, entropy):
    
        ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss = super()._compute_policy_loss(new_log_probs, log_prob_batch, advantages, entropy)
    
        if self._should_log():

            self.lg.writeLine(f"\nRatio calculation: exp(new_log_probs - log_prob_batch)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"    {i}: {ratio[i]} = exp({new_log_probs[i]} - {log_prob_batch[i]})",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nPolicy loss calculation:\nMin(Ratio * Advantages, Clipped values)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"    {i}: Min({ratio[i]} * {advantages[i]} = {surrogate1[i]}, {surrogate2[i]}) -> {policy_loss_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nMean policy loss: {mean_policy_loss}, After entropy (entropy {entropy} * coef {get_value_or_dynamic_value(self.entropy_coef)}): {policy_loss}\n", file=self.__debug_path, use_time_stamp=False)

        return ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss

    def _compute_critic_loss(self, interpreted_trajectory):

        value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss = super()._compute_critic_loss(interpreted_trajectory)
        
        if self._should_log():
        
            self.lg.writeLine(f"\nCritic loss calculation:\nMin(error (critic_predicted, return) , Clipped values)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(value_loss_unclipped)):
                self.lg.writeLine(
                    f"    {i}: Min( ( {values[i]} - {returns[i]} )^2 = {value_loss_unclipped[i]}, {value_loss_clipped[i]}) -> {value_loss_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )
        
            self.lg.writeLine(f"\nMean value loss: {value_loss_mean}, After coef ({self.value_loss_coef}): {value_loss}\n", file=self.__debug_path, use_time_stamp=False)

        return value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss


    def _learn(self, trajectory, discount_factor):

        if self._should_log():
            self.lg.writeLine(f"Interpreting trajectory with keys: {trajectory.keys()}", file=self.__debug_path,
                    use_time_stamp=False,)


        interpreted_trajectory = self.interpret_trajectory(trajectory)

        if self._should_log():
            

            self.lg.writeLine(f"\nDoing learning with shapes:",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            for k, v in interpreted_trajectory.items():
                self.lg.writeLine(f"    {k}: {v.shape}", file=self.__debug_path, use_time_stamp=False)

        should_compare_old_and_new_critic = self.__compare_old_and_new_critic_predictions and self.__current_learning_step % self.compare_old_and_new_critic_predictions_interval == 0

        if should_compare_old_and_new_critic:
            self.__old_critic_model.clone_other_model_into_this(self.critic)

        to_return = super()._learn(trajectory, discount_factor)
            
        if should_compare_old_and_new_critic:

            with torch.no_grad():

                old_values = self.__old_critic_model.predict(interpreted_trajectory["observation"])
                new_values = self.critic.predict(interpreted_trajectory["observation"])

                reward_batch = interpreted_trajectory["reward"]
                action_batch = interpreted_trajectory["action"]
                done_batch = interpreted_trajectory["done"]

                self.lg.writeLine(
                    "\naction, reward, done, old_model_predictions, new_model_predictions\n",
                    file="batch_comparison_critic.txt",
                    use_time_stamp=False,
                )

                for i in range(len(interpreted_trajectory["observation"])):

                    action_val = action_batch[i].detach().cpu().numpy()
                    reward_val = reward_batch[i].detach().cpu().numpy()
                    done_val = done_batch[i].detach().cpu().numpy()
                    old_model_prediction_val = old_values[i].detach().cpu().numpy()
                    new_model_precitions_val = new_values[i].detach().cpu().numpy()

                    self.lg.writeLine(
                        f"    {i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}",
                        file="batch_comparison_critic.txt",
                        use_time_stamp=False,
                    )

        self.__current_learning_step += 1

        return to_return