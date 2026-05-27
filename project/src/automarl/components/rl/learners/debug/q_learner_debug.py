

from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.component import requires_input_process
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema, QLearnerSchema
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.components.ml.models.torch_model_utils import model_parameter_distance
from automarlcore.input_management import ParameterSignature
import torch

from automarl.components.rl.learners.debug.learner_debug import LearnerDebug

class QLearnerDebug(LearnerDebug, QLearnerSchema):

    is_debug_schema = True

    parameters_signature = {
        "interval_beetwenn_computation_writes" : ParameterSignature(default_value=10)
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.lg.open_or_create_relative_folder("learning")

        self.interval_beetwenn_computation_writes = self.get_input_value("interval_beetwenn_computation_writes")
        self.__current_interval_computation_count = 1

        self.lg.writeLine(f"Interval between computation writes will be {self.interval_beetwenn_computation_writes}")


    def _apply_model_prediction_given_state_action_pairs(self, interpreted_trajectory):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed''' 

        predicted_actions_values, predicted_values_for_actions = super()._apply_model_prediction_given_state_action_pairs(interpreted_trajectory)

        action_batch = interpreted_trajectory["action"]

        if self.__path_to_write is not None:

            self.lg.writeLine(f"\nComputed predicted_actions_values and value for action chosen:\n", file=self.__path_to_write, use_time_stamp=False)

            for i in range(len(action_batch)):
                self.lg.writeLine(f"{i}: {predicted_actions_values[i]} [ {action_batch[i]} ] -> {predicted_values_for_actions[i]}", file=self.__path_to_write, use_time_stamp=False)
    
        return predicted_actions_values, predicted_values_for_actions


    def _apply_value_prediction_to_next_state(self, interpreted_trajectory):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''

        done_batch = interpreted_trajectory["done"]

        next_state_q_values, next_state_v_values = super()._apply_value_prediction_to_next_state(interpreted_trajectory)

        if self.__path_to_write is not None:
            self.lg.writeLine(f"\nComputed done, next_state_values computed by target and q value of action chosen:\n", file=self.__path_to_write, use_time_stamp=False)

            for i in range(self.batch_size):
                self.lg.writeLine(f"{i}: {done_batch[i]}, {next_state_q_values[i]} -> {next_state_v_values[i]}", file=self.__path_to_write, use_time_stamp=False)


        return next_state_q_values, next_state_v_values
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):

        old_action_values = next_state_v_values.clone()

        correct_q_values_for_chosen_action = super()._calculate_chosen_actions_correct_q_values(next_state_v_values, discount_factor, reward_batch)

        if self.__path_to_write is not None:

            self.lg.writeLine(f"\nNext action values after multiplying by discount factor {discount_factor} and adding reward:\n", file=self.__path_to_write, use_time_stamp=False)

            for i in range(len(old_action_values)):
                self.lg.writeLine(f"{i}: {correct_q_values_for_chosen_action[i]} = {old_action_values[i]} * {discount_factor} + {reward_batch[i]}", file=self.__path_to_write, use_time_stamp=False)

        return correct_q_values_for_chosen_action
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):

        if self.__path_to_write is not None:

            self.lg.writeLine(f"\nOptimizing using error of original predicted action values and target done on future state:\n", file=self.__path_to_write, use_time_stamp=False)

            for i in range(len(predicted_values)):
                self.lg.writeLine(f"{i}: {predicted_values[i]} vs {correct_values[i]}", file=self.__path_to_write, use_time_stamp=False)

        super()._optimize_with_predicted_model_values_and_correct_values(predicted_values, correct_values)


    def _learn(self, trajectory) -> None:

        if self.interval_beetwenn_computation_writes % self.__current_interval_computation_count == 0:
            self.__path_to_write = self.lg.new_relative_path_if_exists("computation.txt", dir="learning")
        
        else:
            self.__path_to_write = None

        to_return = super()._learn(trajectory)

        self.__current_interval_computation_count += 1

        return to_return





class DQNLearnerDebug(QLearnerDebug, DeepQLearnerSchema):

    is_debug_schema = True

    parameters_signature = {
        "compare_old_and_new_target_predictions" : ParameterSignature(default_value=False),
        "compare_old_and_new_target_model_params" : ParameterSignature(default_value=False),
    }

    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.compare_old_and_new_target_predictions = self.get_input_value("compare_old_and_new_target_predictions")  
        self.compare_old_and_new_target_model_params = self.get_input_value("compare_old_and_new_target_model_params")        
        
        if self.compare_old_and_new_target_predictions: 
            self.lg.writeLine(f"Will compare old and new target predictions")
            self.__temporary_target_model = self.target_net.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_predictions"}, is_deep_clone=True)


        if self.compare_old_and_new_target_model_params:
            self.lg.writeLine(f"Will compare old and new target params")
            self.__temporary_target_model_v2 = self.target_net.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_params"}, is_deep_clone=True)
    
    def _learn(self, trajectory) -> None:

        if self.compare_old_and_new_target_predictions:

            self.__temporary_target_model.clone_other_model_into_this(self.target_net)

            observation_batch, action_batch, next_observation_batch, reward_batch, done_batch, *_ = self.interpret_trajectory(trajectory)

        super()._learn(trajectory)

        if self.compare_old_and_new_target_predictions and self.number_optimizations_done % self.target_update_learn_interval == 0:

            self.lg.writeLine("\naction, reward, done, old_target_predictions, new_target_precitions\n", file="target_batch_comparison.txt", use_time_stamp=False)

            with torch.no_grad():
                old_model_predictions = self.__temporary_target_model.predict(observation_batch)
                new_model_precitions = self.target_net.predict(observation_batch)

            for i in range(len(observation_batch)):

                action_val = action_batch[i].detach().cpu().numpy()
                reward_val = reward_batch[i].detach().cpu().numpy()
                done_val = done_batch[i].detach().cpu().numpy()
                old_model_prediction_val = old_model_predictions[i].detach().cpu().numpy()
                new_model_precitions_val = new_model_precitions[i].detach().cpu().numpy()

                self.lg.writeLine(f"{i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}", file="target_batch_comparison.txt", use_time_stamp=False)

    @requires_input_process            
    def update_target_model(self):

        if self.compare_old_and_new_target_model_params:

            self.lg.writeLine(f"Updating target model\n", file="target_update.txt")
            self.__temporary_target_model_v2.clone_other_model_into_this(self.target_net)

        super().update_target_model()

        if self.compare_old_and_new_target_model_params:

            l2_distance, avg_distance, cosine_sim = model_parameter_distance(self.__temporary_target_model_v2, self.target_net)

            self.lg.writeLine(f"Difference between old and new target model", use_time_stamp=False, file="target_update.txt")
            self.lg.writeLine(f"    l2_distance: {l2_distance}\n    avg_distance: {avg_distance}\n    cosine_sime: {cosine_sim}\n\n", file="target_update.txt", use_time_stamp=False)
    