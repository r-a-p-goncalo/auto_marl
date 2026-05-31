

from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.component import requires_input_process
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema, QLearnerSchema
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.components.ml.models.torch_model_utils import model_parameter_distance
from automarl.core.input_management import ParameterSignature
import torch

class LearnerDebug(LearnerSchema, ComponentDebug):

    is_debug_schema = True

    parameters_signature = {
        "compare_old_and_new_model_predictions" : ParameterSignature(default_value=True)
    }

    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.__agent_model = self.agent.policy.model


        self.compare_old_and_new_model_predictions = self.get_input_value("compare_old_and_new_model_predictions")
        
        if self.compare_old_and_new_model_predictions:
            self.lg.writeLine(f"Will compare old and new model predictions")
            self.__temporary_model : TorchModelComponent = self.__agent_model.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_policy", "create_new_directory" : False}, is_deep_clone=True)        

        else:
            self.lg.writeLine(f"Will not compare old and new model predictions")

    @requires_input_process
    def learn(self, trajectory) -> None:
        
        self.lg.writeLine(f"Learning {self.optimizations_per_learn} timess...")

        super().learn(trajectory)
        
        self.lg.writeLine(f"Ended learning")

    def _learn(self, trajectory) -> None:

        if self.compare_old_and_new_model_predictions:
            self.__temporary_model.clone_other_model_into_this(self.__agent_model)

            interpreted_trajectory = self.interpret_trajectory(trajectory)

        to_return = super()._learn(trajectory)

        self.lg.writeLine("\naction, reward, done, old_model_predictions, new_model_predictions\n", file="batch_comparison.txt", use_time_stamp=False)

        if self.compare_old_and_new_model_predictions:

            action_batch = interpreted_trajectory["action"]
            reward_batch = interpreted_trajectory["reward"]
            done_batch = interpreted_trajectory["done"]

            with torch.no_grad():
                old_model_predictions = self.__temporary_model.predict(interpreted_trajectory["observation"])
                new_model_precitions = self.__agent_model.predict(interpreted_trajectory["observation"])

            self.lg.writeLine(f"Used previous version of model and new version on observation_batch with shape {interpreted_trajectory['observation'].shape} to produce predicitons with shape {new_model_precitions.shape} and {old_model_predictions.shape}", file="batch_comparison.txt", use_time_stamp=False)

            for i in range(len(interpreted_trajectory["observation"])):

                action_val = action_batch[i].detach().cpu().numpy()
                reward_val = reward_batch[i].detach().cpu().numpy()
                done_val = done_batch[i].detach().cpu().numpy()
                old_model_prediction_val = old_model_predictions[i].detach().cpu().numpy()
                new_model_precitions_val = new_model_precitions[i].detach().cpu().numpy()

                self.lg.writeLine(f"{i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}", file="batch_comparison.txt", use_time_stamp=False)

        return to_return