import os
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.components.loggers.logger_component import ComponentWithLogging
import torch
import torch.nn as nn

from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.ml.models.model_components import ModelComponent

from automarl.components.ml.models.torch_model_components import TorchModelComponent

from automarl.components.ml.models.torch_model_utils import model_parameter_distance_by_params, split_shared_params

class TorchModelComponentDebug(TorchModelComponent, ComponentDebug):

    is_debug_schema = True

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "note_model_difference_on_init" : ParameterSignature(default_value=True)
    }    

    exposed_values = {
    }
    
    def _process_input_internal(self):

        
        super()._process_input_internal()

        

    @requires_input_process
    def predict(self, state):
        self.lg.writeLine(f"Predicting value for state with shape {state.shape}...", file="model_predictions.txt")
        to_return = super().predict(state)
        self.lg.writeLine(f"Predicted value with shape {to_return.shape}: {to_return}\n", file="model_predictions.txt")
        return to_return
    
    def _execute_model_initialization_strategy(self):

        self.__note_model_difference_on_init = False if self.model_initialization_strategy is None else self.get_input_value("note_model_difference_on_init")

        if self.__note_model_difference_on_init:
            olds_params = torch.cat([p.flatten() for p in self.model.parameters()])

        super()._execute_model_initialization_strategy()

        if self.__note_model_difference_on_init:

            new_params = torch.cat([p.flatten() for p in self.model.parameters()])

            l2_distance, avg_distance, cosine_sim = model_parameter_distance_by_params(olds_params, new_params)

            self.lg.writeLine(f"Difference between old and new model params, ater executing init strategy: l2: {l2_distance}, avg: {avg_distance}, cos: {cosine_sim}")

        
    
    def clone(self, save_in_parent=True, input_for_clone=None, is_deep_clone=True):

        cloned_component = super().clone(save_in_parent=save_in_parent, input_for_clone=input_for_clone, is_deep_clone=is_deep_clone)

        shared_params, self_only, cloned_only = split_shared_params(self, cloned_component)

        self.lg.writeLine(f"Cloned component: Shared params: {len(shared_params)}, Self only: {len(self_only)}, Cloned only: {len(cloned_only)}")

        return cloned_component


    def _save_model(self):
        
    
        model_path = os.path.join(self.get_artifact_directory(), "model_weights.pth")
        
        if os.path.exists(model_path):
            saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            self.lg.writeLine(f"Model already existed in file, comparing new with old:")

            params_a = torch.cat([
                p.detach().flatten().cpu()
                for p in self.model.state_dict().values()
            ])

            params_b = torch.cat([
                p.detach().flatten().cpu()
                for p in saved_state_dict.values()
            ])


            l2_distance = torch.norm(params_a - params_b, p=2).item()
            avg_distance = l2_distance / params_a.numel()
            cosine_sim = torch.nn.functional.cosine_similarity(
                params_a.unsqueeze(0), params_b.unsqueeze(0)
            ).item()

            self.lg.writeLine(f"L2 dis: {l2_distance}, Avg dist: {avg_distance}, Cos dist: {cosine_sim}")

        super()._save_model()