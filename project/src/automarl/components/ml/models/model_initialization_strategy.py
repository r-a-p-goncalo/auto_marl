import os
from automarl.components.loggers.global_logger import globalWriteLine
import torch
import torch.nn as nn

from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion

from automarl.component import Component, ParameterSignature, requires_input_process


class TorchModelInitializationStrategy(Component):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
    }    

    
    def _process_input_internal(self):
        
        super()._process_input_internal()


    @requires_input_process
    def initialize_model(self, model : nn.Module):
        '''Initializes the model parameters'''


class TorchModelInitializationStrategyOrthogonal(TorchModelInitializationStrategy):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "gain" : ParameterSignature(default_value=1.0)
    }    

    
    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.gain = self.get_input_value("gain")

    
    def _init_layer_orthogonal(self, layer):

        if hasattr(layer, "weight") and layer.weight is not None:
            nn.init.orthogonal_(layer.weight, self.gain)

        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


    def initialize_model(self, model : nn.Module):

        super().initialize_model(model)

        for module in model.modules():

            # Only initialize leaf modules with parameters
            if isinstance(module, nn.Linear):
                self._init_layer_orthogonal(module)

            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                self._init_layer_orthogonal(module)

    

class TorchModelInitializationGaussian(TorchModelInitializationStrategy):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "mean" : ParameterSignature(default_value=0.0),
        "std" : ParameterSignature(default_value=1.0)
    }    

    
    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.mean = self.get_input_value("mean")
        self.std = self.get_input_value("std")

    
    def _init_layer_gaussian(self, layer):

        if hasattr(layer, "weight") and layer.weight is not None:
            nn.init.normal_(layer.weight, mean=self.mean, std=self.std)

        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


    def initialize_model(self, model : nn.Module):

        super().initialize_model(model)

        for module in model.modules():

            # Only initialize supported parameterized layers
            if isinstance(module, nn.Linear):
                self._init_layer_gaussian(module)

            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                self._init_layer_gaussian(module)



