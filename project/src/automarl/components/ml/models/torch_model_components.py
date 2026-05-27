import os
from automarl.components.basic_components.state_management import StatefulComponent
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.components.loggers.logger_component import ComponentWithLogging
import torch
import torch.nn as nn

from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion

from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.ml.models.model_components import ModelComponent

from automarl.core.advanced_input_management import ComponentParameterSignature

from automarl.components.ml.models.model_initialization_strategy import TorchModelInitializationStrategy

class TorchModelComponent(ModelComponent, StatefulComponent, ComponentWithLogging):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {

        "device": ParameterSignature(get_from_parent=True, ignore_at_serialization=True),
        "model" : ParameterSignature(mandatory=False, possible_types=[nn.Module]),

        "parameters_initialization_strategy" : ComponentParameterSignature(mandatory=False)
    }    

    exposed_values = {
        "model" : None
    }
    
    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.lg.writeLine(f"Processing model input...\n")

        self.device = self.get_input_value("device")

        self.model_initialization_strategy : TorchModelInitializationStrategy = self.get_input_value("parameters_initialization_strategy")

        self.__synchro_model_value_attr()
                        
        self._setup_model()

        if self.device != None:
            self.lg.writeLine(f"Model will be on device {self.device}\n")
            self.model.to(self.device)

        self.lg.writeLine(f"Finished processing model input\n")

    def __synchro_model_value_attr(self):

        '''To call mainly when the model attribute is updated'''

        if hasattr(self, "model"):
            self.values["model"] = self.model 

        elif self.values["model"] != None: # if a model is already present in the exposed values, use it
            self.model = self.values["model"]


        
    def _setup_values(self):
        super()._setup_values()
        

    def _execute_model_initialization_strategy(self):

        if self.model_initialization_strategy is None:
            self.lg.writeLine(f"No model initialization passed")

        else:
            self.lg.writeLine(f"Initializing model with strategy of type: {type(self.model_initialization_strategy)}...")
            self.model_initialization_strategy.initialize_model(self.model)


    def _setup_model(self):

        '''Sets up the model, loading it or initializing it'''

        self.lg.writeLine("Setting up model")
        
        model_loaded = self._is_model_loaded()

        if not model_loaded:

            self.lg.writeLine("Model is not loaded, trying to load it...")
            
            model_was_loaded = self._try_load_model()
            
            if model_was_loaded:
                self.lg.writeLine(f"Success in loading the model")

            else:
                self.lg.writeLine(f"Could not load model, initiating initialization strategy...")
                self._initialize_model() # initializes the model using passed values
                self._execute_model_initialization_strategy()

        else:
            self.lg.writeLine(f"Model was already loaded")

        self.__synchro_model_value_attr()

        self._is_model_well_formed() # throws exception if model is not well formed

        self.lg.writeLine(f"Model setup is over\n")


    def _is_model_well_formed(self):
        '''Raises exception if model is not well formed'''
        if not self._is_model_loaded():
            raise Exception("Failed to load the model")
            
            
    def _is_model_loaded(self):
        return hasattr(self, "model") and self.model is not None
    

    def _try_load_model(self):
        '''Tries load model, and returns True if model was loaded, false otherwise'''

        model_was_loaded = self._try_load_model_from_input()

        if model_was_loaded:
            return model_was_loaded
        
        model_was_loaded = self._try_load_model_from_path()

        if model_was_loaded:
            return model_was_loaded

        return False
    
    
    def _try_load_model_from_input(self):

        '''Ties load model from input, and returns True if model was loaded, false otherwise'''

        self.lg.writeLine("Trying to load model from input....")

        input_model = self.get_input_value("model")

        if input_model != None:

            self.model : nn.Module = self.get_input_value("model")
            self.values["model"] = self.model
            self.lg.writeLine(f"Success in loading model from input")
            return True
        
        else:
            self.lg.writeLine("No model available in input to load")
            return False


    def _try_load_model_from_path(self):

        '''Ties load model from input, and returns True if model was loaded, false otherwise'''

        self.lg.writeLine("Trying to load model from path...")

        model_path = os.path.join(self.get_artifact_directory(), "model_weights.pth")
        
        if os.path.exists(model_path):
                
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
            self._initialize_mininum_model_architecture()  # Ensure the model as its architecture initialized before loading weights

            self.lg.writeLine(f"Loading model weights in file into architecture of model...")

            self.model.load_state_dict(state_dict) #loads the saved weights into the model

            self.lg.writeLine("Success in loading model from path\n")

            return True
        
        else:

            self.lg.writeLine("Could not find model weights to load (/model_weights.pth)\n")
        
            return False


        
    def _initialize_mininum_model_architecture(self):

        '''Initializes the minimum architecture, this is used to load weights of the model, so the initial parameters are to be ignored'''

        self.lg.writeLine("Initializing minimum model architecture...")

        pass


    def _initialize_model(self):
        '''Initializes a totally new model'''
        
        self.lg.writeLine("Totally initialized model")
        
    # EXPOSED METHODS --------------------------------------------
    
    @requires_input_process
    def get_model_params(self):
        '''returns a list of model parameters'''
        return list(self.model.parameters())
    
    @requires_input_process
    def get_model_state_dict(self):
        '''returns a list of model parameters'''
        return self.model.state_dict()
    
    @requires_input_process
    def predict(self, state):
        super().predict(state)
        return self.model(state)

    
    @requires_input_process            
    def update_model_with_target(self, target_model, target_model_weight):
        
        '''
        Updates the weights of this model with the weights of the target model
        
        @param target_model_weight is the relevance of the target model, 1 will mean a total copy, 0 will do nothing, 0.5 will be an average between the models
        '''

        if target_model_weight == 1:
            self.clone_other_model_into_this(target_model)

        else:
        
            with torch.no_grad():
                this_model_state_dict = self.model.state_dict()
                target_model_state_dict = target_model.model.state_dict()
    
                for key in target_model_state_dict:
                    this_model_state_dict[key] = (
                        target_model_state_dict[key] * target_model_weight + 
                        this_model_state_dict[key] * (1 - target_model_weight)
                    )
                
                self.model.load_state_dict(this_model_state_dict)
    
    # UTIL -----------------------------------------------------
    
    def _input_to_clone(self):
        input_to_clone = super()._input_to_clone()

        input_to_clone.pop("model", None)
        input_to_clone.pop("parameters_initialization_strategy", None)

        return input_to_clone
    
    def _values_to_clone(self):
        
        values_to_clone =  super()._values_to_clone()

        values_to_clone["model"] = None

        return values_to_clone

    
    def _after_clone(self, original, is_deep_clone):

        '''The cloned model clones the parameters of the original into it'''

        super()._after_clone(original, is_deep_clone)
        self.clone_other_model_into_this(original)

    
    @requires_input_process
    def clone_other_model_into_this(self, other_model):

        other_model : TorchModelComponent = other_model
        other_model.process_input_if_not_processed()
        self.model.load_state_dict(other_model.model.state_dict())
    
    # STATE MANAGEMENT -----------------------------------------------------

    def _save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.get_artifact_directory(), "model_weights.pth"))



    def _save_state_internal(self):
        
        super()._save_state_internal()

        if hasattr(self, "model"):
            self._save_model()
                
    
    def _load_state_internal(self) -> None:
        
        super()._load_state_internal()
                


    @requires_input_process
    def get_model_input_shape(self):
        return self.model.get_input_shape()
    
    @requires_input_process
    def get_model_output_shape(self):
        return self.model.get_output_shape()