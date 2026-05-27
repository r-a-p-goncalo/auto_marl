from automarl.components.basic_components.artifact_management import generate_target_directory
from automarl.component import requires_input_process
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.components.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn

class ModelSequenceComponent(TorchModelComponent):
        
    # The actual model architecture
    class Model_Class(nn.Module):
        
        def __init__(self, models : list[nn.Module]):

            super().__init__()
            
            self.models = nn.ModuleList(models)

        def forward(self, x : torch.Tensor):
            
            for model in self.models:
                x = model(x)
            
            return x
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "models" : ComponentListParameterSignature(),

    }    
    
    def _process_input_internal(self):
        
        super()._process_input_internal()
        

    def _setup_values(self):
        super()._setup_values()    

        self.models : list[TorchModelComponent] = self.get_input_value("models")



    def _initialize_mininum_model_architecture(self):
    
        '''
        Initializes the model with no regard for initial parameters, as they are meant to be loaded
        This method is meant to be called even if the input isn't fully processed
        '''

        super()._initialize_mininum_model_architecture()

        self._setup_values() # this needs the values from the input fully setup

        for model in self.models:
            self.lg.writeLine(f"Initializing model minimum architecture for {model.name}")
            model._initialize_mininum_model_architecture()

        self.model : nn.Module = type(self).Model_Class(
            models=[model.model for model in self.models]
            )


    def _initialize_models(self):

        common_model_input = {
            "device" : self.device
        }


        self.models[0].pass_input({"input_shape" : self.input_shape, **common_model_input})
        self.models[-1].pass_input({"output_shape" : self.output_shape, **common_model_input})

        if len(self.models) >= 2:

            for model_index in range(len(self.models) - 1):
                current_model = self.models[model_index]
                next_model = self.models[model_index + 1]

                current_model.process_input_if_not_processed()

                current_model_output_shape = current_model.get_model_output_shape()

                next_model.pass_input({"input_shape" : current_model_output_shape, **common_model_input})

                self.lg.writeLine(f"Connecting models: {current_model.name} -> {current_model_output_shape} -> {next_model.name}")

            next_model.process_input_if_not_processed() # process last model

            self.lg.writeLine(f"Connecting models: {next_model.name} -> {next_model.get_model_output_shape()}")
        
        else: # initialize the only model available
            self.models[0].process_input_if_not_processed()


    def _initialize_model(self):

        '''Initializes the model with initial parameter strategy'''

        super()._initialize_model()

        self._initialize_models()
    
        self.model : nn.Module = type(self).Model_Class(
            models=[model.model for model in self.models]
            )
        
    def _input_to_clone(self):
        input_to_clone = super()._input_to_clone()

        input_to_clone.pop("models", None)

        return input_to_clone

    def _clone(self, input_for_clone=None, is_deep_clone=False) -> TorchModelComponent:

        if input_for_clone is None:
            input_for_clone = {}

        models_were_cloned = False        

        if not "models" in input_for_clone.keys():
            assumed_cloned_directory = generate_target_directory(input_for_clone)
            
            cloned_models : list[TorchModelComponent] = []
            
            if is_deep_clone:
                cloned_models = [
                    model.clone(
                        save_in_parent=False, is_deep_clone=True, input_for_clone={"create_new_directory" : True, "base_directory" : assumed_cloned_directory}
                        ) for model in self.models
                        ]
                models_were_cloned = True
            
            else:
                cloned_models = [model for model in self.models]
        
            input_for_clone["models"] = cloned_models

        cloned_component = super()._clone(input_for_clone, is_deep_clone)

        if models_were_cloned:
            for cloned_model in cloned_models:
                cloned_component.define_component_as_child(cloned_model)

        return cloned_component
    
    def _save_model(self):
        self.lg.writeLine(f"Model is being saved... Naturally their referenced models will have their parameters saved")
    
    def _is_model_well_formed(self):
        super()._is_model_well_formed()
                            