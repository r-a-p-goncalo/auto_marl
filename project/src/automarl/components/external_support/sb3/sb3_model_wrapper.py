
import copy
from automarl.component import Component, ParameterSignature, requires_input_process

from automarl.components.external_support.sb3.sb3_utils import load_policy_network_from_architecture, load_sb3_net
from automarl.components.ml.models.model_components import ModelComponent
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.serialization.shape_json_utils import CustomSpaceJsonEncoderDecoder # the act of importing this registers it
from automarl.components.loggers.global_logger import globalWriteLine
import torch


SUPPORTED_MODELS = ["dqn-MountainCar-v0", "dqn-CartPole-v1", "ppo-CartPole-v1"]

class SB3WrapperTorch(TorchModelComponent):
    
    parameters_signature = {
        "sb3_model" : ParameterSignature(default_value="dqn-MountainCar-v0", mandatory=False)
    }    
    

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.sb3_model = self.get_input_value("sb3_model")
    

    def _try_load_model(self):
        
        return super()._try_load_model()


    
    def _initialize_model(self):

        super()._initialize_model()

        self.initialize_model_with_sb3_architecture()




    def initialize_model_with_sb3_architecture(self):
    
        sb3_model_name = self.get_input_value("sb3_model")

        if sb3_model_name == None:
            Exception("No sb3_model_provided")

        self.lg.writeLine(f"Sb3 model component {self.name} has no model already loaded and has sb3_model defined with name {sb3_model_name}, loading it...")
        
        # Clone network
        model_net, architecture = load_sb3_net(sb3_model_name)
                
        self.model = copy.deepcopy(model_net)

        self.values["sb3_architecture"] = copy.deepcopy(architecture)
 
        # Explicitly load weights (not strictly needed, deepcopy already does it)
        self.model.load_state_dict(model_net.state_dict())

        self.write_line_to_notes(f"Loaded into this sb3 model using name {sb3_model_name}", use_datetime=True)




    def _initialize_mininum_model_architecture(self):
        """
        Rebuilds the model architecture using stored SB3 metadata (without re-downloading the checkpoint).
        """

        super()._initialize_mininum_model_architecture()

        model_architecture = self.values["sb3_architecture"]
        
        self.model = load_policy_network_from_architecture(model_architecture)

        self.lg.writeLine(f"Success in sb3 model using saved architecture and weights")
        
            
    @requires_input_process
    def predict(self, state):
        
        if state.ndim == 1:
            state = state.unsqueeze(0)   # make it (1, obs_dim)

        
        return self.model(state)
