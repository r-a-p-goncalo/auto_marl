

from automarl.component import  ParameterSignature, requires_input_process



from automarl.components.external_support.sb3.sb3_utils import load_sb3_dqn_model
from automarl.components.rl.policy.policy import PolicyInterface


class SB3Wrapper(PolicyInterface):
    
    parameters_signature = {
        "sb3_model" : ParameterSignature(default_value="dqn-MountainCar-v0")
    }    
    
    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.sb3_model = self.get_input_value("sb3_model")
        
        if isinstance(self.sb3_model, str):
            self.sb3_model = load_sb3_dqn_model(self.sb3_model)
    
    
    @requires_input_process
    def predict(self, state):
    
        action, hidden_state = self.sb3_model.predict(state, deterministic=True) #deterministic = True uses policy, deterministic = False simulates training behavior
        return action
    
    def get_policy_output_shape(self):
        raise NotImplementedError()
    
    
        
    @requires_input_process
    def random_prediction(self, state):
        action, hidden_state =  self.sb3_model.predict(state, deterministic=False)
        return action
    
    

    