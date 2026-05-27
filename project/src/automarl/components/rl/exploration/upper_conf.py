



from automarl.components.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automarl.component import ParameterSignature

import random
import math
import torch


class UpperConfidenceBoundStrategy(ExplorationStrategySchema):
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                       "n_action" : ParameterSignature(),
                       "exploration_param" : ParameterSignature(default_value=0.01)
                       }    
    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
        self.counts = torch.zeros(self.values["output_size"], device=self.device)
        
        self.exploration_param  = self.values["exploration_param"]
        
        
        
    def select_action(self, agent, state):
        
        super().select_action(state)
        
        with torch.no_grad():
            
            q_values = agent.policy_predict(state)
                    
            #if we still have not tried a certain action
            if 0 in self.counts:
                action = torch.argmin(self.counts)
            
            else:
            
                #the value we choose is based on the predicted and
                ucb_values = q_values + self.exploration_param * torch.sqrt(math.log(self.training_context["total_steps"]) / (self.counts))        
                action = torch.argmax(ucb_values)
            
            self.counts[action] += 1
            
            return action.item()