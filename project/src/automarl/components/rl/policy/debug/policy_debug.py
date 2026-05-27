from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.rl.policy.policy import Policy
from automarl.components.rl.policy.qpolicy import QPolicy
from automarl.components.rl.policy.stochastic_policy import StochasticPolicy
from automarl.component import requires_input_process
import torch

class PolicyDebug(Policy, ComponentDebug):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''

    is_debug_schema = True
        
    parameters_signature = {

    }   

    
    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.lg.writeLine(f"Processing policy debug input...\n")
        self.lg.writeLine(f"Finished policy debug input...\n")
    

        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        state_str = str(state)

        if len(state_str) > 35:
            state_str = state_str[:15] + " ... " + state_str[-15:]
        
        self.lg.writeLine(f"Making whole prediction from state: {state_str}", file='predicted_values.txt')

        to_return = super().predict(state)

        self.lg.writeLine(f"predicted: {to_return}\n", file='predicted_values.txt')

        return to_return
    
        
    def random_prediction(self, state):    

        random_predicted_value = super().random_prediction(state)

        self.lg.writeLine(f"Randomly predicted value: {random_predicted_value}", file='random_predicted_values.txt')

        return random_predicted_value
    


class QPolicyDebug(PolicyDebug, QPolicy):

    is_debug_schema = True

    


class StochasticPolicyDebug(PolicyDebug, StochasticPolicy):

    is_debug_schema = True
    

    def distribution_from_model_output(self, model_output, state) -> torch.Tensor:

        to_return = super().distribution_from_model_output(model_output, state)

        self.lg.writeLine(f"    model_output {model_output} -> distribution {to_return}", file='predicted_values.txt')

        return to_return

    
    
    def sample_action_val_from_distribution(self, distribution : torch.distributions, state):

        to_return = super().sample_action_val_from_distribution(distribution, state)

        self.lg.writeLine(f"    distribution {distribution} -> action_val {to_return}", file='predicted_values.txt')

        return to_return

    
    def log_probability_of_action_val(self, distribution, action_val, state):

        to_return = super().log_probability_of_action_val(distribution, action_val, state)

        self.lg.writeLine(f"    distribution {distribution} + action_val {action_val} -> log_prob {to_return}", file='predicted_values.txt')

        return to_return

    
    @requires_input_process
    def predict_action_val_from_model_output_with_log(self, model_output, state):
                
        self.lg.writeLine(f"Predicting action_val and log_prob from model_output:", file='predicted_values.txt')

        to_return = super().predict_action_val_from_model_output_with_log(model_output, state)

        self.lg.writeLine(f"", file='predicted_values.txt')

        return to_return
    
    @requires_input_process
    def predict_action_val_with_log(self, state):
                            
        self.lg.writeLine(f"Predicting action_val and log_prob:", file='predicted_values.txt')

        to_return = super().predict_action_val_with_log(state)

        self.lg.writeLine(f"", file='predicted_values.txt')

        return to_return
    
    @requires_input_process
    def get_action_from_action_val(self, action_val):
        
        to_return = super().get_action_from_action_val(action_val)
        self.lg.writeLine(f"action_val {action_val} -> action {to_return}", file='predicted_values.txt')
        return to_return