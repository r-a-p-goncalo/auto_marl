

from automarl.components.rl.policy.policy import Policy
from automarl.component import requires_input_process
import random

class RandomPolicy(Policy):

    '''
    A policy which has no need for internal model, as it will choose always a random action
    '''

    parameters_signature = {}

    def _process_input_internal(self):
        super()._process_input_internal()

    def _initialize_model(self):
        '''
        Random policy does not use a model.
        Override the parent behavior so no model is required.
        '''
        self.lg.writeLine("RandomPolicy does not require a model.")
        self.model = None
        self.values["model"] = None

    def _setup_model(self):
        '''
        No model setup is needed for a random policy.
        Still compute the output shape for consistency.
        '''
        self._compute_model_output_shape()
        self.lg.writeLine("RandomPolicy has no model to setup.")

    @requires_input_process
    def predict(self, state=None):
        return self.output_action_shape.sample()
    



class RandomPolicyMasked(RandomPolicy):

    '''
    A policy which has no need for internal model, as it will choose always a random action, and allows for masked actions
    '''


    @requires_input_process
    def predict(self, state):
        action_mask = state["action_mask"]
        
        return self.output_action_shape.sample(action_mask)