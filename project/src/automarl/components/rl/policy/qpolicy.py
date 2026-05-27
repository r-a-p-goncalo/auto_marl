from automarl.component import  ParameterSignature, requires_input_process


import torch

import random

from automarl.components.rl.policy.policy import Policy
from automarl.utils.shapes_util import reduce_space_dimension


class QPolicy(Policy):
    '''
    It uses a model to calculate the Q action values of each action and calculates the action as such
    '''
        
    parameters_signature = {
        
    }   
    
    def _process_input_internal(self):
        
        super()._process_input_internal()       

        
    @requires_input_process
    def get_action_val_from_model_output(self,  q_values, state):

        #tensor of max values and tensor of indexes
        _, max_indexes = q_values.max(dim=1)
                        
        return max_indexes
    

class MaskedQPolicy(QPolicy):
    
    parameters_signature = {}

    INVALID_Q_VALUE = -1e9

    def _process_input_internal(self):
        super()._process_input_internal()

    def _normalize_action_mask(self, action_mask, q_values):
        '''
        Converts action_mask to boolean mask on same device as q_values.
        '''
        if action_mask is None:
            return None

        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, device=q_values.device)

        action_mask = action_mask.to(device=q_values.device)

        if action_mask.dtype != torch.bool:
            action_mask = action_mask > 0

        return action_mask

    def _mask_q_values(self, q_values, action_mask):
        '''
        Applies invalid-action masking to Q-values.
        '''
        if action_mask is None:
            return q_values

        action_mask = self._normalize_action_mask(action_mask, q_values)

        return q_values.masked_fill(~action_mask, self.INVALID_Q_VALUE)
    
    def predict_model_output(self, state):

        q_values = super().predict_model_output(state)
        action_mask = state.get("action_mask", None)
        masked_q_values = self._mask_q_values(q_values, action_mask)

        return masked_q_values  

    @requires_input_process
    def random_prediction(self, state):
        action_mask = state["action_mask"]
        return self.output_action_shape.sample(action_mask)