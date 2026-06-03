from abc import abstractmethod

from automarl.component import ParameterSignature, requires_input_process

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.utils.shapes_util import double_final_size
from automarl.core.localizations import get_component_by_localization
from automarl.components.rl.environment.environment_components import EnvironmentComponent
import torch

import random

from automarl.components.ml.models.model_components import ModelComponent

from automarl.components.rl.policy.policy import Policy

class StochasticPolicy(Policy):
    '''
    A policy wich selects actions based on distributions
    The model output is used to compute a distribution for the actions

    A stochastic policy has 4 processing moments:
        Computing the model output
        Computing the model distribution
        Computing a value which is directly tied to the action (essentially the action value without being normalized)
        Computing the action value

    The kwargs are for any extra data
    '''
        
    parameters_signature = {}   
    
    def _process_input_internal(self):
        super()._process_input_internal()
        
        
        
    
    @requires_input_process
    def get_action_val_from_model_output(self, model_output, state):

        distribution = self.distribution_from_model_output(model_output, state)
        
        return self.sample_action_val_from_distribution(distribution, state)



    def prepare_action_val_for_distribution(self, action_val, distribution, state):
        """
        Normalize action_val into the shape/dtype expected by the distribution.

        Base implementation is generic:
        - move to the distribution device when possible
        - preserve shape by default

        Subclasses should override when a distribution requires a specific
        action encoding (e.g. categorical indices as int64).
        """

        if not torch.is_tensor(action_val):
            device = None
            if hasattr(distribution, "probs") and distribution.probs is not None:
                device = distribution.probs.device
            elif hasattr(distribution, "loc") and distribution.loc is not None:
                device = distribution.loc.device
            action_val = torch.as_tensor(action_val, device=device)
        else:
            if hasattr(distribution, "probs") and distribution.probs is not None:
                action_val = action_val.to(distribution.probs.device)
            elif hasattr(distribution, "loc") and distribution.loc is not None:
                action_val = action_val.to(distribution.loc.device)

        return action_val
    
    def reduce_log_prob(self, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Reduce per-event log probs into one log prob per batch item.

        For scalar-action distributions, log_prob is usually already [B].
        For multidimensional independent actions, log_prob is often [B, A]
        and should be summed across the last dimension.
        """
        if log_prob.dim() > 1:
            return log_prob.sum(dim=-1, keepdim=True)
        return log_prob.unsqueeze(-1)
    
    
    @abstractmethod
    def distribution_from_model_output(self, model_output, state) -> torch.Tensor:
        pass
    
    
    def sample_action_val_from_distribution(self, distribution : torch.distributions, state):
        return distribution.sample()
    
    def log_probability_of_action_val(self, distribution, action_val, state):
        action_val = self.prepare_action_val_for_distribution(action_val, distribution, state)
        log_prob = distribution.log_prob(action_val)
        return self.reduce_log_prob(log_prob)    
    
    def predict_action_val_from_model_output_with_log(self, model_output, state):
                
        dist = self.distribution_from_model_output(model_output, state)
        
        action_val = self.sample_action_val_from_distribution(dist, state)
                
        log_prob = self.log_probability_of_action_val(dist, action_val, state)

        return action_val, log_prob    

    
    def predict_action_val_with_log(self, state):
        
        model_output = self.predict_model_output(state)    
                
        return self.predict_action_val_from_model_output_with_log(model_output, state)
    


class CategoricalStochasticPolicy(StochasticPolicy):
    '''
    A policy wich selects actions given on categorical probabilities
    '''
        
    parameters_signature = {
    }   
    
    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
    
    def probabilities_from_logits(self, logits, state) -> torch.Tensor:

        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    def prepare_action_val_for_distribution(self, action_val, distribution, state):
        """
        Categorical expects integer class indices.

        Supported input shapes:
        - scalar
        - [B]
        - [B, 1]  -> squeezed to [B]

        This keeps the learner generic while making categorical policies robust
        to stored action tensors that come in as column vectors.
        """
        action_val = super().prepare_action_val_for_distribution(action_val, distribution, state)

        if action_val.dtype != torch.long:
            action_val = action_val.long()

        if action_val.dim() > 0 and action_val.shape[-1] == 1:
            action_val = action_val.squeeze(-1)

        return action_val
    
    def reduce_log_prob(self, log_prob: torch.Tensor) -> torch.Tensor:
        if log_prob.dim() == 0:
            return log_prob.unsqueeze(0)
        
        if log_prob.dim() == 1:
            return log_prob.unsqueeze(-1)
        
        raise RuntimeError(f"Log prob had wrong shape: {log_prob.shape}")
 
    def distribution_from_model_output(self, model_output, state) -> torch.Tensor:
        
        probabilities = self.probabilities_from_logits(model_output, state)
    
        return torch.distributions.Categorical(probs=probabilities)
        

class MaskedCategoricalStochasticPolicy(CategoricalStochasticPolicy):
    '''
    A categorical stochastic policy that supports action masking.

    Expected state format:
    {
        "observation": <tensor-like model input>,
        "action_mask": <tensor-like mask of legal actions, optional>
    }

    The mask should have the same final dimension as the action logits.
    Valid entries are interpreted as:
    - bool: True = legal, False = illegal
    - numeric: > 0 = legal, <= 0 = illegal
    '''
    
    parameters_signature = {}

    INVALID_LOGIT = -1e9

    def _process_input_internal(self):
        super()._process_input_internal()


    def _normalize_action_mask(self, action_mask, logits):
        '''
        Converts action_mask to boolean mask on same device as logits.
        '''

        if action_mask is None:
            return None

        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, device=logits.device)

        action_mask = action_mask.to(device=logits.device)

        if action_mask.dtype != torch.bool:
            action_mask = action_mask > 0

        return action_mask

    def _mask_logits(self, logits, action_mask):
        '''
        Applies invalid-action masking to logits.
        '''

        if action_mask is None:
            return logits

        action_mask = self._normalize_action_mask(action_mask, logits)

        return logits.masked_fill(~action_mask, self.INVALID_LOGIT)

    def probabilities_from_logits(self, logits, state) -> torch.Tensor:
        masked_logits = self._mask_logits(logits, state["action_mask"])
        return torch.softmax(masked_logits, dim=-1)



        

class NormalStochasticPolicy(StochasticPolicy):
    '''
    A policy wich selects actions given on normal distributions for each output
    This means that the shape of the model is mean and standard deviation for each action value it can compute
    '''

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def _compute_model_output_shape(self):
        
        self.lg.writeLine(f"Action output shape is {self.output_action_shape}")
        # model outputs mean and std for each dimension
        self.model_output_shape = double_final_size(self.output_action_shape)

        self.lg.writeLine(f"Normal policy model output shape is {self.model_output_shape}")


    def distribution_from_model_output(self, model_output, state):

        '''
        Converts model output into a Normal distribution.
        Model output is assumed to contain:
            [mean_1 ... mean_A , log_std_1 ... log_std_A]
        '''

        action_dim = model_output.shape[-1] // 2

        mean = model_output[..., :action_dim]
        log_std = model_output[..., action_dim:]

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # convert log_std → std
        std = torch.exp(log_std)

        return torch.distributions.Normal(mean, std)
    


class ConstrainedNormalStochasticPolicy(NormalStochasticPolicy):
    '''
    A policy wich selects actions given on normal distributions for each output
    This means that the shape of the model is mean and standard deviation for each action value it can compute

    It passes through a tanh activation function
    '''

    EPS = 1e-6

    def _compute_model_output_shape(self):
        
        super()._compute_model_output_shape()

        self._get_bounds_from_shape()

        self.action_range = self.max_action_value - self.min_action_value

        self.lg.writeLine(f"Bound of actions for policies is [{self.min_action_value}, {self.max_action_value}] with a range of {self.action_range}")

    
    def _get_bounds_from_shape(self):

        if hasattr(self.output_action_shape, "low"):
            self.lg.writeLine(f"Output shape has lower bound: {self.output_action_shape.low}")

        if hasattr(self.output_action_shape, "high"):
            self.lg.writeLine(f"Output shape has upper bound: {self.output_action_shape.high}")

        # gym-style Box bounds
        self.min_action_value = torch.as_tensor(
            self.output_action_shape.low,
            device=self.device
        )

        self.max_action_value = torch.as_tensor(
            self.output_action_shape.high,
            device=self.device
        )
    
    def get_action_from_action_val(self, action_val):
        return self.min_action_value + (self.action_range) * 0.5 * (torch.tanh(action_val) + 1.0) 
