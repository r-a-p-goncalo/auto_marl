

from automarl.components.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automarl.component import ParameterSignature

import random
import math
import torch

from automarl.components.rl.agent.agent_components import AgentSchema

class EpsilonGreedyStrategy(ExplorationStrategySchema):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { "epsilon_end" : ParameterSignature(default_value=0.025,
                                custom_dict={
                                    "hyperparameter_suggestion" : ("float", {"low" : 0.001, "high" : 0.15})
                                }),
                       "epsilon_start" : ParameterSignature(default_value=1.0),
                       "epsilon_decay" : ParameterSignature(default_value=0.01),
                       "training_context" : ParameterSignature(
                           validity_verificator= lambda ctx : all(key in ctx.values.keys() for key in ["total_steps", "episodes_done"]))} #training context is a dictionary where we'll be able to get outside data   
    
    exposed_values = {"n_random" : 0, "n_greedy" : 0}
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
        self.EPS_END = self.get_input_value("epsilon_end")                
        self.EPS_START = self.get_input_value("epsilon_start")
        self.EPS_DECAY = self.get_input_value("epsilon_decay")
        self.training_context = self.get_input_value("training_context").values
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------

    def should_select_random_action(self):

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.training_context["episodes_done"] / self.EPS_DECAY)

        sample = random.random()

        return sample < eps_threshold

    
    def select_action(self, agent : AgentSchema,  state):
                
        super().select_action(agent, state)
        
        if self.should_select_random_action():

            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict(state) 
            

        else:
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict(state)
        
    
    def select_action_with_memory(self, agent : AgentSchema):

        super().select_action_with_memory(agent)
        
        #in the case we use our policy net to predict our next action    
        if self.should_select_random_action():
            
            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict(agent.get_current_state_in_memory())      
        else:
   
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict_with_memory()


class EpsilonGreedyLinearStrategy(EpsilonGreedyStrategy):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { }
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
    
    # EXPOSED METHOD --------------------------------------------------------------------------

    def should_select_random_action(self):

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * max(0, 1 - self.EPS_DECAY * self.training_context["episodes_done"])

        sample = random.random()

        return sample < eps_threshold
    


class EpsilonGreedyStepDecayStrategy(EpsilonGreedyStrategy):
    '''
    Epsilon-greedy strategy where epsilon is multiplied by epsilon_decay
    every n_steps_for_decay environment steps.

    Example:
        epsilon_start = 1.0
        epsilon_decay = 0.9
        n_steps_for_decay = 1000

    Then:
        steps 0-999     -> epsilon = 1.0
        steps 1000-1999 -> epsilon = 0.9
        steps 2000-2999 -> epsilon = 0.81
        ...

    epsilon is always clipped below by epsilon_end.
    '''

    parameters_signature = {
        "n_steps_for_decay": ParameterSignature(
            default_value=1000,
            custom_dict={
                "hyperparameter_suggestion": ("int", {"low": 100, "high": 10000, "step": 100})
            }
        )
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.n_steps_for_decay = self.get_input_value("n_steps_for_decay")

        if self.n_steps_for_decay <= 0:
            raise ValueError(f"n_steps_for_decay must be > 0, got {self.n_steps_for_decay}")

        if not (0 < self.EPS_DECAY <= 1):
            raise ValueError(
                f"For step decay, epsilon_decay should be in (0, 1], got {self.EPS_DECAY}"
            )

    def get_current_epsilon(self):
        total_steps = self.training_context["total_steps"]

        n_decays_applied = total_steps // self.n_steps_for_decay

        current_epsilon = self.EPS_START * (self.EPS_DECAY ** n_decays_applied)

        return max(self.EPS_END, current_epsilon)

    def should_select_random_action(self):
        eps_threshold = self.get_current_epsilon()

        sample = random.random()

        return sample < eps_threshold