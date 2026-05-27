from automarl.component import Component, ParameterSignature, requires_input_process
import torch
import numpy as nn

from abc import abstractmethod

from automarl.components.rl.agent.agent_components import AgentSchema


class ExplorationStrategySchema(Component):
    
    parameters_signature =  {
        "training_context" : ParameterSignature(possible_types=[Component])
        } 

    
    @abstractmethod
    @requires_input_process
    def select_action(self, agent : AgentSchema, state):
        
        '''
            Selects an action based on the agent's state (using things like its policy) and this exploration strategy
            
            Args:
                state is the current state as readable by the agent
                
            Returns:
                The index of the action selected
        '''
        
        pass
    

    @abstractmethod
    @requires_input_process
    def select_action_with_memory(self, agent : AgentSchema):
        pass
    