

from automarl.components.rl.environment.environment_components import EnvironmentComponent, EnvironmentSampler, sampled_environment_fun
from automarl.component import requires_input_process

from abc import abstractmethod

from automarl.components.rl.environment.environment_type import EnvironmentType

class AECEnvironmentComponent(EnvironmentComponent):
    
    parameters_signature =  {} 
    
    def get_environment_type(self) -> EnvironmentType:
        return EnvironmentType.AEC
        
    @requires_input_process   
    def observe(self, agent_name : str):
        '''Returns the observation for the given agent'''
        pass
        
    @requires_input_process   
    @abstractmethod
    def last(self,):
        '''Gets last transitions / observations done'''
        pass
    
    @requires_input_process
    @abstractmethod   
    def agent_iter(self):
        '''Returns an iterator for the active agents'''
        pass
    
    @requires_input_process
    @abstractmethod    
    def step(self, action):
        '''Makes a step in the environment for the currently active agent and the given action'''
        pass
        
    @requires_input_process    
    @abstractmethod
    def rewards(self):
        raise NotImplementedError()    
    

    

class AECEnvironmentSampler(EnvironmentSampler, AECEnvironmentComponent):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.environment_input = self.get_input_value("environment_input")
                
        
    def sample(self) -> AECEnvironmentComponent:
        raise NotImplementedError()
        
    @sampled_environment_fun    
    def reset(self):
        pass
            
    @sampled_environment_fun   
    def observe(self, *args):
        pass
        
    @sampled_environment_fun   
    def agents(self):
        pass
    
    @sampled_environment_fun
    def get_agent_action_space(self, agent):
        pass
    
    @sampled_environment_fun
    def get_agent_state_space(self, agent):
        pass

    
    @sampled_environment_fun   
    def last(self):
        pass
    
    @sampled_environment_fun   
    def agent_iter(self):
        pass
    
    @sampled_environment_fun    
    def step(self, *args):
        pass
        
    @sampled_environment_fun    
    def rewards(self):
        pass
    
    @sampled_environment_fun
    def render(self):
        pass
    
    @sampled_environment_fun
    def close(self):
        pass
    
    @sampled_environment_fun
    def get_env_name(self):
        pass
    
class AECEnvironmentCycler(AECEnvironmentSampler):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
