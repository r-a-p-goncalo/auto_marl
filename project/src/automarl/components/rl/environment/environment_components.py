from types import FunctionType
from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.components.basic_components.sampler import Sampler
from abc import abstractmethod


def normalize_observation(raw_obs):
        
        if isinstance(raw_obs, dict):
            if "observation" not in raw_obs:
                raise ValueError("Observation dict must contain key 'observation'")
            return raw_obs

        return {
            "observation": raw_obs
        }

class EnvironmentComponent(Component):

    '''Represents any environment'''

    def __init__(self, input = None):
        super().__init__(input)


    @requires_input_process
    @abstractmethod
    def get_whole_state_shape(self):
        '''
        Returns the shape/space of the centralized whole-environment state
        used by a shared critic.
        '''
        pass

    @requires_input_process
    @abstractmethod
    def get_current_whole_state(self):
        '''
        Returns the current centralized whole-environment state.
        Framework contract:
        {
            "observation": <tensor/array>
        }
        '''
        pass
    
    
    @requires_input_process
    @abstractmethod
    def close(self):
        '''Closes an environment, telling it its execution is not being used'''
        pass

    @requires_input_process
    @abstractmethod
    def get_env_name(self):
        '''Gets the internal name of an environment'''
        pass

    @requires_input_process
    @abstractmethod    
    def reset(self):
        '''A soft reset of the environment, only to guarantee it is in its initial state'''
        pass
    
    @requires_input_process
    @abstractmethod    
    def total_reset(self):
        '''Resets all, including RNG state'''
        pass
    
    @requires_input_process
    @abstractmethod
    def agents(self):
        '''Returns all possible agents to exist on an environment'''
        pass    

    @requires_input_process
    @abstractmethod    
    def get_active_agents(self):
        '''Returns all the active agents'''
        pass


    @requires_input_process
    @abstractmethod
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        pass
    
    @requires_input_process
    @abstractmethod
    def get_agent_state_space(self, agent):
        '''returns the state space for the environment'''
        pass



    
def sampled_environment_fun(func : FunctionType):

    '''
    Returns a function that samples an environment if there is none yet and calls the passed function from the environment
    '''
        
    def wrapper(self : Sampler, *args, **kwargs):
        
        if self.sampled_environment == None: # samples environment if non existent
            self.sampled_environment = self.sample()

        func(self, *args, **kwargs) # evaluates the function on the sampler, probabily not needed
        return getattr(self.sampled_environment, func.__name__)(*args, **kwargs) # the function is actually evaluated in the environment

    return wrapper



class EnvironmentSampler(Sampler, EnvironmentComponent):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environment_input": ParameterSignature(default_value={}),
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.environment_input = self.get_input_value("environment_input")
                
    def sample(self) -> EnvironmentComponent:
        raise NotImplementedError()

    @sampled_environment_fun
    def close(self):
        raise NotImplementedError()
    
    @sampled_environment_fun
    def get_env_name(self):
        raise NotImplementedError()
    
    
class EnvironmentCycler(EnvironmentSampler):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environments": ComponentListParameterSignature(),
        "generate_name" : ParameterSignature(default_value=False)
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _process_input_internal(self):
        super()._process_input_internal()

        self.environments : EnvironmentComponent = self.get_input_value("environments") 
        self.generate_name = self.get_input_value("generate_name")
        self.next_index = 0

        if self.generate_name:
            for index in range(len(self.environments)):

                env = self.environments[index]
                generated_name = f"{env.name}_{index}"
                env.pass_input({"name" : generated_name})


    
    @requires_input_process
    def sample(self) -> EnvironmentComponent:

        to_return = self.environments[self.next_index]
        self.next_index = ( self.next_index + 1 ) % len(self.environments)

        return to_return


    
     
    

     