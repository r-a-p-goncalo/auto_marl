

from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent

from automarl.components.rl.environment.gymnasium.aec_gymnasium_env import AECGymnasiumEnvironmentWrapper
from automarl.components.rl.environment.environment_components import normalize_observation
from automarl.utils.shapes_util import clone_shape
from automarl.components.rl.environment.pettingzoo.petting_zoo_wrapper_env import PettingZooWrapper
import torch
import gymnasium

from automarl.components.rl.environment.pettingzoo import ParallelEnv


# TODO: This should probably extend Gymnasium
class AECPettingZooEnvironmentWrapper(AECGymnasiumEnvironmentWrapper, PettingZooWrapper):
        
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       "environment" : ParameterSignature(default_value="cooperative_pong"),
                       }    
    
    

    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
    
    def _setup_environment(self):

        self.env = self.get_input_value("environment")
                
        if isinstance(self.env , str):
            self._load_environment(self.env )
            
        elif isinstance(self.env , ParallelEnv):
            self.env : ParallelEnv = self.env 
            
        else:
            raise Exception("No valid environment or environment name passed to PettingZoo Wrapper")
        
        
    def _load_environment(self, environment_name : str):
        
        if environment_name == "cooperative_pong":
            
            from pettingzoo.butterfly import cooperative_pong_v5
            self.env : ParallelEnv = cooperative_pong_v5.env(render_mode=self.render_mode)

        elif environment_name == "connect_four":

            from pettingzoo.classic import connect_four_v3
            self.env = connect_four_v3.env(render_mode=self.render_mode)
  
            
        else:
            raise Exception(f"{self.name}: No valid petting zoo environment specified")
        
    
        
    def observe(self, *args):
        return normalize_observation(self.env.observe(*args))
    
    
    @requires_input_process
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        return self.env.action_space(agent)
    

    @requires_input_process
    def get_agent_state_space(self, agent):
        '''
        Returns the state space in the framework contract:
        {
            "observation": <space>,
            ...metadata spaces...
        }
        '''

        obs_space = self.env.observation_space(agent)

        # PettingZoo classic envs like connect_four usually expose a Dict space
        if isinstance(obs_space, gymnasium.spaces.Dict):
            state_space = {}

            for key, subspace in obs_space.spaces.items():
                state_space[key] = clone_shape(subspace)

            if "observation" not in state_space:
                raise ValueError(
                    f"{self.name}: observation space for agent '{agent}' must contain key 'observation'"
                )

            return state_space

        # If env gives a plain space, normalize to framework format
        return {
            "observation": clone_shape(obs_space)
        }
    
    
    def last(self):
        observation, reward, termination, truncation, info = self.env.last()
        
        #returns state, reward, done, info
        return normalize_observation(observation), reward, termination, truncation, info
    
    def agent_iter(self):
        return self.env.agent_iter()

    def _process_action_of_agent(self, action):
            
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu()

                if action.numel() == 1:
                    return action.item()
                
                return action.numpy()

            return action
    
    def step(self, action):
        return self.env.step(self._process_action_of_agent(action))
    
    def rewards(self):
        return self.env.rewards    

    def render(self):
        self.env.render()
    

    def close(self):
        self.env.close()
        
    def reset(self):
        self.env.reset()
    
    def total_reset(self):
        self.env.reset(seed=self.seed)
