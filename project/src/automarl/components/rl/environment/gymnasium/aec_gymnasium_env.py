import itertools
from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.components.basic_components.state_management import StatefulComponent
from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent


from automarl.components.rl.environment.environment_components import EnvironmentSampler, normalize_observation
from automarl.utils.shapes_util import torch_state_shape_from_space

import gymnasium as gym
import torch
import gymnasium
from automarl.utils.shapes_util import clone_shape


class AECGymnasiumEnvironmentWrapper(AECEnvironmentComponent, SeededComponent, StatefulComponent):
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environment": ParameterSignature(default_value="CartPole-v1"),
        "render_mode": ParameterSignature(default_value="rgb_array", validity_verificator=lambda x: x in ["rgb_array", "human", "none"])
    }
        
        

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.last_observation = None
        self.last_reward = 0
        self.last_done = False
        self.last_truncation = False
        self.last_info = {}
        
        self.reset_info = {}

        self.render_mode = self.get_input_value("render_mode")

        self._setup_environment()

        self.total_reset()   


        

    def _setup_environment(self):

        '''Loads the actual Gym environment implementation'''

        self.env = self.get_input_value("environment")
        
        if isinstance(self.env, str):
            self._load_environment(self.env)
        
        elif isinstance(self.env, gym.Env):
            self.env: gym.Env = self.env
        
        else:
            raise Exception("No valid Gymnasium environment or environment name passed.")


    def _load_environment(self, environment_name: str):
        
        try:
            self.env: gym.Env = gym.make(environment_name, render_mode=self.render_mode)
        
        except Exception as e:
            raise Exception(f"{self.name}: Failed to load gym environment '{environment_name}': {str(e)}")
        
        
    def get_env_name(self):
        try:
            return self.env.spec.id
        except:
            return type(self.env)


    @requires_input_process
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        return self.env.action_space
    
    @requires_input_process
    def get_agent_state_space(self, agent):
        obs_space = self.env.observation_space

        if isinstance(obs_space, gymnasium.spaces.Dict):
            return {
                key: clone_shape(subspace)
                for key, subspace in obs_space.spaces.items()
            }

        return {
            "observation": clone_shape(obs_space)
        }

    
    def reset(self):

        '''Resets the environment, with an optinal seed'''        


        observation, info = self.env.reset()
        
        observation = normalize_observation(observation)

        self.reset_info = info
        
        self.last_observation = observation
            
        return observation

    def total_reset(self):

        '''Resets the environment, with an optinal seed'''        


        observation, info = self.env.reset(seed=self.seed)
        
        observation = normalize_observation(observation)
        self.reset_info = info
        
        self.last_observation = observation
            
        return observation
    

    def last(self):
        return self.last_observation, self.last_reward, self.last_done, self.last_truncation, self.last_info

    def agents(self):
        return ["agent"]

    def agent_iter(self):
        return itertools.repeat("agent") if not self.last_done and not self.last_truncation else iter([])
    


    def step(self, action):
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        obs, reward, terminated, truncated, info = self.env.step(action)       

        obs = normalize_observation(obs)
                
        self.last_observation = obs
        self.last_reward = reward
        self.last_done = terminated
        self.last_truncation = truncated
        self.last_info = info
                
        return obs, reward, terminated, truncated, info


    def render(self):
        return self.env.render()


    def close(self):
        return self.env.close()


    def observation_space(self):
        return self.env.observation_space

    
    def observe(self, *args):
        return self.last_observation
    
    def get_env_info(self):
        return self.reset_info
    

    # STATE MANAGEMENT -------------------------------------------------------------------
    
    def _on_unload(self):
        
        super()._on_unload()
        
        self.env.close()


class GymnasiumEnvironmentWrapperSampler(EnvironmentSampler):
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
    }

    @requires_input_process
    def sample(self) -> AECEnvironmentComponent:
        return AECGymnasiumEnvironmentWrapper(self.environment_input)