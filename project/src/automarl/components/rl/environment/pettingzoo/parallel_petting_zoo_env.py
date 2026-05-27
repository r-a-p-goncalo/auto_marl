from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.component import Component, ParameterSignature, requires_input_process

from automarl.components.fundamentals.translator.translator import Translator
from automarl.components.rl.environment.parallel_environment import ParallelEnvironmentComponent
from automarl.components.rl.environment.environment_components import normalize_observation
from automarl.components.rl.environment.pettingzoo.petting_zoo_wrapper_env import PettingZooWrapper
import torch
from automarl.utils.shapes_util import clone_shape


from pettingzoo import ParallelEnv
import gymnasium

import numpy as np

class PettingZooEnvironmentWrapperParallel(ParallelEnvironmentComponent, SeededComponent, PettingZooWrapper):

    parameters_signature = { 
        "environment": ParameterSignature(default_value="cooperative_pong"),
        "render_mode": ParameterSignature(default_value="rgb_array", validity_verificator=lambda x: x in ["rgb_array", "human"]),
        "device": ParameterSignature(ignore_at_serialization=True)
    }


    def _process_input_internal(self):
        super()._process_input_internal()

        self.render_mode = self.get_input_value("render_mode")
        self.device = self.get_input_value("device")

        self._setup_environment()


    def _setup_environment(self):
        env_input = self.get_input_value("environment")

        if isinstance(env_input, str):
            self._load_environment(env_input)

        elif isinstance(env_input, ParallelEnv):
            self.env: ParallelEnv = env_input

        else:
            raise Exception("Invalid environment for PettingZooEnvironmentWrapper")


    def _load_environment(self, environment_name: str):

        if environment_name == "cooperative_pong":
            from pettingzoo.butterfly import cooperative_pong_v5
            self.env: ParallelEnv = cooperative_pong_v5.parallel_env(render_mode=self.render_mode, **self.environment_input)

        elif environment_name == "multiwalker":
            from pettingzoo.sisl import multiwalker_v9
            self.env: ParallelEnv = multiwalker_v9.parallel_env(render_mode=self.render_mode, **self.environment_input)

        elif environment_name == 'simple_spread':
            from pettingzoo.mpe import simple_spread_v3
            self.env: ParallelEnv = simple_spread_v3.parallel_env(render_mode=self.render_mode, **self.environment_input)
        else:
            raise Exception(f"No valid PettingZoo environment named '{environment_name}'")
        
        self.env_name = environment_name


    def observe(self, agent):
        """
        Parallel API: observations are not retrieved with env.observe(agent)
        but from the dictionary returned by reset() and step().
        This method simply accesses the last stored obs dict.
        """
        return self._last_obs[agent]


    @requires_input_process
    def get_agent_action_space(self, agent):
        return self.env.action_space(agent)


    @requires_input_process
    def get_agent_state_space(self, agent):
        obs_space = self.env.observation_space(agent)

        if isinstance(obs_space, gymnasium.spaces.Dict):
            return {
                key: clone_shape(subspace)
                for key, subspace in obs_space.spaces.items()
            }

        return {
            "observation": clone_shape(obs_space)
        }
    

    def _normalize_parallel_observations(self, observations: dict):
        return {
            agent_name: normalize_observation(agent_obs)
            for agent_name, agent_obs in observations.items()
        }


    def reset(self):
        """
        Returns:
            obs_dict[agent] = observation
            info_dict[agent] = info
        """
        obs, info = self.env.reset()
        self._last_obs = self._normalize_parallel_observations(obs)  # store so .observe(agent) can work
        self.reset_info = info
        return self._normalize_parallel_observations(obs)
    
    def total_reset(self):
        """
        Returns:
            obs_dict[agent] = observation
            info_dict[agent] = info
        """
        obs, info = self.env.reset(seed=self.seed)
        self._last_obs = self._normalize_parallel_observations(obs)  # store so .observe(agent) can work
        self.reset_info = info
        return self._normalize_parallel_observations(obs)
    
    def _process_action_of_agent(self, agent, action):
            
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()


            return action

    def _process_actions(self, actions: dict):
        """
        Clips actions to the environment bounds when available.
        Returns a new dict.
        """
        processed = {}

        for agent, action in actions.items():
            processed[agent] = self._process_action_of_agent(agent, action)

        return processed

    def step(self, actions):
        """
        Args:
            actions: {agent: action}
        Returns:
            next_obs_dict, rewards, terminations, truncations, infos
        """
        actions = self._process_actions(actions)

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self._last_obs = self._normalize_parallel_observations(next_obs)
        return self._normalize_parallel_observations(next_obs), rewards, terminations, truncations, infos


    # For debugging or compatibility
    def rewards(self):
        return self.env.rewards

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @requires_input_process
    def get_env_name(self):
        return self.env_name
