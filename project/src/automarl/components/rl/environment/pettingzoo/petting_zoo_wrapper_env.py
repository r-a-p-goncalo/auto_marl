

from automarl.components.rl.environment.environment_components import EnvironmentComponent
from automarl.component import requires_input_process
import gymnasium
from automarl.utils.shapes_util import clone_shape
from automarl.core.input_management import ParameterSignature


class PettingZooWrapper(EnvironmentComponent):

    parameters_signature = { 
        "environment_input": ParameterSignature(mandatory=False)

    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.environment_input = self.get_input_value("environment_input")
        if self.environment_input is None:
            self.environment_input = {}

    @requires_input_process
    def get_current_whole_state(self):
        if hasattr(self.env, "state"):

            if callable(self.env.state):
                state = self.env.state()
            else:
                state = self.env.state

            return {"observation": state}
        
        else:
            raise NotImplementedError(f"Do not have a observation concetation to build whole state")
        
    @requires_input_process
    def get_whole_state_shape(self):
        """
        Returns the centralized state space for MAPPO/shared critic usage.
        Prefers env.state_space() when available, otherwise concatenates all
        agent observation spaces.
        """
        if hasattr(self.env, "state_space"):

            if callable(self.env.state_space):
                state_space = self.env.state_space()
            else:
                state_space = self.env.state_space

            if isinstance(state_space, gymnasium.spaces.Dict):
                if "observation" not in state_space.spaces:
                    raise ValueError(
                        f"{self.name}: env.state_space() Dict must contain key 'observation'"
                    )
                return {
                    key: clone_shape(subspace)
                    for key, subspace in state_space.spaces.items()
                }

            return {
                "observation": clone_shape(state_space)
            }
                
        else:
            state = self.get_current_whole_state()

            state = self.get_current_whole_state()

            if not isinstance(state, dict):
                raise ValueError(
                    f"{self.name}: get_current_whole_state() must return a dict, got {type(state)}"
                )

            if "observation" not in state:
                raise ValueError(
                    f"{self.name}: get_current_whole_state() must contain key 'observation'"
                )

            return {
                key: value.shape
                for key, value in state.items()
            }

        
    @requires_input_process
    def agents(self):
        return self.env.possible_agents
        
    
    @requires_input_process    
    def get_active_agents(self):
        '''Returns all the active agents'''
        return self.env.agents
