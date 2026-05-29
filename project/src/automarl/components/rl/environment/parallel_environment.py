
from automarl.components.rl.environment.environment_components import EnvironmentComponent
from automarl.component import requires_input_process

from abc import abstractmethod

from automarl.components.rl.environment.environment_type import EnvironmentType

class ParallelEnvironmentComponent(EnvironmentComponent):
    
    parameters_signature =  {} 
        

    @abstractmethod
    def step(self, actions):
        """
        Args:
            actions: {agent: action}
        Returns:
            next_obs_dict, rewards, terminations, truncations, infos
        """
        pass

    def get_environment_type(self) -> EnvironmentType:
        return EnvironmentType.PARALLEL
        