

from automarl.components.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug


class PettingZooEnvironmentWrapperParallelDebug(PettingZooEnvironmentWrapperParallel, ComponentDebug):

    is_debug_schema = True

    def _process_input_internal(self):
        super()._process_input_internal()


    def get_agent_action_space(self, agent):
        to_return = super().get_agent_action_space(agent)
        self.lg.writeLine(f"Getting agent {agent} action space: {to_return}")
        return to_return


    def get_agent_state_space(self, agent):
        to_return = super().get_agent_state_space(agent)
        self.lg.writeLine(f"Getting agent {agent} state space: {to_return}")
        return to_return
        
    
    def _process_actions(self, actions: dict):
        """
        Clips actions to the environment bounds when available.
        Returns a new dict.
        """
        
        processed = super()._process_actions(actions)

        self.lg.writeLine(f"Received actions to process: ")

        for agent in actions.keys():
            self.lg.writeLine(f"        {agent}: {actions[agent]} -> {processed[agent]}")

        self.lg.writeLine(f"")

        return processed

