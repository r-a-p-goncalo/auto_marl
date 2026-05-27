
from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug


class RLPlayerDebug(RLPlayer, ComponentDebug):

    is_debug_schema = True

    def _do_agent_step(self, agent_name):

        self.lg.writeLine(f"Agent {agent_name} will do a step", file="agent_steps.txt")

        self.lg.writeLine(f"        Episode score {self.values['episode_score']}, Episode Step {self.values['episode_steps']}, {agent_name} reward {self.values['agents_episode_score'][agent_name]}", file="agent_steps.txt")
        
        to_return = super()._do_agent_step(agent_name)
        self.lg.writeLine(f"            --->", file="agent_steps.txt")

        self.lg.writeLine(f"        Episode score {self.values['episode_score']}, Episode Step {self.values['episode_steps']}, {agent_name} reward {self.values['agents_episode_score'][agent_name]}", file="agent_steps.txt")



        return to_return
    
    def _setup_episode(self):

        self.lg.writeLine(f"Episode starting with values: {self.values}\n", file="agent_steps.txt")

        super()._setup_episode()



    def _end_episode(self):

        super()._end_episode()
        
        self.lg.writeLine(f"Finished episode {self.values['episodes_done']} with results: {self.values}\n", file="agent_steps.txt")
    