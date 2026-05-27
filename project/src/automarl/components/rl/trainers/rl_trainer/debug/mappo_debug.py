

from automarl.components.rl.trainers.rl_trainer.rl_trainer_mappo import RLTrainerMAPPO
from automarl.core.input_management import ParameterSignature
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.utils.str_utils import generate_str_fixed_chars


class RLTrainerMAPPODebug(RLTrainerMAPPO, ComponentDebug):
    is_debug_schema = True

    parameters_signature = {
        "note_agent_memory_coordination": ParameterSignature(default_value=True),
    }

    def _process_input_internal(self):

        super()._process_input_internal()

        self.lg.writeLine("Starting MAPPO debug trainer setup...")


        self.note_agent_memory_coordination = self.get_input_value("note_agent_memory_coordination")

        self.lg.writeLine("Finished MAPPO debug trainer setup\n")

    def _push_shared_transition(
        self,
        prev_whole_state,
        next_whole_state,
        observations,
        rewards,
        actions,
        dones, truncations, 
        agent_names
    ):
        if self.note_agent_memory_coordination:
            prev_obs = prev_whole_state.get("observation", None)
            next_obs = next_whole_state.get("observation", None)

            self.lg.writeLine(
                f"{self.values['episodes_done']}, {self.values['episode_steps']}: {generate_str_fixed_chars(prev_obs, 50)} -> {generate_str_fixed_chars(next_obs, 50)}, {dones} with {rewards} reward\n",
                file="mappo_shared_transitions.txt",
                use_time_stamp=False,
            )

            for agent_name in rewards.keys():
                self.lg.writeLine(
                    f"        {agent_name}: {generate_str_fixed_chars(observations.get(agent_name, None), 50)} + {actions.get(agent_name, None)} -> {generate_str_fixed_chars(observations.get(agent_name, None), 50)}, {dones.get(agent_name, None)} with {rewards.get(agent_name, None)} reward",
                    file="mappo_shared_transitions.txt",
                    use_time_stamp=False,
                )

        return super()._push_shared_transition(
            prev_whole_state,
            next_whole_state,
            observations,
            rewards,
            actions,
            dones, truncations,
            agent_names
        )
