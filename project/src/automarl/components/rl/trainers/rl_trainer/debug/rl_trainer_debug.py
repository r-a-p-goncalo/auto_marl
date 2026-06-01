#from automarl.components.rl.trainers.debug.rl_trainer_debug import RLTrainerDebug
from automarl.components.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from automarl.core.input_management import ParameterSignature


class RLTrainerDebug(RLTrainerComponentParallel):

    is_debug_schema = True

    parameters_signature = {
        "plot_learning_graph" : ParameterSignature(default_value=False, ignore_at_serialization=True)
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self._plot_learning_graph = self.get_input_value("plot_learning_graph")

        if self._plot_learning_graph:

            plt.ion()  # turn on interactive mode

            self.fig, self.ax = plt.subplots(figsize=(6,4))


    def run_episode_step_for_agent_name(self, i_episode, agent_name):

        done, truncated = super().run_episode_step_for_agent_name( i_episode, agent_name)

        self.lg.writeLine(f"Doing episode step in episode {i_episode} for agent {agent_name} was over: {done}", file="observations.txt", use_time_stamp=False)
                        
        return done, truncated
    
    def run_single_episode(self, i_episode):
                        
        super().run_single_episode(i_episode)

        if self._plot_learning_graph:

            clear_output(wait=True)
    
            self.ax.clear()
    
            self.get_results_logger().plot_confidence_interval(x_axis='episode', y_column='episode_reward',show_std=False, to_show=False, ax=self.ax)
            self.get_results_logger().plot_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False, y_label='linear', ax=self.ax)
    
            self.ax.set_title(f"Training progress (update {i_episode})")
            display(self.fig)

