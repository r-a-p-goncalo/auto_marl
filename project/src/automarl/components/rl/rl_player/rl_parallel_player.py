from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.core.input_management import ParameterSignature
import torch
from automarl.components.rl.environment.parallel_environment import ParallelEnvironmentComponent


class RLParallelPlayer(RLPlayer):

    parameters_signature = {

        "use_average_reward" : ParameterSignature(default_value = True)
                                
                       }

    def _process_input_internal(self):
        """
        Extend RLPlayer's input processing, but ensure environment is ParallelEnvironmentComponent.
        """
        super()._process_input_internal()

        self.env : ParallelEnvironmentComponent = self.env

        if not isinstance(self.env, ParallelEnvironmentComponent):
            raise Exception(f"RLParallelPlayer requires ParallelEnvironmentComponent, got {type(self.env)}")

        self.use_average_reward = self.get_input_value("use_average_reward")


    def _do_agent_step_parallel(self):

        actions = {}

        active_agents = self.env.get_active_agents()

        with torch.no_grad():
            for agent_name in active_agents:

                agent = self.agents[agent_name]

                act = agent.policy_predict_with_memory()
                actions[agent_name] = act.squeeze(0)

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        done = True

        for agent_name in active_agents:

            agent_reward = rewards[agent_name]

            self.values["agents_episode_score"][agent_name] += agent_reward

            self.agents[agent_name].update_state_memory(observations[agent_name])

            done = done and (terminations[agent_name] or truncations[agent_name])

        return rewards, done



    def _run_episode(self):
        """
        Overrides RLPlayer.__run_episode()  
        Uses a single parallel step per loop rather than iterating over agents.
        """

        while True:

            rewards, done = self._do_agent_step_parallel()

            rewards = rewards.values()

            if self.use_average_reward:
                reward = sum(rewards) / len(rewards)
            
            else:
                reward = sum(rewards)

            self.values["episode_score"] += reward
            self.values["episode_steps"] += 1
            self.values["total_steps"] += 1

            if done:
                break
