from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.core.input_management import ParameterSignature
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
from automarl.components.rl.agent.agent_components import AgentSchema
import torch
from automarl.components.rl.environment.parallel_environment import ParallelEnvironmentComponent


class RLAECPlayer(RLPlayer):

    parameters_signature = {
                                
                       }

    def _process_input_internal(self):

        super()._process_input_internal()

        self.env : AECEnvironmentComponent = self.env

        if not isinstance(self.env, AECEnvironmentComponent):
            raise Exception(f"RLParallelPlayer requires AECEnvironmentComponent, got {type(self.env)}")


    def _do_agent_step(self, agent_name):
        
        agent : AgentSchema = self.agents[agent_name]
        
        observation, reward, done, truncated, info = self.env.last()
        agent.update_state_memory(observation)

        if done or truncated:
            self.env.step(None)
        else:
            action = agent.policy_predict_with_memory()
            self.env.step(action)
                        
        self.values["episode_score"] = self.values["episode_score"] + reward
                      
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step

        self.values["agents_episode_score"][agent_name] += reward

        
        return reward, done, truncated


    def _run_episode(self):
        
        for agent_name in self.env.agent_iter():
            
            reward, done, truncated = self._do_agent_step(agent_name)

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
