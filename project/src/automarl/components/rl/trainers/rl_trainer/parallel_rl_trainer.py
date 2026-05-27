
from automarl.components.rl.trainers.rl_trainer_component import RLTrainerComponent

from automarl.components.rl.environment.parallel_environment import ParallelEnvironmentComponent
from automarlcore.input_management import ParameterSignature
import torch

class RLTrainerComponentParallel(RLTrainerComponent):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {

        "use_average_reward" : ParameterSignature(default_value = True)
                                
                       }

    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()

        if not isinstance(self.env, ParallelEnvironmentComponent):
            raise Exception(f"Parallel RL training requires parallel component")
        
        self.env : ParallelEnvironmentComponent = self.env

        self.lg.writeLine(f"Setting up RL trainer Component Parallel")
        
        self.chosen_actions = {}
        
        self.use_average_reward = self.get_input_value("use_average_reward")

        self.lg.writeLine(f"Finished setting up RL trainer component parallel\n")
                                                                                     

    
    def choose_actions_for_agents(self, agents_names : list[str], i_episode):

        self.chosen_actions.clear()

        for agent_name in agents_names:

            agent_trainer = self.agents_trainers[agent_name]
        
            with torch.no_grad():                
                action = agent_trainer.select_action_with_memory() # decides the next action to take (can be random)

            self.chosen_actions[agent_name] = action.squeeze(0)

        return self.chosen_actions
    


    def process_env_step_for_agents(self, i_episode, agents_names : list[str], actions, observations, rewards, terminations, truncations):

        done = True

        for agent_name in agents_names:
            
            agent_trainer = self.agents_trainers[agent_name]

            action = actions[agent_name]
            observation = observations[agent_name]
            reward = rewards[agent_name]
            termination = terminations[agent_name]
            truncation = truncations[agent_name]

            agent_trainer.do_after_training_step(i_episode=i_episode, 
                                                 action=action, 
                                                 next_state=observation,
                                                   reward=reward, 
                                                   done=termination, 
                                                   truncated=truncation)

            done = done and termination

        return done

    def _aggregated_reward(self, rewards):
            
        reward = rewards.values()

        if self.use_average_reward:
            reward = sum(reward) / len(reward)
            
        else:
            reward = sum(reward)

        return reward
    
    def run_single_episode(self, i_episode):
                        
        self.setup_single_episode(i_episode)

        while True: # this runs a step of the episode

            agent_names = [*self.env.get_active_agents()]

            if len(agent_names) == 0:
                break

            actions = self.choose_actions_for_agents(agent_names, i_episode)

            observations, rewards, terminations, truncations, infos = self.env.step(actions)

            done = self.process_env_step_for_agents(i_episode, agent_names, actions, observations, rewards, terminations, truncations)

            reward = self._aggregated_reward(rewards)

            self.after_environment_step(reward)

            if done or self._check_if_to_end_episode():
                break
    

        for agent_in_training in self.agents_trainers.values():
            agent_in_training.end_episode(
                env=self.env,
                i_episode=i_episode
            ) 
        
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        self.values["episodes_done_in_session"] = self.values["episodes_done_in_session"] + 1
        
        self.calculate_and_log_results()
            
        
                   

                        