

from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent


class SingleRLTrainer(RLTrainerComponent):


    def _do_environment_agent_interaction_loop(self, i_episode):

        done = False
        truncated = False

        while not (done or truncated):

            agent_name = self.env.agents()[0]
            trainer = self.agents_trainers[agent_name]

            # this method expects AEC behavior, and such this reward, done and truncated values are the previous ones for a single agent behavior
            reward, done, truncated = trainer.do_training_step(i_episode, self.env)

            _, reward, done, truncated, _ = self.env.last()

            print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")

            self.after_environment_step(reward)

            if self._check_if_to_end_episode():
                break