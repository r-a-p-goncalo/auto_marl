

from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent


class AECRLTrainer(RLTrainerComponent):

    def run_single_episode(self, i_episode):

        self.setup_single_episode(i_episode)

        while True:

            for agent_name in self.env.agent_iter():

                trainer = self.agents_trainers[agent_name]

                reward, done, truncated = trainer.do_training_step(i_episode, self.env)

                self.after_environment_step(reward)

                if self._check_if_to_end_episode():
                    break

            # AEC env ends naturally when agent_iter becomes empty
            break

        self._finalize_episode(i_episode)

    def setup_single_episode(self, i_episode):
        self.env.reset()
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0

        for t in self.agents_trainers.values():
            t.setup_episode(self.env)

    def _finalize_episode(self, i_episode):
        for t in self.agents_trainers.values():
            t.end_episode(env=self.env, i_episode=i_episode)

        self.values["episodes_done"] += 1
        self.values["episodes_done_in_session"] += 1

        self.calculate_and_log_results()