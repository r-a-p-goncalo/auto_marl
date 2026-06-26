import torch

from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
from automarl.components.rl.rl_player.rl_player import RLPlayer


class RLAECPlayer(RLPlayer):
    """RLPlayer implementation for AEC environments."""

    parameters_signature = {}

    def _process_input_internal(self):
        super()._process_input_internal()

        self.env: AECEnvironmentComponent = self.env

        if not isinstance(self.env, AECEnvironmentComponent):
            raise Exception(
                f"RLAECPlayer requires AECEnvironmentComponent, got {type(self.env)}"
            )

    def _do_agent_step(self, agent_name):
        agent: AgentSchema = self.agents[agent_name]

        observation, reward, done, truncated, info = self.env.last()
        agent.update_state_memory(observation)

        if done or truncated:
            self.env.step(None)
        else:
            with torch.no_grad():
                action = agent.policy_predict_with_memory()
            self.env.step(action)

        self.values["episode_score"] = self.values["episode_score"] + reward
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1
        self.values["agents_episode_score"][agent_name] += reward

        return reward, done, truncated

    def _run_episode(self):
        for agent_name in self.env.agent_iter():
            self._do_agent_step(agent_name)