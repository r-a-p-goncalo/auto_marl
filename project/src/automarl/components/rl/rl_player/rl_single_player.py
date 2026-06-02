
from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.components.rl.agent.agent_components import AgentSchema
import torch


class RLSinglePlayer(RLPlayer):
    

    def _process_input_internal(self):
        super()._process_input_internal()

        # ensure exactly one agent exists
        if len(self.agents) != 1:
            raise Exception(
                f"RLSinglePlayer expects exactly 1 agent, got {len(self.agents)}"
            )

        self.agent_name = list(self.agents.keys())[0]
        self.agent: AgentSchema = self.agents[self.agent_name]

    def _do_agent_step(self, agent_name=None):
        """
        Single-agent step (no AEC scheduling).
        """

        # current state BEFORE action
        observation, reward, done, truncated, info = self.env.last()

        # update memory
        self.agent.update_state_memory(observation)

        # choose action or null if terminal
        if done or truncated:
            action = None
        else:
            with torch.no_grad():                
                action = self.agent.policy_predict_with_memory().squeeze(0)

        # step environment
        self.env.step(action)

        # update episode stats
        self.values["episode_score"] += reward
        self.values["episode_steps"] += 1
        self.values["total_steps"] += 1

        self.values["agents_episode_score"][self.agent_name] += reward

        return reward, done, truncated

    def _run_episode(self):
        """
        Runs a full episode in standard Gym-style loop.
        """

        done = False
        truncated = False

        while not (done or truncated):
            _, done, truncated = self._do_agent_step(self.agent_name)