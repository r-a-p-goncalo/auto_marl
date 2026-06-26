"""Round-robin evaluator for adversarial RL agents.

RLPipelineComponent contestants are expanded into their internal agents for the
actual matches. The pipeline's final score is the score of its best internal
agent.

Standalone AgentSchema contestants are evaluated normally.

The saved results dataframe stores only per-match data.
Aggregate agent/component summaries are exposed in this tournament's values:
- last_agent_evaluation
- last_component_evaluation
- last_selected_agents
- last_match_count
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import torch

from automarl.component import Component
from automarl.components.basic_components.evaluator_component import GroupEvaluator
from automarl.components.loggers.component_with_results import ComponentWithResults
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.environment_components import EnvironmentComponent
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.core.input_management import ParameterSignature
from automarl.serialization.json_component_utils import gen_component_from


@dataclass
class _TournamentAgent:
    """One concrete agent entry used internally by the tournament."""

    key: str
    component_index: int
    component_name: str
    agent_name: str
    agent: AgentSchema
    evaluate_component_with_best_agent: bool


class RLTournament(GroupEvaluator, ComponentWithResults, ComponentWithLogging):
    """Evaluate RL components in a two-player round-robin tournament."""

    exposed_values = {
        "last_agent_evaluation": {},
        "last_component_evaluation": {},
        "last_selected_agents": {},
        "last_match_count": 0,
    }

    parameters_signature = {
        "environment": ComponentParameterSignature(),
        "matches_per_configuration": ParameterSignature(default_value=3),
        "device": ParameterSignature(default_value="cpu"),
        "agent_names": ParameterSignature(mandatory=False),
        "require_matching_agent_names": ParameterSignature(default_value=True),
    }

    results_columns = [
        "environment",
        "match_index",
        "slot_0",
        "component_0",
        "agent_0",
        "reward_0",
        "outcome_0",
        "slot_1",
        "component_1",
        "agent_1",
        "reward_1",
        "outcome_1",
    ]

    def _process_input_internal(self):
        super()._process_input_internal()

        self.values["last_agent_evaluation"] = {}
        self.values["last_component_evaluation"] = {}
        self.values["last_selected_agents"] = {}
        self.values["last_match_count"] = 0

        self.environment_definition: EnvironmentComponent = self.get_input_value(
            "environment"
        )

        self.matches_per_configuration = int(
            self.get_input_value("matches_per_configuration")
        )

        if self.matches_per_configuration < 1:
            raise ValueError("matches_per_configuration must be >= 1")

        self.device = torch.device(self.get_input_value("device"))
        self.agent_names = self.get_input_value("agent_names")
        self.require_matching_agent_names = self.get_input_value(
            "require_matching_agent_names"
        )

        env = self._new_environment()
        env.process_input_if_not_processed()

        self.environment_name = self._get_environment_name(env)
        self.lg.writeLine(
            f"Tournament will be made in the environment: {self.environment_name}"
        )
        self.lg.writeLine(
            f"{self.matches_per_configuration} matches will be made per configuration"
        )

        env_agent_names = list(env.agents())
        env.close()

        if len(env_agent_names) != 2:
            raise ValueError(
                "RLTournament currently supports exactly 2 environment agents; "
                f"got {env_agent_names}"
            )

        if self.agent_names is None:
            self.agent_names = env_agent_names
        else:
            self.agent_names = list(self.agent_names)
            if len(self.agent_names) != 2:
                raise ValueError(
                    "agent_names must contain exactly two environment agent names"
                )

    def _get_environment_name(self, env: EnvironmentComponent) -> str:
        if hasattr(env, "get_env_name"):
            return env.get_env_name()

        return getattr(env, "name", type(env).__name__)

    def get_metrics_strings(self) -> list[str]:
        return [
            "result",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "wins",
            "losses",
            "draws",
            "matches",
        ]

    def _new_environment(self) -> EnvironmentComponent:
        """Return a fresh environment instance for a match."""

        if isinstance(self.environment_definition, Component):
            try:
                return self.environment_definition.clone(save_in_parent=False)
            except Exception:
                return self.environment_definition

        return gen_component_from(self.environment_definition, self)

    def _component_name(self, component: Any, index: int) -> str:
        return getattr(component, "name", None) or f"component_{index}"

    def _agents_from_rl_pipeline(
        self,
        component: RLPipelineComponent,
    ) -> dict[str, AgentSchema]:
        component.process_input_if_not_processed()

        if isinstance(component, RLPipelineComponent) or hasattr(component, "get_agents"):
            agents = {}

            for agent_internal_name, agent in component.get_agents().items():
                self.lg.writeLine(
                    f"Processing {agent_internal_name} of {component.name}"
                )
                agents[agent_internal_name] = agent

            self.lg.writeLine(
                f"Processing agents of component {component.name}: {agents.keys()}"
            )

            return agents

        raise TypeError(
            "RLTournament contestants must be AgentSchema instances, "
            "RLPipelineComponent instances, or components exposing get_agents(). "
            f"Got {type(component)}"
        )

    def _build_tournament_agents(
        self,
        components_to_evaluate: list[Component],
    ) -> list[_TournamentAgent]:
        """Expand top-level components into concrete agents to be matched."""

        tournament_agents = []

        for component_index, component in enumerate(components_to_evaluate):
            component_name = self._component_name(component, component_index)

            if isinstance(component, AgentSchema):
                component.process_input_if_not_processed()

                tournament_agents.append(
                    _TournamentAgent(
                        key=component_name,
                        component_index=component_index,
                        component_name=component_name,
                        agent_name=component.name,
                        agent=component,
                        evaluate_component_with_best_agent=False,
                    )
                )
                continue

            if isinstance(component, RLPipelineComponent) or hasattr(
                component,
                "get_agents",
            ):
                agents = self._agents_from_rl_pipeline(component)

                if len(agents) == 0:
                    raise ValueError(
                        f"Pipeline/component {component_name} has no agents to evaluate"
                    )

                for agent_name, agent in agents.items():
                    tournament_agents.append(
                        _TournamentAgent(
                            key=f"{component_name}::{agent_name}",
                            component_index=component_index,
                            component_name=component_name,
                            agent_name=agent_name,
                            agent=agent,
                            evaluate_component_with_best_agent=True,
                        )
                    )

                continue

            raise TypeError(
                "RLTournament contestants must be AgentSchema instances, "
                "RLPipelineComponent instances, or components exposing get_agents(). "
                f"Got {type(component)}"
            )

        return tournament_agents

    def _agent_for_slot(
        self,
        tournament_agent: _TournamentAgent,
        slot_agent_name: str,
        env: EnvironmentComponent,
    ) -> AgentSchema:
        """Prepare an agent so it can play the requested environment slot."""

        agent = tournament_agent.agent

        if agent.name != slot_agent_name:
            agent = agent.clone(
                save_in_parent=False,
                input_for_clone={"name": slot_agent_name},
            )

        agent.pass_input(
            {
                "device": self.device,
                "state_shape": env.get_agent_state_space(slot_agent_name),
                "action_shape": env.get_agent_action_space(slot_agent_name),
            }
        )
        agent.process_input_if_not_processed()

        return agent

    def _reset_agents(
        self,
        env: EnvironmentComponent,
        agents_by_slot: dict[str, AgentSchema],
    ) -> None:
        for agent_name, agent in agents_by_slot.items():
            agent.reset_agent_in_environment(env.observe(agent_name))

    def _play_one_episode(
        self,
        env: EnvironmentComponent,
        agents_by_slot: dict[str, AgentSchema],
    ) -> dict[str, float]:
        env.reset()
        self._reset_agents(env, agents_by_slot)

        episode_rewards = {
            agent_name: 0.0
            for agent_name in agents_by_slot.keys()
        }

        for agent_name in env.agent_iter():
            observation, reward, done, truncated, info = env.last()

            if agent_name in episode_rewards:
                episode_rewards[agent_name] += float(reward)

            agent = agents_by_slot[agent_name]
            agent.update_state_memory(observation)

            if done or truncated:
                env.step(None)
            else:
                with torch.no_grad():
                    action = agent.policy_predict_with_memory()
                env.step(action)

        return episode_rewards

    def _empty_agent_scoreboard(
        self,
        tournament_agents: list[_TournamentAgent],
    ) -> dict[str, dict[str, float]]:
        return {
            tournament_agent.key: {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "matches": 0,
            }
            for tournament_agent in tournament_agents
        }

    def _outcome_from_reward(self, reward: float) -> str:
        if reward > 0:
            return "win"

        if reward < 0:
            return "loss"

        return "draw"

    def _register_result(
        self,
        scoreboard: dict[str, dict[str, float]],
        tournament_agent_key: str,
        reward: float,
    ) -> None:
        scoreboard[tournament_agent_key]["matches"] += 1

        outcome = self._outcome_from_reward(reward)

        if outcome == "win":
            scoreboard[tournament_agent_key]["wins"] += 1
        elif outcome == "loss":
            scoreboard[tournament_agent_key]["losses"] += 1
        else:
            scoreboard[tournament_agent_key]["draws"] += 1

    def _score_from_stats(
        self,
        stats: dict[str, float],
    ) -> dict[str, float]:
        matches = stats["matches"]
        wins = stats["wins"]
        losses = stats["losses"]
        draws = stats["draws"]

        win_rate = wins / matches if matches > 0 else 0.0
        loss_rate = losses / matches if matches > 0 else 0.0
        draw_rate = draws / matches if matches > 0 else 0.0

        return {
            "result": win_rate,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "matches": matches,
        }

    def _log_match_result(
        self,
        agents_by_component_slot: dict[str, _TournamentAgent],
        rewards: dict[str, float],
        match_index: int,
    ) -> None:
        """Log only one specific match row to the results dataframe."""

        slot_0 = self.agent_names[0]
        slot_1 = self.agent_names[1]

        tournament_agent_0 = agents_by_component_slot[slot_0]
        tournament_agent_1 = agents_by_component_slot[slot_1]

        reward_0 = rewards.get(slot_0, 0.0)
        reward_1 = rewards.get(slot_1, 0.0)

        self.log_results(
            {
                "environment": self.environment_name,
                "match_index": match_index,
                "slot_0": slot_0,
                "component_0": tournament_agent_0.component_name,
                "agent_0": tournament_agent_0.agent_name,
                "reward_0": reward_0,
                "outcome_0": self._outcome_from_reward(reward_0),
                "slot_1": slot_1,
                "component_1": tournament_agent_1.component_name,
                "agent_1": tournament_agent_1.agent_name,
                "reward_1": reward_1,
                "outcome_1": self._outcome_from_reward(reward_1),
            }
        )

    def _finalize_scoreboard(
        self,
        agent_scoreboard: dict[str, dict[str, float]],
        tournament_agents: list[_TournamentAgent],
        components_to_evaluate: list[Component],
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[str, dict[str, Any]],
        dict[str, str],
    ]:
        """Collapse agent-level scores back to component-level scores.

        For standalone agents, this returns that agent's score.

        For pipelines, this returns the score of the best-performing internal
        agent belonging to that pipeline.
        """

        agent_scores = {}

        for tournament_agent in tournament_agents:
            score = self._score_from_stats(agent_scoreboard[tournament_agent.key])

            agent_scores[tournament_agent.key] = {
                "environment": self.environment_name,
                "component": tournament_agent.component_name,
                "agent": tournament_agent.agent_name,
                "component_index": tournament_agent.component_index,
                "evaluate_component_with_best_agent": (
                    tournament_agent.evaluate_component_with_best_agent
                ),
                **score,
            }

        finalized = {}
        selected_agents = {}

        for component_index, component in enumerate(components_to_evaluate):
            component_name = self._component_name(component, component_index)

            component_agents = [
                tournament_agent
                for tournament_agent in tournament_agents
                if tournament_agent.component_index == component_index
            ]

            if len(component_agents) == 0:
                finalized[component_name] = self._score_from_stats(
                    {
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "matches": 0,
                    }
                )
                selected_agents[component_name] = ""
                continue

            if isinstance(component, AgentSchema):
                selected_agent = component_agents[0]
            else:
                selected_agent = max(
                    component_agents,
                    key=lambda tournament_agent: agent_scores[
                        tournament_agent.key
                    ]["result"],
                )

            finalized[component_name] = {
                key: value
                for key, value in agent_scores[selected_agent.key].items()
                if key
                in {
                    "result",
                    "win_rate",
                    "loss_rate",
                    "draw_rate",
                    "wins",
                    "losses",
                    "draws",
                    "matches",
                }
            }

            selected_agents[component_name] = selected_agent.agent_name

        return finalized, agent_scores, selected_agents

    def _evaluate(
        self,
        components_to_evaluate: list[Component],
    ) -> dict[str, dict[str, float]]:

        tournament_agents = self._build_tournament_agents(components_to_evaluate)

        if len(tournament_agents) < 2:
            raise ValueError("RLTournament needs at least two agents to evaluate")

        agent_scoreboard = self._empty_agent_scoreboard(tournament_agents)
        match_index = 0

        component_indices = {
            tournament_agent.component_index
            for tournament_agent in tournament_agents
        }
        allow_same_component_matches = len(component_indices) == 1

        for left_index, right_index in permutations(range(len(tournament_agents)), 2):
            left_agent = tournament_agents[left_index]
            right_agent = tournament_agents[right_index]

            if (
                left_agent.component_index == right_agent.component_index
                and not allow_same_component_matches
            ):
                continue

            agents_by_component_slot = {
                self.agent_names[0]: left_agent,
                self.agent_names[1]: right_agent,
            }

            for _ in range(self.matches_per_configuration):
                match_index += 1

                env = self._new_environment()
                env.process_input_if_not_processed()

                try:
                    agents_by_slot = {
                        slot_name: self._agent_for_slot(
                            tournament_agent,
                            slot_name,
                            env,
                        )
                        for slot_name, tournament_agent
                        in agents_by_component_slot.items()
                    }

                    rewards = self._play_one_episode(env, agents_by_slot)

                finally:
                    env.close()

                for slot_name, tournament_agent in agents_by_component_slot.items():
                    reward = rewards.get(slot_name, 0.0)

                    self._register_result(
                        agent_scoreboard,
                        tournament_agent.key,
                        reward,
                    )

                self._log_match_result(
                    agents_by_component_slot=agents_by_component_slot,
                    rewards=rewards,
                    match_index=match_index,
                )

        if match_index == 0:
            raise ValueError(
                "RLTournament did not schedule any matches. "
                "Pass at least two components, or use a pipeline with at least two "
                "agents when evaluating a single component."
            )

        finalized, agent_scores, selected_agents = self._finalize_scoreboard(
            agent_scoreboard,
            tournament_agents,
            components_to_evaluate,
        )

        self.values["last_agent_evaluation"] = agent_scores
        self.values["last_component_evaluation"] = finalized
        self.values["last_selected_agents"] = selected_agents
        self.values["last_match_count"] = match_index

        self.save_dataframe()

        return finalized

    def plot_wins_per_agent(
        self,
        title: str = "Tournament Wins per Agent",
        figsize: tuple[int, int] = (8, 5),
        save_path: str | None = None,
        show: bool = True,
    ):
        """Plot the number of tournament wins per concrete internal agent.
    
        Uses self.values["last_agent_evaluation"], which is populated after
        calling evaluate(...).
    
        Returns:
            tuple: (fig, ax)
        """
    
        agent_evaluation = self.values.get("last_agent_evaluation", {})
    
        if len(agent_evaluation) == 0:
            raise ValueError(
                "No agent evaluation data found. "
                "Call tournament.evaluate(...) before plotting wins."
            )
    
        agent_labels = []
        wins = []
    
        for _, agent_data in agent_evaluation.items():
            component_name = agent_data["component"]
            agent_name = agent_data["agent"]
    
            agent_labels.append(f"{component_name}\n{agent_name}")
            wins.append(agent_data["wins"])
    
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots(figsize=figsize)
    
        ax.bar(agent_labels, wins)
        ax.set_title(title)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Wins")
    
        ax.tick_params(axis="x", rotation=30)
    
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    
        fig.tight_layout()
    
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
    
        if show:
            plt.show()
    
        return fig, ax