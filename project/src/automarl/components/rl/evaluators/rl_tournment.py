"""Round-robin evaluator for adversarial RL agents.

The class name keeps the original misspelling (`RLTournament`) so existing
configuration files do not break. New code may prefer importing the alias
`RLTournament` at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

from automarl.components.loggers.logger_component import ComponentWithLogging
import torch

from automarl.component import Component
from automarl.components.basic_components.evaluator_component import GroupEvaluator
from automarl.components.loggers.component_with_results import ComponentWithResults
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
    """Evaluate RL components in a two-player round-robin tournament.

    Standalone AgentSchema components are evaluated normally.

    RLPipelineComponent components are expanded into their internal agents for
    the actual matches. After the round-robin, the pipeline receives the score
    of whichever internal agent performed best.
    """

    parameters_signature = {
        "environment": ComponentParameterSignature(),
        "matches_per_configuration": ParameterSignature(default_value=3),
        "device": ParameterSignature(default_value="cpu"),
        "agent_names": ParameterSignature(mandatory=False),
        "require_matching_agent_names": ParameterSignature(default_value=True),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.environment_definition : EnvironmentComponent = self.get_input_value("environment")

        self.lg.writeLine(f"Tournament will be made in the environment: {self.environment_definition.get_env_name()}")

        self.matches_per_configuration = int(
            self.get_input_value("matches_per_configuration")
        )

        self.lg.writeLine(f"{self.matches_per_configuration} will be made per configuration")
        
        self.device = torch.device(self.get_input_value("device"))
        self.agent_names = self.get_input_value("agent_names")
        self.require_matching_agent_names = self.get_input_value(
            "require_matching_agent_names"
        )


        env = self._new_environment()
        env.process_input_if_not_processed()
        env_agent_names = list(env.agents())
        env.close()

        if self.agent_names is None:
            self.agent_names = env_agent_names
        
        else:
            self.agent_names = list(self.agent_names)
            if len(self.agent_names) != 2:
                raise ValueError(
                    "agent_names must contain exactly two environment agent names"
                )

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
        """Return a fresh environment instance for multiple matches."""

        if isinstance(self.environment_definition, Component):
            try:
                return self.environment_definition.clone(save_in_parent=False)
            except Exception:
                return self.environment_definition

        return gen_component_from(self.environment_definition, self)

    def _component_name(self, component: Any, index: int) -> str:
        return getattr(component, "name", None) or f"component_{index}"
    

    def _agents_from_rl_pipeline(self, component: RLPipelineComponent) -> dict[str, AgentSchema]:
        
        component.process_input_if_not_processed()

        if isinstance(component, RLPipelineComponent) or hasattr(component, "get_agents"):

            agents = {}

            for agent_internal_name, agent in component.get_agents().items():
                
                self.lg.writeLine(f"Processing {agent_internal_name} of {component.name}")
                
                agents[agent_internal_name] = agent

            
            self.lg.writeLine(f"Processing agents of component {component.name}: {agents.keys()}")

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
        """Expand top-level components into concrete agents to be matched.

        Standalone agents produce one tournament entry.

        Pipelines produce one tournament entry per internal agent. The pipeline
        will later be scored using the best of those entries.
        """

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

            if isinstance(component, RLPipelineComponent) or hasattr(component, "get_agents"):
                
                agents = self._agents_from_rl_pipeline(component)

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

        for agent_name in env.agent_iter():
            observation, reward, done, truncated, info = env.last()
            agent = agents_by_slot[agent_name]
            agent.update_state_memory(observation)

            if done or truncated:
                env.step(None)
            else:
                with torch.no_grad():
                    action = agent.policy_predict_with_memory()
                env.step(action)

        return {
            agent_name: float(reward)
            for agent_name, reward in env.rewards().items()
        }

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

    def _register_result(
        self,
        scoreboard: dict[str, dict[str, float]],
        tournament_agent_key: str,
        reward: float,
    ) -> None:
        scoreboard[tournament_agent_key]["matches"] += 1

        if reward > 0:
            scoreboard[tournament_agent_key]["wins"] += 1
        elif reward < 0:
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
    

    def _finalize_scoreboard(
        self,
        agent_scoreboard: dict[str, dict[str, float]],
        tournament_agents: list[_TournamentAgent],
        components_to_evaluate: list[Component],
    ) -> dict[str, dict[str, float]]:
        """Collapse agent-level scores back to component-level scores.

        For standalone agents, this returns that agent's score.

        For pipelines, this returns the score of the best-performing internal
        agent belonging to that pipeline.
        """

        agent_scores = {
            tournament_agent.key: self._score_from_stats(
                agent_scoreboard[tournament_agent.key]
            )
            for tournament_agent in tournament_agents
        }

        finalized = {}

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

            finalized[component_name] = agent_scores[selected_agent.key]

        return finalized

    def _evaluate(
        self,
        components_to_evaluate: list[Component],
    ) -> dict[str, dict[str, float]]:
        if len(components_to_evaluate) < 2:
            raise ValueError("RLTournament needs at least two components to evaluate")

        tournament_agents = self._build_tournament_agents(components_to_evaluate)

        if len(tournament_agents) < 2:
            raise ValueError("RLTournament needs at least two agents to evaluate")

        agent_scoreboard = self._empty_agent_scoreboard(tournament_agents)

        for left_index, right_index in permutations(range(len(tournament_agents)), 2):
            left_agent = tournament_agents[left_index]
            right_agent = tournament_agents[right_index]

            if left_agent.component_index == right_agent.component_index:
                continue

            agents_by_component_slot = {
                self.agent_names[0]: left_agent,
                self.agent_names[1]: right_agent,
            }

            for _ in range(self.matches_per_configuration):
                env = self._new_environment()
                env.process_input_if_not_processed()

                agents_by_slot = {
                    slot_name: self._agent_for_slot(
                        tournament_agent,
                        slot_name,
                        env,
                    )
                    for slot_name, tournament_agent in agents_by_component_slot.items()
                }

                rewards = self._play_one_episode(env, agents_by_slot)
                env.close()

                for slot_name, tournament_agent in agents_by_component_slot.items():
                    self._register_result(
                        agent_scoreboard,
                        tournament_agent.key,
                        rewards.get(slot_name, 0.0),
                    )

        return self._finalize_scoreboard(
            agent_scoreboard,
            tournament_agents,
            components_to_evaluate,
        )
