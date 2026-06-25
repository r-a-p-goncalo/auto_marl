from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch

from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.component import ParameterSignature, requires_input_process
from automarl.components.rl.mcts.information_state import InformationState
from automarl.components.rl.mcts.mcts_node import MCTSNode


@dataclass
class SearchResult:
    action: Any
    action_val: Any
    search_policy: np.ndarray
    root_value: float
    visit_counts: np.ndarray
    root: MCTSNode
    sampled_hidden_states_info: list[Any] = field(default_factory=list)


class ImperfectInformationSearchPlanner(ComponentWithLogging):
    """
    AlphaZero-style PUCT planner.

    This class intentionally contains the temporary adapter logic too.  Later,
    when the framework stabilizes, split the adapter methods into a proper
    search environment/search backend interface.

    Supported env/search-state hooks, tried in this order:
    - env.get_search_state(), env.clone_state(), information_state.public_state
    - env.sample_hidden_state(information_state) for determinizations
    - env.current_player(state), state.current_player(), env.current_agent
    - env.legal_actions(state, player), state.legal_actions(player)
    - env.action_mask(state, player), state.action_mask(player)
    - env.next_search_state(state, action), state.next_search_state(action)
    - env.step_search(state, action), state.step_search(action)
    - env.is_terminal(state), state.is_terminal()
    - env.terminal_value(state, root_player), state.terminal_value(root_player)
    """

    parameters_signature = {
        "num_simulations": ParameterSignature(default_value=200),
        "c_puct": ParameterSignature(default_value=1.5),
        "temperature": ParameterSignature(default_value=1.0),
        "hidden_state_samples": ParameterSignature(default_value=1),
        "use_public_tree": ParameterSignature(default_value=True),
        "dirichlet_alpha": ParameterSignature(default_value=0.0),
        "dirichlet_epsilon": ParameterSignature(default_value=0.25),
        "max_actions": ParameterSignature(mandatory=False),
    }

    def _process_input_internal(self):
        super()._process_input_internal()
        self.num_simulations = int(self.get_input_value("num_simulations"))
        self.c_puct = float(self.get_input_value("c_puct"))
        self.temperature = float(self.get_input_value("temperature"))
        self.hidden_state_samples = int(self.get_input_value("hidden_state_samples"))
        self.use_public_tree = bool(self.get_input_value("use_public_tree"))
        self.dirichlet_alpha = float(self.get_input_value("dirichlet_alpha"))
        self.dirichlet_epsilon = float(self.get_input_value("dirichlet_epsilon"))
        self.max_actions = self.get_input_value("max_actions")
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @requires_input_process
    def search(
        self,
        env,
        agent_name: str,
        information_state: InformationState | None,
        evaluator,
        opponent_policy=None,
        training: bool = True,
    ) -> SearchResult:
        if information_state is None:
            information_state = InformationState.from_env(env, agent_name)

        root_state, sampled_info = self._make_root_state(env, information_state)
        root_player = agent_name
        root_player_to_move = self._current_player(env, root_state, default=agent_name)

        root = MCTSNode(
            state=root_state,
            parent=None,
            prior=1.0,
            player_to_move=root_player_to_move,
            root_player=root_player,
        )

        self._expand_leaf(
            root,
            env=env,
            root_player=root_player,
            information_state=information_state,
            evaluator=evaluator,
            opponent_policy=opponent_policy,
            add_root_noise=training,
        )

        sims_per_sample = max(1, self.num_simulations // max(1, self.hidden_state_samples))

        for sample_index in range(max(1, self.hidden_state_samples)):
            if sample_index > 0 and not self.use_public_tree:
                root_state, info = self._make_root_state(env, information_state)
                sampled_info.append(info)
                root = MCTSNode(
                    state=root_state,
                    parent=None,
                    prior=1.0,
                    player_to_move=self._current_player(env, root_state, default=agent_name),
                    root_player=root_player,
                )
                self._expand_leaf(
                    root,
                    env=env,
                    root_player=root_player,
                    information_state=information_state,
                    evaluator=evaluator,
                    opponent_policy=opponent_policy,
                    add_root_noise=training,
                )

            for _ in range(sims_per_sample):
                self._run_simulation(
                    root,
                    env=env,
                    root_player=root_player,
                    information_state=information_state,
                    evaluator=evaluator,
                    opponent_policy=opponent_policy,
                )

        max_actions = self._infer_max_actions(env, root, information_state)
        temperature = self.temperature if training else 0.0
        search_policy = root.visit_distribution(max_actions=max_actions, temperature=temperature)
        visit_counts = root.visit_counts(max_actions=max_actions)
        action = root.sample_action(temperature=temperature, rng=self.rng)

        return SearchResult(
            action=action,
            action_val=action,
            search_policy=search_policy,
            root_value=root.q_value,
            visit_counts=visit_counts,
            root=root,
            sampled_hidden_states_info=sampled_info,
        )

    # ------------------------------------------------------------------
    # Core MCTS
    # ------------------------------------------------------------------

    def _run_simulation(self, root, env, root_player, information_state, evaluator, opponent_policy=None):
        node = root

        while node.is_expanded and not self._is_terminal(env, node.state):
            _, node = node.select_child(self.c_puct)

        if self._is_terminal(env, node.state):
            value = self._terminal_value(env, node.state, root_player)
        else:
            value = self._expand_leaf(
                node,
                env=env,
                root_player=root_player,
                information_state=information_state,
                evaluator=evaluator,
                opponent_policy=opponent_policy,
                add_root_noise=False,
            )

        node.backup(value)
        return value

    def _expand_leaf(
        self,
        node,
        env,
        root_player,
        information_state,
        evaluator,
        opponent_policy=None,
        add_root_noise=False,
    ):
        player_to_move = self._current_player(env, node.state, default=root_player)
        node.player_to_move = player_to_move

        actions, action_mask = self._legal_actions_and_mask(env, node.state, player_to_move, information_state)

        if len(actions) == 0:
            return self._terminal_value(env, node.state, root_player)

        eval_obj = evaluator if player_to_move == root_player or opponent_policy is None else opponent_policy
        priors, value = self._evaluate(eval_obj, node.state, action_mask, player_to_move, information_state, env)
        priors = self._priors_for_actions(priors, actions, action_mask)

        if add_root_noise and self.dirichlet_alpha > 0 and len(actions) > 1:
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
            priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
            priors = self._normalize_np(priors)

        for action, prior in zip(actions, priors):
            next_state = self._next_state(env, node.state, action)
            next_player = self._current_player(env, next_state, default=root_player)
            node.add_child(action, next_state, prior, next_player)

        return float(value)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, evaluator, state, action_mask, agent_name, information_state, env):
        for method_name in ("evaluate_search_state", "evaluate_mcts_state", "evaluate"):
            method = getattr(evaluator, method_name, None)
            if method is not None:
                result = self._call_with_supported_args(
                    method,
                    (state, action_mask, agent_name, information_state, env),
                    (state, action_mask, agent_name, information_state),
                    (state, action_mask, agent_name),
                    (state, action_mask),
                    (state,),
                )
                if result is not None:
                    return self._parse_evaluation_result(result, action_mask)

        if callable(evaluator):
            result = self._call_with_supported_args(
                evaluator,
                (state, action_mask, agent_name, information_state, env),
                (state, action_mask, agent_name, information_state),
                (state, action_mask, agent_name),
                (state, action_mask),
                (state,),
            )
            if result is not None:
                return self._parse_evaluation_result(result, action_mask)

        policy = getattr(evaluator, "policy", None)
        if policy is None and hasattr(evaluator, "get_policy"):
            policy = evaluator.get_policy()

        if policy is not None:
            obs_state = self._state_to_policy_input(state, action_mask, agent_name, information_state, env)
            with torch.no_grad():
                logits = policy.predict_model_output(obs_state)
                masked_logits = self._mask_logits_if_needed(logits, obs_state.get("action_mask"))
                priors = torch.softmax(masked_logits, dim=-1)
                priors = priors.detach().cpu().reshape(-1).numpy()

            value = 0.0
            critic = getattr(evaluator, "critic", None)
            if critic is not None:
                with torch.no_grad():
                    value_tensor = critic.predict(obs_state["observation"])
                    value = float(value_tensor.reshape(-1)[0].detach().cpu())

            return priors, value

        max_actions = self._mask_size(action_mask)
        if max_actions is None:
            max_actions = self.max_actions or 1
        return np.ones(max_actions, dtype=np.float32) / max_actions, 0.0

    def _parse_evaluation_result(self, result, action_mask):
        if isinstance(result, dict):
            priors = result.get("priors", result.get("policy", result.get("probabilities")))
            value = result.get("value", result.get("root_value", 0.0))
            if priors is None and "logits" in result:
                priors = torch.softmax(torch.as_tensor(result["logits"]), dim=-1)
            return self._to_numpy_1d(priors, action_mask), self._to_float(value)

        if isinstance(result, tuple) and len(result) >= 2:
            return self._to_numpy_1d(result[0], action_mask), self._to_float(result[1])

        return self._to_numpy_1d(result, action_mask), 0.0

    def _state_to_policy_input(self, state, action_mask, agent_name, information_state, env):
        observation = None

        if isinstance(state, dict):
            if "observation" in state:
                observation = state["observation"]
            elif agent_name in state:
                observation = state[agent_name]

        if observation is None:
            for obj in (state, env):
                if obj is None:
                    continue
                for method_name in ("observe", "observation_for", "private_observation"):
                    method = getattr(obj, method_name, None)
                    if method is not None:
                        observation = self._call_with_supported_args(method, (agent_name, state), (agent_name,), (state,), ())
                        if observation is not None:
                            break
                if observation is not None:
                    break

        if isinstance(observation, dict):
            policy_input = dict(observation)
            policy_input.setdefault("observation", observation.get("observation"))
        else:
            policy_input = {"observation": observation}

        if policy_input["observation"] is None:
            policy_input["observation"] = information_state.private_observation

        if action_mask is not None:
            policy_input["action_mask"] = action_mask
        elif information_state.action_mask is not None:
            policy_input["action_mask"] = information_state.action_mask

        return policy_input

    # ------------------------------------------------------------------
    # Search-state adapter methods
    # ------------------------------------------------------------------

    def _make_root_state(self, env, information_state):
        sampled_info = None

        sampler = getattr(env, "sample_hidden_state", None)
        if sampler is not None:
            result = self._call_with_supported_args(sampler, (information_state,), ())
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
            if result is not None:
                return result, {"source": "sample_hidden_state"}

        for method_name in ("get_search_state", "clone_state"):
            method = getattr(env, method_name, None)
            if method is not None:
                state = self._call_with_supported_args(method, (information_state,), ())
                if state is not None:
                    return state, sampled_info

        if information_state.public_state is not None:
            return deepcopy(information_state.public_state), sampled_info

        raise RuntimeError(
            "MCTS needs a root search state. Provide env.get_search_state(), "
            "env.clone_state(), env.sample_hidden_state(), or information_state.public_state."
        )

    def _current_player(self, env, state, default=None):
        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("current_player", "player_to_move", "current_agent_name"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    value = self._call_with_supported_args(method, (state,), ())
                    if value is not None:
                        return value

        for attr in ("current_agent", "agent_selection", "player", "turn"):
            if hasattr(env, attr):
                return getattr(env, attr)
            if hasattr(state, attr):
                return getattr(state, attr)

        return default

    def _legal_actions_and_mask(self, env, state, player, information_state):
        actions = None
        action_mask = None

        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("legal_actions", "get_legal_actions", "available_actions"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    actions = self._call_with_supported_args(method, (state, player), (player, state), (player,), (state,), ())
                    if actions is not None:
                        break
            if actions is not None:
                break

        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("action_mask", "get_action_mask", "legal_action_mask"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    action_mask = self._call_with_supported_args(method, (state, player), (player, state), (player,), (state,), ())
                    if action_mask is not None:
                        break
            if action_mask is not None:
                break

        if action_mask is None:
            action_mask = information_state.action_mask

        if actions is None and action_mask is not None:
            mask_np = self._to_bool_mask(action_mask)
            actions = [int(i) for i, is_legal in enumerate(mask_np) if is_legal]

        if actions is None:
            raise RuntimeError(
                "MCTS needs legal actions. Provide legal_actions(...) or action_mask(...)."
            )

        actions = [self._normalize_action(a) for a in list(actions)]
        return actions, action_mask

    def _next_state(self, env, state, action):
        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("next_search_state", "next_state", "step_search", "search_step"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    next_state = self._call_with_supported_args(method, (state, action), (action, state), (action,))
                    if next_state is not None:
                        return next_state

        restore = getattr(env, "restore_state", None)
        clone = getattr(env, "clone_state", None)
        step_search = getattr(env, "step_search", None)

        if restore is not None and clone is not None and step_search is not None:
            restore(deepcopy(state))
            step_search(action)
            return clone()

        raise RuntimeError(
            "MCTS needs a non-destructive transition. Provide next_search_state(...) "
            "or clone_state/restore_state/step_search."
        )

    def _is_terminal(self, env, state):
        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("is_terminal", "terminal", "is_done", "done"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    value = self._call_with_supported_args(method, (state,), ())
                    if value is not None:
                        return bool(value)

        return False

    def _terminal_value(self, env, state, root_player):
        for obj in (env, state):
            if obj is None:
                continue
            for method_name in ("terminal_value", "value", "result_value", "get_result_value"):
                method = getattr(obj, method_name, None)
                if method is not None:
                    value = self._call_with_supported_args(method, (state, root_player), (root_player, state), (root_player,), (state,), ())
                    if value is not None:
                        return self._to_float(value)

        rewards = getattr(env, "rewards", None)
        if rewards is not None:
            try:
                reward_dict = rewards()
                if isinstance(reward_dict, dict) and root_player in reward_dict:
                    return float(reward_dict[root_player])
            except Exception:
                pass

        return 0.0

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _infer_max_actions(self, env, root, information_state):
        if self.max_actions is not None:
            return int(self.max_actions)

        if information_state.action_mask is not None:
            size = self._mask_size(information_state.action_mask)
            if size is not None:
                return size

        if root.children:
            numeric_actions = [a for a in root.children if isinstance(a, (int, np.integer))]
            if numeric_actions:
                return int(max(numeric_actions)) + 1
            return len(root.children)

        action_space = None
        method = getattr(env, "get_agent_action_space", None)
        if method is not None:
            action_space = self._call_with_supported_args(method, (information_state.agent_name,), ())
        if action_space is not None and hasattr(action_space, "n"):
            return int(action_space.n)

        return 1

    def _priors_for_actions(self, priors, actions, action_mask):
        priors = self._to_numpy_1d(priors, action_mask)

        selected = []
        for action in actions:
            if isinstance(action, (int, np.integer)) and 0 <= int(action) < len(priors):
                selected.append(float(priors[int(action)]))
            else:
                selected.append(1.0)

        selected = np.asarray(selected, dtype=np.float64)
        return self._normalize_np(selected)

    def _normalize_np(self, values):
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 0.0, None)
        total = float(values.sum())
        if total <= 0:
            return np.ones_like(values, dtype=np.float64) / max(1, len(values))
        return values / total

    def _to_numpy_1d(self, value, action_mask=None):
        if value is None:
            size = self._mask_size(action_mask) or self.max_actions or 1
            return np.ones(size, dtype=np.float32) / size

        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).numpy()
        else:
            value = np.asarray(value, dtype=np.float32).reshape(-1)

        if value.size == 0:
            size = self._mask_size(action_mask) or self.max_actions or 1
            value = np.ones(size, dtype=np.float32)

        return self._normalize_np(value).astype(np.float32)

    def _to_float(self, value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().reshape(-1)[0])
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    def _to_bool_mask(self, mask):
        if torch.is_tensor(mask):
            return (mask.detach().cpu().reshape(-1).numpy() > 0)
        return np.asarray(mask).reshape(-1) > 0

    def _mask_size(self, mask):
        if mask is None:
            return None
        if torch.is_tensor(mask):
            return int(mask.numel())
        return int(np.asarray(mask).size)

    def _mask_logits_if_needed(self, logits, action_mask):
        if action_mask is None:
            return logits
        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, device=logits.device)
        action_mask = action_mask.to(device=logits.device)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask > 0
        while action_mask.dim() < logits.dim():
            action_mask = action_mask.unsqueeze(0)
        return logits.masked_fill(~action_mask, -1e9)

    def _normalize_action(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().reshape(-1)
            if action.numel() == 1:
                return int(action.item())
            return tuple(int(x.item()) for x in action)
        if isinstance(action, np.ndarray):
            action = action.reshape(-1)
            if action.size == 1:
                return int(action.item())
            return tuple(int(x) for x in action.tolist())
        if isinstance(action, list):
            return tuple(action) if len(action) != 1 else action[0]
        return action

    def _call_with_supported_args(self, fun: Callable, *arg_sets):
        for args in arg_sets:
            try:
                return fun(*args)
            except TypeError:
                continue
        return None
