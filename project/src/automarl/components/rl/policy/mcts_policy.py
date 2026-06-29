import torch
import numpy as np

from automarl.components.rl.policy.policy import Policy
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.component import requires_input_process


class MCTSPolicy(Policy):
    """
    AlphaZero-style policy:

    - Policy network -> prior probabilities (used by MCTS)
    - Value network  -> state value (used by leaf evaluation)
    - MCTS planner   -> action selection

    This policy does NOT directly act from logits/Q-values.
    Instead it runs search at inference time.
    """

    parameters_signature = {
        "value_network": ComponentParameterSignature(mandatory=False),
        "search_planner": ComponentParameterSignature(mandatory=False),
    }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _process_input_internal(self):
        super()._process_input_internal()

        self.lg.writeLine("Initializing MCTSPolicy (AlphaZero-style)...")

        # Optional value network
        self.value_network = self.get_input_value("value_network", mandatory=False)

        # Optional search planner (can also be injected)
        self.search_planner = self.get_input_value("search_planner", mandatory=False)

        if self.search_planner is None:
            raise Exception(
                "MCTSPolicy requires a search_planner (ImperfectInformationSearchPlanner)."
            )

        self.lg.writeLine("MCTSPolicy initialized successfully.")

    # ------------------------------------------------------------------
    # Evaluator interface used by MCTS
    # ------------------------------------------------------------------

    def evaluate_search_state(self, state, action_mask, agent_name, information_state=None, env=None):
        """
        Called by MCTS as evaluator(state).
        Returns:
            (policy_probs, value)
        """

        # --- policy prior ---
        model_out = self.predict_model_output(state)

        if torch.is_tensor(model_out):
            logits = model_out
            if action_mask is not None:
                logits = self._mask_logits(logits, action_mask)
            priors = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
        else:
            priors = np.asarray(model_out).reshape(-1)

        priors = priors / (priors.sum() + 1e-8)

        # --- value ---
        value = 0.0
        if self.value_network is not None:
            with torch.no_grad():
                obs = state.get("observation", state)
                v = self.value_network.predict(obs)
                value = float(v.detach().cpu().reshape(-1)[0])

        return priors, value

    def evaluate(self, state, action_mask=None, agent_name=None):
        return self.evaluate_search_state(state, action_mask, agent_name)

    # ------------------------------------------------------------------
    # Core AlphaZero behavior
    # ------------------------------------------------------------------

    @requires_input_process
    def get_action_val_from_model_output(self, model_output, state):
        """
        Instead of using raw model output, we run MCTS.
        """

        # Expected state structure:
        # {
        #   "env": environment,
        #   "agent_name": str,
        #   "information_state": optional InformationState
        # }

        env = state.get("env", None)
        agent_name = state.get("agent_name", None)

        if env is None or agent_name is None:
            raise RuntimeError(
                "MCTSPolicy requires state['env'] and state['agent_name'] for MCTS search."
            )

        information_state = state.get("information_state", None)

        result = self.search_planner.search(
            env=env,
            agent_name=agent_name,
            information_state=information_state,
            evaluator=self,
            training=False,
        )

        return result.action

    # ------------------------------------------------------------------
    # Action conversion (compatibility layer)
    # ------------------------------------------------------------------

    @requires_input_process
    def get_action_from_action_val(self, action_val):
        return action_val

    @requires_input_process
    def get_action_val_shape(self):
        return self.output_action_shape

    @requires_input_process
    def random_prediction(self, state):
        return self.output_action_shape.sample()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mask_logits(self, logits, action_mask):
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