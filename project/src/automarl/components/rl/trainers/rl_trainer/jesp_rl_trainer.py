import copy
from automarl.components.rl.policy.policy import Policy
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.components.rl.trainers.rl_trainer_component import RLTrainerComponent
import torch

from automarl.component import ParameterSignature
from automarl.components.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel




class JESPTrainer(RLTrainerComponent):

    """
    Exhaustive-JESP style trainer for parallel environments.

    Starting from the current joint policy, each agent is trained sequentially
    while the remaining agents are kept fixed. If, after a full iteration over
    all agents, no agent changed its policy, training is considered converged.

    """

    TRAIN_LOG = "train.txt"

    parameters_signature = {
        "policy_change_tolerance": ParameterSignature(
            default_value=0.0,
            description="Tolerance used when comparing policy parameters."
        ),
        "max_steps_per_jesp_maximization" : ParameterSignature(
            default_value=-1
        )
    }

    exposed_values = {
        "converged": False,
        
        # these two is to mantain state in case algorithm is interrupted mid execution
        "current_agent_index" : 0, 
        "changed_any_agent" : False, # if any agent was changed in a pass
        "step_before_curr_jesp_iteraition" : -1
    }

    def _process_input_internal(self):

        super()._process_input_internal()

        self.policy_change_tolerance = self.get_input_value("policy_change_tolerance")
        self.max_steps_per_jesp_maximization = self.get_input_value("max_steps_per_jesp_maximization")

        self.agent_order = [*self.agents_trainers.keys()]

        self.lg.writeLine(f"JESP trainer initialized with agent order: {self.agent_order}\n")

    # -------------------------------------------------------------------------
    # Policy snapshot / comparison
    # -------------------------------------------------------------------------

    def _get_policy_state(self, agent_name: str, clone=False):
        agent_trainer = self.agents_trainers[agent_name]
        policy : Policy = agent_trainer.agent.policy
        model : TorchModelComponent = policy.model

        state = model.get_model_state_dict()

        with torch.no_grad():

            if clone:
                return {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in state.items()}
            else:
                return state


    def _policy_states_equal(self, old_state, new_state) -> bool:

        if old_state.keys() != new_state.keys():
            return False

        for key in old_state.keys():
            old_val = old_state[key]
            new_val = new_state[key]

            if torch.is_tensor(old_val) and torch.is_tensor(new_val):
                
                if not torch.allclose(old_val, new_val, 
                                      atol=self.policy_change_tolerance if self.policy_change_tolerance > 0 else 1e-12, 
                                      rtol=0.0):
                    return False
            else:
                if old_val != new_val:
                    return False

        return True


    def _did_agent_policy_change(self, agent_name: str, old_state) -> bool:
        new_state = self._get_policy_state(agent_name)
        return not self._policy_states_equal(old_state, new_state)

    # -------------------------------------------------------------------------
    # Agent activation / freezing
    # -------------------------------------------------------------------------

    def _set_only_agent_training(self, active_agent_name: str):
        for agent_name, agent_trainer in self.agents_trainers.items():
            agent_trainer.is_training = (agent_name == active_agent_name)

        self.lg.writeLine(f"Only agent '{active_agent_name}' is set to train in this JESP turn")

    def _set_all_agents_not_training(self):
        for agent_trainer in self.agents_trainers.values():
            agent_trainer.is_training = False

    # -------------------------------------------------------------------------
    # JESP turn
    # -------------------------------------------------------------------------

    def _should_stop_due_to_max_jesp_steps(self):
        return self.max_steps_per_jesp_maximization > 0 and self.values["total_steps"] - self.values["step_before_curr_jesp_iteraition"] >= self.max_steps_per_jesp_maximization
    
    def _run_best_response_for_agent(self, agent_name: str) -> bool:
        """
        Train a single agent while keeping all others fixed.
        Returns True if the agent's policy changed during this turn.
        """
        self.lg.writeLine(f"Starting JESP best-response turn for agent '{agent_name}'")

        old_policy_state = self._get_policy_state(agent_name, clone=True)

        self._set_only_agent_training(agent_name)

        active_trainer = self.agents_trainers[agent_name]

        active_trainer.setup_training_session()

        # this is so this number is resumed in the case the training was interrupted mid execution
        self.values["step_before_curr_jesp_iteraition"] = self.values["total_steps"] if self.values["step_before_curr_jesp_iteraition"] < 0 else self.values["step_before_curr_jesp_iteraition"]

        changed = True # we assume changes in the cases we do not evaluate

        while active_trainer.is_agent_training():

            self.run_single_episode(self.values["episodes_done"])

            if self._check_if_to_end_training_session():
                self.lg.writeLine(
                    f"Global stop condition reached while training best response for agent '{agent_name}'"
                )
                break

            if self._should_stop_due_to_max_jesp_steps():
                self.lg.writeLine(f"Reached max steps for jesp iteration when training agent {agent_name}")
                break

        if active_trainer.is_agent_training():
            
            if self._should_stop_due_to_max_jesp_steps(): # we stopped due to max steps
                changed = self._did_agent_policy_change(agent_name, old_policy_state)
                self.values["step_before_curr_jesp_iteraition"] = -1 
            
            active_trainer.end_training()

        else: # if the training ended naturally and not because of RLTrainer constraints
            changed = self._did_agent_policy_change(agent_name, old_policy_state)
            self.values["step_before_curr_jesp_iteraition"] = -1

        if changed:
            self.lg.writeLine(f"Agent '{agent_name}' changed its policy in this JESP turn")
        else:
            self.lg.writeLine(f"Agent '{agent_name}' did not change its policy in this JESP turn")

        self.lg.writeLine(f"Ended JESP best-response turn for agent '{agent_name}'\n")

        return changed

    # -------------------------------------------------------------------------
    # Training session control
    # -------------------------------------------------------------------------

    def setup_training_session(self):
        """
        Custom setup for JESP:
        - reset environment
        - reset trainer counters
        - do NOT automatically start all agent trainers at once
        """
        self.lg._writeLine(
            f"Starting JESP training with number of episodes: {self.num_episodes} "
            f"and total step limit: {self.limit_total_steps}"
        )

        if self._fraction_training_to_do is not None:
            if self._fraction_training_to_do <= 0 or self._fraction_training_to_do > 1:
                raise Exception(
                    f"Fraction of training to do must be between 0 and 1, was {self._fraction_training_to_do}"
                )

            self.lg._writeLine(f"Only doing a fraction of {self._fraction_training_to_do} of the training")

        self.lg._writeLine("Resetting the environment...")
        self.env.total_reset()

        self.values["episodes_done_in_session"] = 0
        self.values["steps_done_in_session"] = 0
        self.values["converged"] = False # this can be initialized here because, if we're running the training, we certainly want it to not be over right away

        self.external_should_end_training_session = False
        self._set_all_agents_not_training()


    def end_training_session(self):
        self.lg._writeLine(f"Ended JESP training with values: {self.values}")

        for agent_in_training in self.agents_trainers.values():
            agent_in_training.end_training()

        self.env.close()

    def _check_and_end_training_due_to_agents(self):
        pass

    def _check_if_to_end_training_by_agent_behavior(self):
        return False # JESP takes control of ending the training due to agent behavior
    
    def _should_be_over_due_to_agents_trainers_over(self):
        return False

    def _is_over(self):
        
        isover = super()._is_over()

        if not isover:
            if self.values["converged"]:
                self.lg.writeLine(f"As the values converged, the training is considered as over")
                isover = True

            else:
                self.lg.writeLine(f"JESP algorithm did not detect all trainers as having converged")

        return isover

    # -------------------------------------------------------------------------
    # Main JESP loop
    # -------------------------------------------------------------------------

    def run_episodes(self):
        """
        Outer loop = JESP iterations.
        Inner loop = one best-response optimization per agent.
        Converges when a full pass over all agents produces no policy changes.
        """
        self.setup_training_session()

        self.lg.writeLine(f"Starting JESP run with initial values: {self.values}")

        while True:
            
            if self.values["current_agent_index"] == 0: # if we are still in first index (in other cases, this may have been resumed)
                self.values["changed_any_agent"] = False
            
            self.lg.writeLine(
                f"Starting JESP outer iteration..."
            )

            for agent_name_index in range(self.values["current_agent_index"], len(self.agent_order)):

                self.values["current_agent_index"] = agent_name_index
                agent_name = self.agent_order[agent_name_index]

                if self._check_if_to_end_training_session():
                    break

                changed = self._run_best_response_for_agent(agent_name)

                if changed:
                    self.values["changed_any_agent"] = True

                if self._check_if_to_end_training_session():
                    break

            self.values["current_agent_index"] = 0

            if not self.values["changed_any_agent"]:
                self.values["converged"] = True
                self.lg.writeLine(
                    f"JESP converged, as there were no detected changes to agents policies"
                )
                break
            
            else:
                self.values["converged"] = False

            if self._check_if_to_end_training_session():
                break

        self.end_training_session()


class JESPParalelTrainer(JESPTrainer, RLTrainerComponentParallel):
    '''
    JESP for Parallel trainers
    '''