from automarl.components.basic_components.dynamic_value import get_value_or_dynamic_value
from automarl.component import Component, ParameterSignature, requires_input_process

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.fundamentals.acessories import AcessoryComponent


from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.rl.trainers.agent_trainer_component import AgentTrainer
import torch


class ConvergenceDetector(AcessoryComponent, ComponentWithLogging):
    
    '''
    Detects convergence by comparing old and new values and noting their differences
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "memory_size" : ParameterSignature(default_value=256),
                        
                        "convergence_treshold" : ParameterSignature(default_value=0.001),
                        
                        "old_values_new_values_keys" : ParameterSignature(default_value=["log_prob", "new_log_probs"])

                        }    
    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
                
        self.kl_divergence_memory_size = self.get_input_value("memory_size")

        self.affected_component : LearnerSchema = self.affected_component

        self.agent_trainer : AgentTrainer = self.affected_component.agent_trainer

        if self.agent_trainer is None:
            raise Exception(f"Convergence detector needs agent trainer defined in learner")
        
        self.agent_trainer.initialize_external_end_request(self.name)

        # circular buffer
        self.kl_divergence_memory = []
        self._kl_memory_index = 0

        self.kl_divergence_treshold = self.get_input_value("convergence_treshold")

        self.lg.writeLine(f"Convergence will be noted with kl divergence, using an average of {self.kl_divergence_memory_size} values, that must be bellow {self.kl_divergence_treshold}")

        self.old_values_new_values = self.get_input_value("old_values_new_values_keys")

        self.old_values_key = self.old_values_new_values[0]
        self.new_values_key = self.old_values_new_values[1]



    def _update_convergence_and_check(self, new_log_probs, old_log_probs):
        """
        Computes approximate KL divergence between old and new policy.
        Stores it and checks convergence.
        """

        with torch.no_grad():
            # Standard PPO approximate KL
            approx_kl = (old_log_probs - new_log_probs).mean().item()

        # Fill buffer
        if len(self.kl_divergence_memory) < self.kl_divergence_memory_size:
            self.kl_divergence_memory.append(approx_kl)
        else:
            self.kl_divergence_memory[self._kl_memory_index] = approx_kl
            self._kl_memory_index = (self._kl_memory_index + 1) % self.kl_divergence_memory_size

        # Only check after buffer full
        if len(self.kl_divergence_memory) < self.kl_divergence_memory_size:
            return False

        avg_kl = sum(self.kl_divergence_memory) / self.kl_divergence_memory_size

        if avg_kl < self.kl_divergence_treshold and avg_kl > -self.kl_divergence_treshold:

            # Notify trainer
            self.agent_trainer.request_end_from_external(self.name)

            return True

        return False
    

    @requires_input_process
    def pos_fun(self, values : dict):
        '''To be called after a functionality is executed'''

        new_values = values.get(self.new_values_key)
        old_values = values.get(self.old_values_key)

        if new_values is None or old_values is None:
            raise Exception(f"Could not compute, wanted both <{self.old_values_key}> and <{self.new_values_key}>, values had keys: {values.keys()}")
    
        self._update_convergence_and_check(new_values, old_values)
