
from automarl.components.basic_components.exec_component import ParameterSignature
from automarl.component import Component, requires_input_process
from automarlcore.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automarl.components.ml.memory.memory_components import MemoryComponent
from automarl.components.ml.memory.memory_samplers.memory_sampler import MemorySampler
from automarl.components.ml.memory.memory_utils import interpret_values
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.rl.learners.ppo_learner import PPOLearner
import torch


class PPOAdvantagesCalcSampler(MemorySampler):

    '''
    Calculates and normalizes values for PPO before mini batch sampling
    It assumes the memory will be cleared after, and so it does in_place transformations
    '''

    parameters_signature = {
        "learner" : ComponentParameterSignature(),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.learner : PPOLearner = self.get_input_value("learner")


    def prepare(self, memory : MemoryComponent = None):
        '''Does the in place transformations'''
        
        super().prepare(memory)

        self.memory : TorchMemoryComponent = self.memory

        self.device = memory.device

        processed_memory =  self.learner.interpret_trajectory(self.memory.get_all())

        with torch.no_grad():

            if not "observation_old_critic_value" in processed_memory.keys():

                observation_critic_values, next_obs_critic_values = self.learner.compute_values_estimates(processed_memory)
            
                processed_memory["observation_old_critic_value"] = observation_critic_values
                processed_memory["next_obs_old_critic_value"] = next_obs_critic_values
                
            # we compute the advantages using the whole memory
            critic_obs_pred_error, non_normalized_advantages, advantages, returns = self.learner.compute_error_and_advantage( 
                                                                                                                             processed_memory,
                                                                                                                             processed_memory["observation_old_critic_value"],
                                                                                                                             processed_memory["next_obs_old_critic_value"])
    
            processed_memory["critic_obs_pred_error"] = critic_obs_pred_error
            processed_memory["non_normalized_advantages"] = non_normalized_advantages
            processed_memory["advantages"] = advantages
            processed_memory["returns"] = returns


        self.field_names = processed_memory.keys()
        self.transitions = processed_memory

    def let_go(self):
        super().let_go()

        self.field_names = None
        self.transitions = None

    @requires_input_process
    def sample(self, batch_size):

        '''Returns <batch_size> random elements from the saved transitions'''
        
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")
        
        indices = torch.randint(0, len(self.memory), (batch_size,), device=self.memory.device)
                
        batch_data = {
            field_name: self.transitions[field_name][indices]
            for field_name in self.field_names
        }        
        
        return batch_data
    
    @requires_input_process
    def sample_all_with_batches(self, batch_size) -> list:
        '''
        Samples all memory divided by a list of batches of the specified size, without repeating information

        If there is more memory than the allowed by the batches, some of it is left out
        '''
        
        total_full_batches = len(self.memory) // batch_size

        total_to_sample = total_full_batches * batch_size

        # Random permutation without repetition
        indices = torch.randperm(len(self.memory), device=self.device)[:total_to_sample]

        batches = []

        for i in range(0, total_to_sample, batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_data = {
                field_name: self.transitions[field_name][batch_indices]
                for field_name in self.field_names
            }

            batches.append(batch_data)

        return batches
    
    @requires_input_process
    def get_all(self):
        '''Returns the total memory'''

        batch_data = {
            field_name: self.transitions[field_name][:len(self.memory)]
            for field_name in self.field_names
        }
        
        return batch_data