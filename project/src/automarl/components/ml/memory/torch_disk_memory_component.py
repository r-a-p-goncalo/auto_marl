

from collections import namedtuple
from typing import Iterable
from automarl.component import Component, requires_input_process
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.utils.maths import nearest_highest_multiple, nearest_multiple
from automarl.utils.shapes_util import discrete_output_layer_size_of_space
import torch
from automarl.components.ml.memory.memory_components import MemoryComponent


import os
import torch
from pathlib import Path

from automarl.component import Component, requires_input_process
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.utils.shapes_util import discrete_output_layer_size_of_space
from automarl.components.ml.memory.memory_components import MemoryComponent


class TorchDiskMemoryComponent(MemoryComponent, ComponentWithLogging):

    parameters_signature = {
        "capacity": ParameterSignature(default_value=1_000),
        "device": ParameterSignature(default_value="cuda"),
        "max_in_memory": ParameterSignature(default_value=80),
        "storage_dir": ParameterSignature(default_value="./memory_storage"),
    }
    

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.device = self.get_input_value("device")
        self.max_in_memory = self.get_input_value("max_in_memory")

        self.set_capaticy()

        self.lg.writeLine("Initializing hybrid TorchMemoryComponent...")

        self.allocate_computer_memory_to_transitions()

        self.position = 0
        self.size_in_memory = 0
        self.total_size = 0
        
        self.initialize_disk_files()

        self.lg.writeLine("TorchMemoryComponent initialized.")
        

    def set_capaticy(self):
        
        '''Sets capacity to an appropriate value, multiple of the max transitions in memory'''

        prev_capacity = self.get_input_value("capacity")

        self.capacity = nearest_highest_multiple(prev_capacity, self.max_in_memory)
        
        if self.capacity != prev_capacity:
            self.lg.writeLine(f"Capacity of {prev_capacity} was changed to {self.capacity} due to it not being a multiple of max in memory ({self.max_in_memory})")

        
        
        
    def allocate_computer_memory_to_transitions(self):
        
        '''Allocates the necessary space '''
        
        self.transitions = self._allocate_computer_memory_to_transitions_dictionary() # transitions saved in memory        
        
            
            
    def _allocate_computer_memory_to_transitions_dictionary(self):
            
        transitions : dict[str, torch.Tensor] = {}
        
        for field_name, field_shape, data_type in self.fields_shapes:
            
            if not isinstance(field_shape, Iterable):
                field_shape = (field_shape,)
                
            
                
            #normally it would be state, action, next_state and reward 
            
            if data_type == None:
                   
                transitions[field_name] = torch.zeros((self.max_in_memory, *field_shape),
                                                       device=self.device)
            else:
                transitions[field_name] = torch.zeros((self.max_in_memory, *field_shape),
                                                       device=self.device, dtype=data_type)
            
        return transitions
            
        
    def initialize_disk_files(self):
        
        self.storage_dir = Path(os.path.join(self.get_artifact_directory(), self.get_input_value("storage_dir")))

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.disk_file_position = 0
        
        number_of_times_size_in_memory_in_capacity = int(self.capacity / self.max_in_memory)
        
        self.lg.writeLine(f"There are {number_of_times_size_in_memory_in_capacity} times the max_in_memory in the whole capacity of the memory")
        
        self.number_of_files = number_of_times_size_in_memory_in_capacity - 1
                
        self.disk_files = [os.path.join(self.storage_dir, f"transitions_{i}.pt") for i in range(0, self.number_of_files)]
        
        self.created_files = 0
        

    @requires_input_process
    def push(self, transition):
        
        idx = self.position

        for field_name in self.field_names:
                        
            self.transitions[field_name][idx].copy_(transition[field_name])

        self.position = (self.position + 1) % self.max_in_memory
        self.total_size += 1 #this is correct, we just added one to memory

        if self.size_in_memory < self.max_in_memory:
            self.size_in_memory += 1
    
            
        else:
            # Buffer is full → flush to disk
            self._flush_to_disk()

    

    def _flush_to_disk(self):
        
        """
        Save current memory buffer to disk and clear memory.
        """
        file_path = self.disk_files[self.disk_file_position]

        
        torch.save({
            **self.transitions,
            "size": self.size_in_memory
        }, file_path)

        self.lg.writeLine(f"Flushed {self.size_in_memory} transitions to disk at {file_path}")
        
        self.disk_file_position = (self.disk_file_position + 1) % self.number_of_files

        # We don't clear the memory because we don't need to, only new positions will be used
        self.position = 0
        self.size_in_memory = 0
        
        
        if self.total_size == self.capacity:
        
            self.total_size -= self.size_in_memory #we deleted self.size_in_memory entries previously stored
            
        if self.created_files < self.number_of_files:
            self.created_files += 1
        
    

    @requires_input_process
    def sample(self, batch_size):
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")

        # Probability of sampling from disk vs memory
        prob_disk = len(self.disk_files) / (len(self.disk_files) + 1e-5)
        from_disk = False

        if len(self.disk_files) > 0:
            from_disk = torch.rand(1).item() < prob_disk

        if from_disk:
            self.lg.writeLine("Samplig from disk...")
            return self._sample_from_disk(batch_size)
        else:
            self.lg.writeLine("Sampling from memory...")
            return self._sample_from_memory(batch_size)

    def _sample_from_memory(self, batch_size):
        
        indices = torch.randint(0, self.size_in_memory, (batch_size,), device=self.device)
                
        batch_data = {
            field_name: self.transitions[field_name][indices].cpu()
            for field_name in self.field_names
        }
        

        batch = self.Transition(**batch_data)
        
        return batch

    def _sample_from_disk(self, batch_size):
        
        file_path = np.random.choice(self.disk_files[0:self.created_files])

        data, size = self._load_data_from_file(file_path)

        indices = torch.randint(0, size, (batch_size,))
        
        batch_data = {
            field_name: data[field_name][indices].to(self.device)
            for field_name in self.field_names
        }
        
        batch = self.Transition(**batch_data)
        
        return batch
    
    def _load_data_from_file(self, file_path):
        
        data = torch.load(file_path, map_location="cpu")
        size = data["size"]
        
        return data, size

    @requires_input_process
    def sample_transposed(self, batch_size):
        batch = self.sample(batch_size)
        return self.Transition(*batch)
    

    def write_to_file(self):
        #self.lg.writeLine(, file="readable.txt")
        pass #TODO: do this
        
        
    def _transitions_to_str(self, transitions_dict : dict[str, torch.Tensor], size, transitions_to_save):
        
        str_to_return = "" 
        
        for i in range(size): # for each row
        
            for field_name in transitions_to_save:    
            
                str_to_return += f"{field_name}: {transitions_dict[field_name][i]} "
                
            str_to_return += "\n"
            
        return str_to_return
            
            
         

    @requires_input_process
    def clear(self):
        
        '''Logicaly cleans the memory, without doing any deletion operation'''
        
        self.position = 0
        self.disk_file_position = 0
        
        self.created_files = 0
        
        
    def __len__(self):
        return self.total_size