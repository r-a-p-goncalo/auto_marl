

import os

from automarl.component import  requires_input_process
from automarl.core.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.utils.shapes_util import torch_shape_from_space
import torch
from automarl.components.ml.memory.memory_components import MemoryComponent

from automarl.component import requires_input_process
from automarl.core.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.ml.memory.memory_components import MemoryComponent


class TorchMemoryComponent(MemoryComponent, ComponentWithLogging):

    parameters_signature = {
        "capacity": ParameterSignature(default_value=1_000),
        "device": ParameterSignature(default_value="cuda"),
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position = 0
        self.total_size = 0


    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.device = self.get_input_value("device")

        self.lg.writeLine("Initializing TorchMemoryComponent...")

        self.allocate_computer_memory_to_transitions()
        
        self.position = 0
        self.total_size = 0
        
        self.lg.writeLine("TorchMemoryComponent initialized.")
        
        
        
    def allocate_computer_memory_to_transitions(self):
        
        '''Allocates the necessary space '''
        
        self.transitions = self._allocate_computer_memory_to_transitions_dictionary() # transitions saved in memory        


            
    def _allocate_computer_memory_to_transitions_dictionary(self) -> dict[str, torch.Tensor]:

        '''Allocates necessary space given the specification of name, shape and data type in field_shapes'''
            
        transitions : dict[str, torch.Tensor] = {}

        
        for field_name, field_shape, data_type in self.fields_shapes:

            field_shape = torch_shape_from_space(field_shape)

            #normally it would be state, action, next_state and reward 
            
            if data_type == None:
                   
                transitions[field_name] = torch.zeros((self.capacity, *field_shape),
                                                       device=self.device)                
            else:
                transitions[field_name] = torch.zeros((self.capacity, *field_shape),
                                                       device=self.device, dtype=data_type)
                
        for key in transitions.keys():
            self.lg.writeLine(f"Component {self.name} allocated for key '{key}' torch tensor with shape {transitions[key].shape}")
            
        return transitions
            
        

    @requires_input_process
    def push(self, transition):
        """Pushes a transition into memory, coercing values to the slot shape when safe."""
        idx = self.position
    
        for field_name in self.field_names:
            target = self.transitions[field_name][idx]
            value = transition[field_name]
    
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            else:
                value = value.to(device=target.device, dtype=target.dtype)
    
            # Fast path
            if value.shape != target.shape:
                # Remove trivial singleton dimensions first, e.g. [1, 3] -> [3]
                squeezed = value.squeeze()
    
                if squeezed.shape == target.shape:
                    value = squeezed
    
                # If same number of elements, reshape safely
                elif value.numel() == target.numel():
                    value = value.reshape(target.shape)
    
                else:
                    raise RuntimeError(
                        f"Cannot store field '{field_name}': "
                        f"memory slot shape={tuple(target.shape)}, "
                        f"received shape={tuple(value.shape)}, "
                        f"numel(slot)={target.numel()}, numel(value)={value.numel()}"
                    )
    
            target.copy_(value)
    
        self.position = (self.position + 1) % self.capacity
    
        if self.total_size < self.capacity:
            self.total_size += 1
        
    def _get_batch_from_indices(self, indices=None):

        if indices is not None:

            batch_data = {
                field_name: self.transitions[field_name][indices]
                for field_name in self.field_names
            }        

            for key in self.extra_columns:
                batch_data[key] = self.extra_columns[key][indices]
            
        else:
            batch_data = {
                field_name: self.transitions[field_name][:self.total_size]
                for field_name in self.field_names
            }        

            for key in self.extra_columns:
                batch_data[key] = self.extra_columns[key][:self.total_size]
        
        return batch_data


    @requires_input_process
    def sample(self, batch_size):

        '''Returns <batch_size> random elements from the saved transitions'''
        
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")
        
        indices = torch.randint(0, self.total_size, (batch_size,), device=self.device)
                
        batch_data = self._get_batch_from_indices(indices)
        
        return batch_data
    
    
    @requires_input_process
    def sample_all_with_batches(self, batch_size) -> list:
        '''
        Samples all memory divided by a list of batches of the specified size, without repeating information

        If there is more memory than the allowed by the batches, some of it is left out
        '''
        
        total_full_batches = self.total_size // batch_size

        total_to_sample = total_full_batches * batch_size

        # Random permutation without repetition
        indices = torch.randperm(self.total_size, device=self.device)[:total_to_sample]

        batches = []

        for i in range(0, total_to_sample, batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_data = self._get_batch_from_indices(batch_indices)

            batches.append(batch_data)

        return batches
    

    @requires_input_process
    def get_all(self):
        '''Returns the total memory'''

        batch_data = self._get_batch_from_indices()
        
        return batch_data


    @requires_input_process
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        pass


        
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
        
        super().clear()

        self.position = 0
        self.total_size = 0
        
        
        
    def __len__(self):
        return self.total_size
    
    def _save_state_internal(self):

        super()._save_state_internal()

        if self.total_size == 0:
            self.lg.writeLine("No transitions to save.")
            return

        self._save_transitions_to_disk()

    def _save_transitions_to_disk(self):

        file_path = os.path.join(self.get_artifact_directory(), "transitions.pt")

        if self.total_size == 0:
            self.lg.writeLine("No transitions to save.")
            return

        self.lg.writeLine(f"Saving {self.total_size} transitions to {file_path}")

        # Save only the used portion of memory
        transitions_to_save = {
            field_name: (
                self.transitions[field_name][:self.total_size]
                .detach()
                .cpu()
                .clone()
            )
            for field_name in self.field_names
        }

        save_dict = {
            "total_size": self.total_size,
            "position": self.position,
            "transitions": transitions_to_save,
        }
    
        torch.save(save_dict, file_path)
    
        self.lg.writeLine("Finished saving transitions.")


    def _load_state_internal(self):

        super()._load_state_internal()
        self._load_transitions_from_disk()

        

    def _load_transitions_from_disk(self):

        file_path = os.path.join(self.get_artifact_directory(), "transitions.pt")

        if not os.path.exists(file_path):
            self.lg.writeLine("No transitions file found.")
            return

        self.lg.writeLine(f"Loading transitions from {file_path}")

        if not hasattr(self, "device"):
            self.device = self.get_input_value("device")
            self.lg.writeLine(f"Transitions stored on device: {self.device}")

        checkpoint = torch.load(file_path, map_location=self.device)

        self.clear()

        loaded_transitions = checkpoint["transitions"]
        loaded_size = checkpoint["total_size"]

        if loaded_size > self.capacity:
            raise ValueError(
                f"Saved transitions ({loaded_size}) exceed memory capacity ({self.capacity})."
            )

        for field_name in self.field_names:
            self.transitions[field_name][:loaded_size].copy_(
                loaded_transitions[field_name].to(self.device)
            )

        self.total_size = loaded_size
        self.position = checkpoint.get("position", loaded_size % self.capacity)

        self.lg.writeLine(f"Loaded {self.total_size} transitions from disk.")