from collections import namedtuple
from collections import deque
import random



from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.ml.memory.memory_components import MemoryComponent

class DeqeueMemoryComponent(MemoryComponent):
    
    
    parameters_signature = {
                    }

    def _process_input_internal(self):
        
        raise NotImplementedError("This is not up to date")

        super()._process_input_internal()
        
        self.memory = deque([], maxlen=self.capacity)

    #save a transition
    @requires_input_process
    def push(self, *args):
        self.memory.append(self.Transition(*args))

    @requires_input_process
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    @requires_input_process
    def sample_transposed(self, batch_size):
        return self.transpose(self.sample(batch_size))    
    
    
    def transpose(self, transitions):
        return self.Transition(*zip(*transitions))
    
    @requires_input_process
    def clear(self):
        self.memory.clear()
        
    @requires_input_process
    def get_all(self):
        return list(self.memory) 

    @requires_input_process
    def __len__(self):
        return len(self.memory)
    


