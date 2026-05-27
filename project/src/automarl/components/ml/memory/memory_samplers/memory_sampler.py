

from automarl.component import Component, requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automarl.core.input_management import ParameterSignature
from automarl.components.ml.memory.memory_components import MemoryComponent


class MemorySampler(Component):

    '''Used with memory to do extra processing before sampling'''

    parameters_signature = {

    }

    def _process_input_internal(self):
        super()._process_input_internal()


    @requires_input_process
    def prepare(self, memory : MemoryComponent = None):
        '''Method to do any extra processing with whole memory before sampling'''

        self.memory : MemoryComponent = memory if memory is not None else self.memory

        if memory is None:
            raise Exception("No memory passed")
        

    @requires_input_process
    def sample(self, batch_size):
        return self.memory.sample(batch_size)
    
        
    @requires_input_process
    def get_all(self):
        '''Returns the total memory'''
        return self.memory.get_all()


    @requires_input_process
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        return self.memory.get_all_segmented(batch_size)
    
    @requires_input_process
    def get_capacity(self):
        return self.memory.get_capacity()

    @requires_input_process
    def let_go(self):
        '''let go of any references to fields in memory'''
        pass

    @requires_input_process
    def add_extra_columns(self):
        pass

    @requires_input_process
    def __len__(self):
        return len(self.memory)