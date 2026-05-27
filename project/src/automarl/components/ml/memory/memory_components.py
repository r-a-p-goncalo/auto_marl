from collections import namedtuple
from collections import deque
import random



from automarl.components.basic_components.state_management import StatefulComponent
from automarl.component import Component, ParameterSignature, requires_input_process

class MemoryComponent(StatefulComponent):
    
    parameters_signature = {
                        "capacity" : ParameterSignature(default_value=1000),
                        "transition_data" : ParameterSignature(description="A list of data there is for the transitions, of tuples (name, shape, (type)?)")
                    }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.capacity = self.get_input_value("capacity")
        
        self.fields_shapes = []

        self.transition_data = self.get_input_value("transition_data")
        
        for transition_data in self.transition_data:
            
            if len(transition_data) < 3:
                data_name, shape = transition_data
                data_type = None
            
            else:
                data_name, shape, data_type = transition_data
                
            self.fields_shapes.append((data_name, shape, data_type))
                
        self.field_names = [data_name for (data_name, _, _) in self.fields_shapes]
        
        self.extra_columns = {}
                          

    #save a transition
    @requires_input_process
    def push(self, transition):
        raise NotImplementedError()

    @requires_input_process
    def sample(self, batch_size):
        raise NotImplementedError()
    
    @requires_input_process
    def sample_all_with_batches(self, batch_size):
        '''
        Samples all memory divided by batches of the specified size, without repeating information

        If there is more memory than the allowed by the batches, some of it is left out
        '''
        raise NotImplementedError()
    
    
    @requires_input_process
    def clear(self):
        self.extra_columns.clear()
        
    @requires_input_process
    def get_all(self):
        '''Returns the total memory'''
        pass


    @requires_input_process
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        pass

    @requires_input_process
    def get_capacity(self):
        return self.capacity


    @requires_input_process
    def __len__(self):
        raise NotImplementedError()
    
    @requires_input_process
    def is_full(self):
        return len(self) == self.capacity
    
    @requires_input_process
    def add_extra_columns(self, extra_columns : dict):
        '''Adds new temporary columns'''

        self.extra_columns = {**self.extra_columns, **extra_columns}
