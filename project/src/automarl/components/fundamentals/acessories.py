

from automarl.component import Component, requires_input_process
from automarlcore.advanced_input_management import ComponentParameterSignature

class AcessoryComponent(Component):
    
    '''
    Executes some acessory functionality to an original component, it is responsibility of components that have acessories to know when to call them
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "affected_component" : ComponentParameterSignature(mandatory=False)
    }    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.affected_component : Component = self.get_input_value("affected_component")
                
    
    @requires_input_process
    def pre_fun(self, values : dict = None):
        '''To be called before a functionality is executed'''
        pass

    @requires_input_process
    def as_fun(self, values : dict = None):
        '''To be called as a functionality is executed'''
        pass

    @requires_input_process
    def pos_fun(self, values : dict):
        '''To be called after a functionality is executed'''
        pass







