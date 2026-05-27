

from automarl.component import Component
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.core.input_management import ParameterSignature


class Transformer(Component):

    '''A strategy that transforms data, such as normalizing it'''

    parameters_signature = {

    }

    def _process_input_internal(self):
        super()._process_input_internal()


    def transform_data(self, data : dict):
        '''Method called to transform data'''
    
