from automarl.components.basic_components.artifact_management import ParameterSignature
from automarl.component import Component
import unittest


class _SimpleComponent(Component):
    
    parameters_signature = {"number" : ParameterSignature()}
    

class _SimpleExtendendComponent(_SimpleComponent):
    
    parameters_signature = {"number" : ParameterSignature(default_value=10)}
    
    
class _SimpleComponent_2(Component):
    
    parameters_signature = {"number" : ParameterSignature(default_value=10)}
    

class _SimpleExtendendComponent_2(_SimpleComponent_2):
    
    parameters_signature = {"number" : ParameterSignature()}


class TestDefaultOverlap(unittest.TestCase):
        
    def test_default_in_super(self):
                
        simple_extended_component = _SimpleExtendendComponent(input={})
        
        simple_extended_component.process_input()
            
        assert simple_extended_component.input["number"] == 10
        
    def test_default_in_child(self):
        
        
        simple_extended_component_2 = _SimpleExtendendComponent_2(input={})
        simple_extended_component_2.process_input()
        assert simple_extended_component_2.input["number"] == 10
              
        

if __name__ == '__main__':
    unittest.main()