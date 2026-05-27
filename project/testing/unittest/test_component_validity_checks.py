from automarl.component import Component, ParameterSignature
import unittest



    

class TestComponentLambdaVerificator(unittest.TestCase):
        
    class _ComponentLambdaVerificator(Component):
    
        N_MUST_BE_GREATER_THAN = 5
    
        parameters_signature = {"number" : ParameterSignature(validity_verificator=lambda n : n > TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN)}        
        
    def test_wrong_input(self):
                
        simple_extended_component = TestComponentLambdaVerificator._ComponentLambdaVerificator(input={"number" : TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN - 1})
        
        try:
            simple_extended_component.process_input()
            assert False
        
        except:
            assert True    
        
    def test_right_input(self):
                
        simple_extended_component = TestComponentLambdaVerificator._ComponentLambdaVerificator(input={"number" : TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN + 1})
        
        try:
            simple_extended_component.process_input()
            assert True
        
        except:
            assert False    
        

if __name__ == '__main__':
    unittest.main()