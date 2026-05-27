from automarl.components.basic_components import loop_components
import unittest

from automarl.components.basic_components.exec_component import ExecComponent


class TestExecComponent(unittest.TestCase):
    
    def test_output_reference(self):
        
        exec_component = ExecComponent()
        
        exec_component.process_input()
        
        output = exec_component.run()
                
        assert output is exec_component.output



def condition(self):
    return self.i < 10

def execution(self):
    self.i = self.i + 1

def pre_execution(self):
    self.i = 0

def post_execution(self):
    self.output["result"] = self.i

class TestWhileComponent(unittest.TestCase):
    
    def test_correct(self):
        
        while_component = loop_components.WhileFunDoFunComponent({
                "condition": condition,
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            })
        
        while_component.process_input()
        
        while_component.run()
                
        assert while_component.i == 10

    def test_correct_by_output(self):
        
        while_component = loop_components.WhileFunDoFunComponent({
                "condition": condition,
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            })
        
        while_component.process_input()
        
        output = while_component.run()
                
        assert output["result"] == 10
        
    def test_wrong_type(self):
        

        
        try:
                        
            while_component = loop_components.WhileFunDoFunComponent(                {
                    "condition": 5,
                    "execution": execution,
                    "pre_execution": pre_execution,
                    "post_execution" : post_execution
                })
            
            while_component.process_input()
        
            output = while_component.run()
            
            assert False
        
        except:
            assert True
                
        


        

class TestDoNTimesComponent(unittest.TestCase):
    
    def test_correct_default(self):
        
        do_n_times_component = loop_components.DoNTimesComponent(            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            })
        
        do_n_times_component.process_input()
        
        output = do_n_times_component.run()
        
        assert output["result"] == loop_components.DoNTimesComponent.DEFAULT_TIMES_TO_DO
        
    def test_correct_non_default(self):
        
        custom_times_to_do = 20

        
        do_n_times_component = loop_components.DoNTimesComponent(
                        {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            }
        )
        
        do_n_times_component.process_input()
        
        output = do_n_times_component.run()
        
        assert output["result"] == custom_times_to_do
        
    def test_correct_non_default_input_in_init(self):
        
        custom_times_to_do = 30
        
        do_n_times_component = loop_components.DoNTimesComponent( #pass input in initialization
            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            })
        
        
        output = do_n_times_component.run()
        
        assert output["result"] == custom_times_to_do
        
    def test_correct_non_default_input_in_between(self):
        
        custom_times_to_do = 30
        
        do_n_times_component = loop_components.DoNTimesComponent(
                        {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            }
        )
        
        
        output = do_n_times_component.run()
        
        assert output["result"] == custom_times_to_do

        

if __name__ == '__main__':
    unittest.main()