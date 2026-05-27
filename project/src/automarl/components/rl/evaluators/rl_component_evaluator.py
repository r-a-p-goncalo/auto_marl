
from automarl.component import Component, requires_input_process
from automarl.components.basic_components.evaluator_component import EvaluatorComponent
from automarl.components.rl.rl_pipeline import RLPipelineComponent


class RLPipelineEvaluator(EvaluatorComponent):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {}
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_process
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):
        return super()._evaluate(component_to_evaluate)