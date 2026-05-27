

from automarl.components.basic_components.evaluator_component import EvaluatorComponent
from automarl.component import Component, requires_input_process
import random

class RandomMockEvaluator(EvaluatorComponent):
    

    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), "result"]
    
    @requires_input_process
    def _evaluate(self, component_to_evaluate : Component):

        result = random.random()
        return {"result" : result, **super().evaluate(component_to_evaluate)}