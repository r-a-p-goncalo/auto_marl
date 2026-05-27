from automarl.components.basic_components.evaluator_component import EvaluatorComponent
from automarl.component import Component
from automarl.core.input_management import ParameterSignature


class ValueEvaluator(EvaluatorComponent):

    parameters_signature = {
        "value_to_use" : ParameterSignature(description="The exposed value to use")
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.value_to_use = self.get_input_value("value_to_use")

    def _evaluate(self, component_to_evaluate : Component) -> dict:
        return {"result": component_to_evaluate.values[self.value_to_use] }