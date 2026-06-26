

from automarl.component import Component, requires_input_process
from automarl.core.advanced_input_management import ComponentListParameterSignature, ComponentParameterSignature
from automarl.serialization.json_component_utils import gen_component_from
from automarl.components.basic_components.exec_component import ExecComponent

class GroupEvaluator(ExecComponent):
    
    '''
    A component that evaluates a group of components, being able to give it a single numeric score for each    
    '''

    
    parameters_signature = {
        "components_to_evaluate" : ComponentListParameterSignature(mandatory=False)
                    }    
    
    exposed_values = {
        
        "last_evaluation" : 0
    
    }

    def __init__(self, input = None):
        super().__init__(input)

        if self.values["last_evaluation"] == 0:
            self.values["last_evaluation"] = {}

    def _process_input_internal(self):
        
        super()._process_input_internal()


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_process # needs to be extended
    def get_metrics_strings(self) -> list[str]:
        '''
        Gets the keys for the evaluation this evaluator
        This method is meant to be extended
        '''
        return []
    
    @requires_input_process
    def evaluate(self, components_to_evaluate : list[Component]) -> dict:
        '''
        Returns a dictionary with the results of the evaluation for each component
        A value for the key "result" will always exist
        '''

        if not isinstance(list, components_to_evaluate):
            components_to_evaluate = [components_to_evaluate]

        for index in range(len(components_to_evaluate)):

            component_to_evaluate = components_to_evaluate[index] 

            if not isinstance(component_to_evaluate, Component):
                component_to_evaluate = gen_component_from(component_to_evaluate)

            components_to_evaluate[index] = component_to_evaluate

        results = self._evaluate(components_to_evaluate)

        self.values["last_evaluation"] = results

        return results

    #needs to be extended
    def _evaluate(self, components_to_evaluate : list[Component]) -> dict[str, dict]:
        return { component_to_evaluate.name : {} for component_to_evaluate in components_to_evaluate}
    
    def _algorithm(self):
        components_to_evaluate = self.get_input_value("components_to_evaluate")

        if components_to_evaluate is None:
            raise Exception(f"Tried to run evaluator without passing a components to evaluate")
        
        results = self.evaluate(components_to_evaluate)

        self.output = {**results}


class EvaluatorComponent(ExecComponent):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
        "component_to_evaluate" : ComponentParameterSignature(mandatory=False)
                    }    
    
    exposed_values = {
        
        "last_evaluation" : 0
    
    }

    def __init__(self, input = None):
        super().__init__(input)

        if self.values["last_evaluation"] == 0:
            self.values["last_evaluation"] = {}

    def _process_input_internal(self):
        
        super()._process_input_internal()


    # EVALUATION -------------------------------------------------------------------------------

    
    @requires_input_process # needs to be extended
    def get_metrics_strings(self) -> list[str]:
        '''
        Gets the keys for the evaluation this evaluator
        This method is meant to be extended
        '''
        return []
    
    @requires_input_process
    def evaluate(self, component_to_evaluate : Component) -> dict:
        '''
        Returns a dictionary with the results of the evaluation
        A value for the key "result" will always exist
        '''

        if not isinstance(component_to_evaluate, Component):
            component_to_evaluate = gen_component_from(component_to_evaluate)

        results = self._evaluate(component_to_evaluate)

        self.values["last_evaluation"] = results

        return results

    #needs to be extended
    def _evaluate(self, component_to_evaluate : Component) -> dict:
        return {}
    
    def _algorithm(self):
        component_to_evaluate = self.get_input_value("component_to_evaluate")

        if component_to_evaluate is None:
            raise Exception(f"Tried to run evaluator without passing a component to evaluate")
        
        results = self.evaluate(component_to_evaluate)

        self.output = {**results}

        


class ComponentWithEvaluator(Component):
    
    '''
    A component that has an evaluator
    It is used to evaluate the component itself
    '''
    
    parameters_signature = {
        "component_evaluator" : ComponentParameterSignature(mandatory=False),
    }
    
    exposed_values = {
        "last_evaluation" : None
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.last_evaluation = {}
    
    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.component_evaluator : EvaluatorComponent = self.get_input_value("component_evaluator", look_in_attribute_with_name="component_evaluator")        
    
    def pass_evaluation(self, evaluation):
        '''
        Passes an evaluation of this component to it for internal processing, such as verifying if the algorithm is over
        '''

    @requires_input_process
    def evaluate_this_component(self) -> dict:
        
        '''
        Evaluates this component using its evaluator
        '''
        
        if self.component_evaluator is None:
            raise Exception("This component does not have an evaluator")
        
        self.values["last_evaluation"] = self._evaluate_this_component()
        
        return self.values["last_evaluation"]

    def _evaluate_this_component(self) -> dict:
        return self.component_evaluator.evaluate(self)

    @requires_input_process
    def get_last_evaluation(self):
        
        to_return = self.values.get("last_evaluation", None)
        
        return to_return