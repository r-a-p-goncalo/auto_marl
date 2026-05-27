



from automarl.components.basic_components.component_group import RunnableComponentGroup
from automarl.components.basic_components.evaluator_component import EvaluatorComponent


def config_dict(component_config_dict : dict, component_evaluator : EvaluatorComponent, number_of_components=3):

    return {
    
    "__type__": str(RunnableComponentGroup),
    "name": "ComponentGroup",
    "input": {
        
        "number_of_components" : number_of_components,
        "component_dic": component_config_dict,
        "component_evaluator" : component_evaluator
    },
    "child_components": [
    ]
}