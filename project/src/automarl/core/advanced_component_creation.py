

from automarl.component import Component
from automarl.utils.class_util import get_all_subclasses, get_class_from, get_class_from_string
from automarl.utils.json_utils.json_component_utils import gen_component_from


def create_component_with_look_for_class(component_definition):
    
    if isinstance(component_definition, tuple): #if it is a tuple (class, input)
        
        return component_from_tuple_definition_with_look_for_class(component_definition)
    
    else:
        return gen_component_from(component_definition)
    
    


def component_from_tuple_definition_with_look_for_class(tuple_definition) -> Component:

    '''
    Generates a component from a tuple definition (Component_definition, input)
    
    '''
     
    class_definition = tuple_definition[0]
    
    if len(tuple_definition) > 2:
        raise Exception(f"Tuple definition has more than 2 elements, but it should have only 2, got {len(tuple_definition)}")
    
    elif len(tuple_definition) == 2:
        input = tuple_definition[1]
        
        if not isinstance(input, dict):
            raise Exception(f"Input in tuple is not a dict, but {type(input)}")
    
    else:
        input = {} #input was not passed
        
    class_of_component : type = get_class_from(class_definition)
    
    class_of_component = get_sub_class_with_correct_parameter_signature(class_of_component, input)
        
    component_to_return = class_of_component(input=input)
    return component_to_return



def get_sub_class_with_correct_parameter_signature(cls, input : dict) -> type[Component]:
    
    all_subclasses = get_all_subclasses(cls)

    for sub_cls in all_subclasses:
        if class_has_correct_parameter_signature(sub_cls, input):
            return sub_cls

    raise Exception(f"No subclass of {cls} has the correct parameter signature for input with keys {input.keys()}. Subclasses: {all_subclasses}")



def class_has_correct_parameter_signature(cls : type[Component], input):
    
    '''Checks if a class has the correct parameter signature'''
    
    for input_key in input.keys():
        if cls.get_schema_parameter_signature(input_key) is None: # if there was an input_key not in the class
            return False
    
    return True