import json
import os
from typing import Union

from automarl.component import Component, ParameterSignature, ParameterMetaData

from automarl.utils.class_util import get_class_from, is_valid_str_class_definition

from enum import Enum

from automarl.utils.json_utils.custom_json_logic import get_custom_strategy
from automarlcore.localizations import get_component_by_localization
from automarl.components.loggers.global_logger import globalWriteLine

import inspect

TRY_CORRECT_ABSTRACT_CLASSES = True


# ENCODING --------------------------------------------------

class ComponentValuesElementsEncoder(json.JSONEncoder):
    
    '''Encodes elements of a component input or exposed value, which can be a component (defined by its localization) or a primitive type'''
    
    def __init__(self, *args, source_component=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.source_component = source_component
    
    def default(self, obj):
                        
        if isinstance(obj, Component):

            localization_from_source_to_obj, origin_of_computed_loc = obj.get_index_localization(self.source_component, accept_source_component_besides_targets=True)

            if origin_of_computed_loc != self.source_component:
                return "__outside_component"
            
            else:

                return {
                    "__type__": str(type(obj)),
                    "name" : obj.name,
                    "localization" : localization_from_source_to_obj
                }

        
        #elif isinstance(obj, Enum):
        #    return obj.value
        
        elif isinstance(obj, type):
            return str(obj)
        
        elif hasattr(obj, "to_dict"): # if it has a custom to_dict method
            
            if not hasattr(type(obj), "from_dict"):
                globalWriteLine(f"WARNING: Object {obj} has a to_dict method, but not a from_dict method in its type {type(obj)}", file="encoding_decoding.txt")

            return {"__type__": str(type(obj)), "object" : obj.to_dict()}
        
        # if we reached here, the type was none we treat by default
        custom_json_logic = get_custom_strategy(type(obj))

        if custom_json_logic != None:
            return {"__type__" : str(type(obj)), "object" : custom_json_logic.to_dict(obj)}


        # if none of our special conditions were verified, we use the default encoder on the object
        try:
            return super().default(obj) # this will actually always raise an exception, the correct pre processing was already done
        
        # if it fails, we give it a null value
        except Exception as e:
            globalWriteLine(f"WARNING: Exception when encoding obj: {obj}, {e}", file="encoding_decoding.txt" )
            return None
        

class ComponentInputEncoder(json.JSONEncoder):
    
    '''Used with component to encode its input (not the input itself)'''
    
    def __init__(self, *args, ignore_defaults : bool, source_component, respect_ignore_order, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
        self.source_component = source_component
        self.respect_ignore_order = respect_ignore_order
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            input = obj.input
            
            toReturn = {}
            
            for key in input.keys():
                
                parameter_meta : ParameterMetaData = obj.get_input_meta()[key]
                parameters_signature : ParameterSignature = parameter_meta.parameter_signature
                
                # we we're set to ignore defaults and the value set is a default value
                to_ignore_due_to_defaults =  ( not parameter_meta.was_custom_value_passed() ) and self.ignore_defaults 

                # This is due to cases where we may want to save and reload the configuration exactly as is    
                to_ignore_due_to_ignore_order = parameter_meta.ignore_at_serialization and self.respect_ignore_order
            
                if ( not to_ignore_due_to_ignore_order ) and ( not to_ignore_due_to_defaults) :
                    
                    serialized_value = json.dumps(input[key], cls=ComponentValuesElementsEncoder, source_component=self.source_component) #for each value in input, loads

                    
                    if serialized_value != 'null': # we default to not save null values, we ignore them instead
                        toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
        
        
class ComponentExposedValuesEncoder(json.JSONEncoder):
    
    '''Used with component to encode its exposed values'''
    
    def __init__(self, *args, source_component, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.source_component = source_component
    
    
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            exposed_values = obj.values
            
            toReturn = {}
            
            for key in exposed_values.keys():
                                   
                serialized_value = json.dumps(exposed_values[key], cls=ComponentValuesElementsEncoder, source_component=self.source_component) #for each value in exposed value, loads

                if serialized_value != 'null': # we default to not save null values, we ignore them instead
                    toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
            

class ComponentEncoder(json.JSONEncoder):
    
    
    '''Encodes the definition of a component, focusing on its child components'''
    
    def __init__(self, *args, ignore_defaults : bool, save_exposed_values : bool, source_component : Component, respect_ignore_order, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
        self.save_exposed_values = save_exposed_values
        self.source_component = source_component
        self.respect_ignore_order = respect_ignore_order
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            toReturn = {
                "__type__": str(type(obj)),
                "name": obj.name,
                }
            
            #load input if it exists
            component_input = json.loads(json.dumps(obj, cls= ComponentInputEncoder, ignore_defaults=self.ignore_defaults, source_component=self.source_component, respect_ignore_order=self.respect_ignore_order))
            if component_input != {}:
                toReturn["input"] = component_input

            #load notes if they exist
            component_notes = obj.get_notes()
            if component_notes != []:
                toReturn["__notes__"] = component_notes
            
            if self.save_exposed_values:
                
                if obj.values != {}:
                    toReturn["exposed_values"] = json.loads(json.dumps(obj, cls= ComponentExposedValuesEncoder, source_component=self.source_component))

            if len(obj.child_components) > 0:
                
                toReturn["child_components"] = self.default(obj.child_components)
                        
            return toReturn
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
        
        
            
        return None

def json_string_of_component_dict(component_dict : dict, ignore_defaults = True, save_exposed_values = False, respect_ignore_order = True):
    
    component_from_dict = gen_component_from_dict(component_dict)
    
    return json_string_of_component(component_from_dict, ignore_defaults=ignore_defaults, save_exposed_values=save_exposed_values, respect_ignore_order = respect_ignore_order)


# TODO: maybe it should be source_component.get_source_component()?
def json_string_of_component(component, ignore_defaults = False, save_exposed_values = False, respect_ignore_order = True, source_component=None):

    source_component = component if source_component is None else source_component

    return json.dumps(component, cls=ComponentEncoder, indent=4, ignore_defaults=ignore_defaults, save_exposed_values = save_exposed_values, source_component=source_component, respect_ignore_order=respect_ignore_order)

# DECODING --------------------------------------------------------------------------

def decode_a_component_element(source_component : Component, component_element_dict : dict):
    
    keys = component_element_dict.keys()
    
    if "localization" in keys:
            
        component_to_return = source_component.get_child_by_localization(component_element_dict["localization"])
        
    elif "name" in keys: # if it does not have localization and has name
        
        component_to_return = source_component.get_child_by_name(component_element_dict["name"])
        
    if "name" in keys:
        component_to_return.name = component_element_dict["name"]
        
    return component_to_return
    

def decode_element_custom_strategy(source_component : Component, element_dict : dict, custom_strategy):
        
    element_type_str = element_dict["__type__"]
    
    element_type = get_class_from(element_type_str)
    
    if hasattr(element_type, "from_dict"):
        
        if "object" in element_dict.keys():
            
            # from dict can also use the decoding function we're using to decode nested elements
            try:
                instanced_object = element_type.from_dict(element_dict["object"], element_type, decode_components_input_element, source_component)

            except Exception as e:
                raise Exception(f"Could not decode element of type {element_type_str} with custom strategy, using its dict: {element_dict}, due to error {e}") from e

            return instanced_object
        
        # this is a more dangerous branch as there is no separation from "__type__"
        else:
            try:
                instanced_object = element_type.from_dict(element_dict, element_type, decode_components_input_element, source_component)
            
            except Exception as e:
                raise Exception(f"Could not decode element of type {element_type_str} with custom strategy, using its dict: {element_dict}, due to error {e}") from e


            return instanced_object

    # we reach here if the element type does not have a "from_dict" method
    custom_strategy = get_custom_strategy(element_type)

    if custom_strategy is not None: # if there is a custom strategy to deal with this types

        try:
            instanced_object = custom_strategy.from_dict(element_dict["object"], element_type, decode_components_input_element, source_component)

        except Exception as e:
            raise Exception(f"Could not decode element of type {element_type_str} with custom strategy, using its dict: {element_dict}, due to error {e}") from e

        return instanced_object

    else:
        raise Exception("No from_dict method defined for type " + element_type_str)
    



    


def decode_components_input_element(source_component : Component, element):
    
    '''
    Decodes an element in a dictionary of a component (which should already be converted from json)
    One of the things it does is get a component with a localization and set it
    '''

    # these ifs are for custom strategies, when the value saved in the dictionary may not represet directly the value we want to put in the component
    
    # in the case it is a dict, it may represent a "pointer" to a component or a value with a custom decoding strategy, if it has "__type__" defined
    if isinstance(element, dict):
        keys = element.keys()

        if "__type__" in keys:
            
            class_of_element : type = get_class_from(element['__type__'])
            
            if issubclass(class_of_element, Component): #if it is a Schema
                
                return decode_a_component_element(source_component, element)
            
            else:
                return decode_element_custom_strategy(source_component, element, class_of_element)
            
        else:
            
            dict_to_return = {}
            
            for key in keys:
                
                dict_to_return[key] = decode_components_input_element(source_component, element[key])
                
            return dict_to_return
                
    
    elif isinstance(element, list):
                
        return [decode_components_input_element(source_component, value) for value in element]
    
    elif isinstance(element, str):

        if element == "__outside_component":
            raise Exception("Outside component, can't get it")
        else:
            return element
        
    elif isinstance(element, tuple):

        return tuple(decode_components_input_element(source_component, value) for value in element)
            
    else:
        return element # if there is no custom strategy, we let the value stay as is
    

def decode_components_exposed_values(component : Component, source_component : Component, component_dict : dict):
    
    if "exposed_values" in component_dict: #if there are exposed values
    
        saved_exposed_values = component_dict["exposed_values"]

        for exposed_values_key, exposed_value in saved_exposed_values.items():
            try:
                component.values[exposed_values_key] = decode_components_input_element(source_component, exposed_value)
            
            except Exception as e:
                globalWriteLine(f"WARNING: Exception while decoding exposed values for key '{exposed_values_key}': {e}", file="encoding_decoding.txt")

    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_exposed_values(child_component, source_component, component_dict["child_components"][i])
    


def decode_components_input(component : Component, source_component : Component, component_dict : dict):
    
    '''
        From a component to change
        The source_component, which is the root of the component tree
        And a dictionary which specifies the component and its children
    '''
    
    # if there was input specified in the component
    if "input" in component_dict.keys():

        input_to_pass = {}

        component_dict_input = component_dict["input"]

        for key in component_dict_input.keys():
            try:
                input_to_pass[key] = decode_components_input_element(source_component, component_dict_input[key])

            except Exception as e:
                globalWriteLine(f"WARNING: Exception while decoding input for key '{key}': {e}", file="encoding_decoding.txt")

            
        component.pass_input(input_to_pass)
    
    # for each child component, also decode input
    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_input(child_component, source_component, component_dict["child_components"][i])



def decode_components_notes(component : Component, source_component : Component, component_dict : dict):
        
    component_notes : list[str] = component_dict["__notes__"]
    
    for note in component_notes:
        component.write_line_to_notes(note, use_datetime=False) # timestamp should not be used here, as it would be false
    
    for i in range(0, len(component.child_components)): # for each child component. also decode notes
        
        child_component = component.child_components[i]
            
        decode_components_notes(child_component, source_component, component_dict["child_components"][i])
                


def decode_components_from_dict(dict : dict):
    
    '''Decodes only the components and child components from a dictionary representation of a system.'''
    
    component_type_name = dict["__type__"]
    component_name = dict["name"]
        
    component_type = get_class_from(component_type_name)
    
    component : Component = component_type()
    component.name = component_name
    
    try:
    
        child_components = dict.get("child_components") 

        if child_components != None:

            for child_dict in child_components:

                decoded_child_component = decode_components_from_dict(child_dict)

                component.define_component_as_child(decoded_child_component)
    
    except Exception as e:
        raise e
            
    return component
    
# EXPOSED DECODING METHODS ---------------------------------------------------------------------------------   

def set_values_of_dict_in_component(component : Component, dict_representation : dict):
    
    '''Receives a component with children and uses the dictionary to set its values (input, exposed values, ...)'''
    
    source_component = component.get_source_component()

    decode_components_input(component, source_component, dict_representation)
    decode_components_exposed_values(component, source_component, dict_representation)
    
    # decode notes:

    
    
def gen_component_from_special_dict(dict_representation : dict, parent_component_for_generated : Component =None) -> Component:

    to_return = None

    if parent_component_for_generated is not None and "__get_by_name__" in dict_representation.keys():

        to_return = parent_component_for_generated.look_for_component_with_name(dict_representation["__get_by_name__"])

        if to_return is None:
            globalWriteLine(f"WARNING: Tried to get component by name {dict_representation['__get_by_name__']} with parent component {parent_component_for_generated.name} but couldn't get it", file="encoding_decoding.txt")
    
    return to_return



def gen_component_from_definition_dict(dict_representation : dict, parent_component_for_generated : Component =None) -> Component:

    source_component_with_children = decode_components_from_dict(dict_representation)
    
    set_values_of_dict_in_component(source_component_with_children, dict_representation)
    
    return source_component_with_children
    
    
def gen_component_from_dict(dict_representation : dict, parent_component_for_generated : Component =None) -> Component:
    '''
    Returns a component, decoding it from a dictionary representation of a system
    
    This is both called when generating whole component trees from a configuration file or when processing Component inputs

    '''

    # we try to check if the dictionary has some special representation that should get a component
    to_return = gen_component_from_special_dict(dict_representation, parent_component_for_generated)
    
    if to_return is not None:
        return to_return
    
    # if it is not a special dictionary, we return the normal generation
    to_return = gen_component_from_definition_dict(dict_representation, parent_component_for_generated)

    if to_return is not None:
        return to_return





def dict_from_json_string(json_string) -> dict:
    '''Returns a dictionary representation of a system, decoded from a json string, without being interpreted'''
    return json.loads(json_string)


def dict_from_path(path) -> dict:

    with open(path, "r") as f:
        configuration_dict = f.read()

    return dict_from_json_string(configuration_dict)


def component_from_json_string(json_string) -> Component:
    '''Returns a component, reading it from a json_string'''
    
    dict_representation = dict_from_json_string(json_string)
        
    return gen_component_from_dict(dict_representation)


def value_from_json_string(source_component, json_string):

    '''Decodes a value from a json string, interpreting it'''

    dict_representation = dict_from_json_string(json_string)

    return decode_components_input_element(source_component, dict_representation)


def generate_component_from_class_input_definition(class_of_component, input : dict):

    '''Generate a component from its class and the input to pass'''

    class_definition = class_of_component
    class_of_component : type = get_class_with_context(class_definition, input)

    component_to_return = class_of_component(input=input)
    return component_to_return


def get_class_with_context(class_definition, input=None):
    '''
    Loads a class and may use some additional context
    '''
    to_return = get_class_from(class_definition)
    
    if TRY_CORRECT_ABSTRACT_CLASSES and input is not None and inspect.isabstract(to_return):
        globalWriteLine(f"Trying to load class {class_definition}, which is abstract, will try to look for substitute...")
        from automarlcore.advanced_component_creation import get_sub_class_with_correct_parameter_signature
        try:
            to_return = get_sub_class_with_correct_parameter_signature(to_return, input)
            globalWriteLine(f"Class was substituted by {to_return}")
        except:
            globalWriteLine(f"Did not manage to find class with correct input signature")

    return to_return
                

def is_valid_component_tuple_definition(tuple_definition):

    '''If a tuple or list could represent a component, that is, it is of form: (component_class, component_input)'''

    if isinstance(tuple_definition, (tuple, list)):

        if len(tuple_definition) == 2:

            if not isinstance(tuple_definition[1], dict):
                return False
            
            if isinstance(tuple_definition[0], type):
                return True
            
            if isinstance(tuple_definition[0], str) and is_valid_str_class_definition(tuple_definition[0]):
                return True
        
    return False



def component_from_tuple_definition(tuple_definition, context_component=None, assume_localization=True) -> Component:

    '''
    Generates a component from a tuple definition (Component_definition, input) or a localization
    '''
         
    if len(tuple_definition) > 2:
        raise Exception(f"Tuple definition has more than 2 elements, but it should have only 2, got {len(tuple_definition)}")
    
    elif len(tuple_definition) == 2:

        if isinstance(tuple_definition[1], dict): # if it can be a definition (class_def, input)

            return generate_component_from_class_input_definition(tuple_definition[0], tuple_definition[1])

        elif context_component != None: # if it can be a localization 

            to_return = get_component_by_localization(context_component, tuple_definition)

            if to_return == None:
                raise Exception(f"Could not find component, with context_component {context_component}, using tuple {tuple_definition}")
        
            return to_return

    elif assume_localization: # if we are to assume a localization in the case of badly formed arguments

        to_return = get_component_by_localization(context_component, tuple_definition)

        if to_return == None:
            raise Exception(f"Could not find component, with context_component {context_component}, using tuple {tuple_definition}")
        
        return to_return

    else:
        
        raise Exception(f"Non valid number of arguments passed to generate a component: {len(tuple_definition)}, with arguments: {tuple_definition}")
        
        
    raise Exception(f"Invalid tuple definition: {tuple_definition} as len must be 2 and is {len(tuple_definition)}")
        



def gen_component_from(definition :  Union[Component, dict, str, tuple, list], parent_component_for_generated : Component = None, input_if_generated=None) -> Component:
    
    '''Generates a component from a definition or returns it if it is already a component'''

    if isinstance(definition, Component):
        return definition
    
    else: # gen component if it was a definition
    
        if isinstance(definition, dict):
            generated_component = gen_component_from_dict(definition, parent_component_for_generated)

        elif isinstance(definition, str):

            if os.path.exists(definition): #if is path
                generated_component = gen_component_from_path(definition, parent_component_for_generated)

            else: # is json
                try:
                    generated_component = component_from_json_string(definition)

                except Exception as e:
                    raise Exception(f"Could not decode string as json and is not a path: \n{definition}\nException:\n{e}") from e


        elif isinstance(definition, tuple) or isinstance(definition, list): # if it could be a tuple definition

            generated_component =  component_from_tuple_definition(definition, parent_component_for_generated)

        else:
            raise Exception(f"Definition for key is not a Component, dict, str or tuple | list, but {type(definition)} with name {definition.__name__}")
    

        if not isinstance(generated_component, Component):
            msg_error = "Something went wrong generating component as it was generated as None"

            if parent_component_for_generated != None:
                msg_error += f", with parent_component {parent_component_for_generated}"

            raise Exception(f"{msg_error}, with definition:\n{definition}")

        if parent_component_for_generated != None and generated_component.parent_component == None:
            parent_component_for_generated.define_component_as_child(generated_component)
    
        if input_if_generated != None:
            generated_component.pass_input(input_if_generated)

        return generated_component
    
    

def gen_component_from_path(path, parent_component_for_generated : Component = None) -> Component:

    '''Gens a component from a path, either a file or a directory'''

    from automarl.utils.file_component_utils import gen_component_in_directory, gen_component_in_file_path

    if not os.path.exists(path):
        raise Exception(f"Path does not exist: {path}")
    
    elif os.path.isdir(path):
        generated_component =  gen_component_in_directory(path, parent_component_for_generated)
    
    elif os.path.isfile(path):
        generated_component =  gen_component_in_file_path(path)
    
    else:
        raise ValueError(f"Path '{path}' is neither a file nor a directory.")
    
    return generated_component
    

# OTHER METHODS ---------------------------------------------------------------------------------

def get_child_dict_from_index_localization(component_dict, localization : int) -> dict:

    '''Receives a component dict with child components and looks for the one corresponding to the int in the localization (the index of the child component)'''
    
    if "child_components" in component_dict:
        
        try:
            child_components : list = component_dict["child_components"]
            return child_components[localization]  
        
        except IndexError:
            
            raise IndexError(f"Localization index {localization} out of range for component with children: {child_components}")
            
            
    return None

def get_child_dict_from_str_localization(component_dict, localization : str) -> dict:

    '''Receives a component dict with child components and looks for the one corresponding to the string in the localization (the name of the child component)'''
    
    if "child_components" in component_dict:
        
        child_components : list = component_dict["child_components"]
        
        for child_component in child_components:
            
            if child_component["name"] == localization:
                return child_component

        return None     

    return None


def get_child_dict_from_localization(component_dict, localization) -> dict:

    '''Gets a child dictionary from either the localization or the list'''
    
    if isinstance(localization, int):
        return get_child_dict_from_index_localization(component_dict, localization)
    
    elif isinstance(localization, str):
        return get_child_dict_from_str_localization(component_dict, localization)
    
    elif isinstance(localization, list):
        
        if len(localization) == 0:
            return component_dict
                    
        elif len(localization) == 1:
            return get_child_dict_from_localization(component_dict, localization[0])
        
        else:
        
            child_component_dict = get_child_dict_from_localization(component_dict, localization[0])

            return get_child_dict_from_localization(child_component_dict, localization[1:])
    
    else:
        raise Exception(f"Localization is not an int or a str, but {type(localization)}")
    
    
