
'''
This represents a localization of a component in a component tree,
made by following the source component and its children

Total localizations are given by lists in which each element is a step of going through the current component to another

Localizations are usually stored in a component, and so there are two different types of them:
    "absolute", which goes from the source component
    "relative", which goes from the current component

    Each element of a localization can:
    An integer, to represent jump from the current component to some component at that index in the child components
    A string, to represent jump from the current component to some component with that name in the child components or their own child components
    "__up__", to represent a step to the parent component (this should only be used in a relative localization)
    A tuple (operation_str, {parameters}) with some special operation:
        ("__get_by_name__", { name_of_component })
        ("__get_exposed_value__", {value_localization})
        ("__get_input_value__", {value_localization})

for example:
    ("absolute", [1, 1])
    ("relative", ["__up__"])
'''

from collections import deque

from automarl.utils.class_util import get_class_from
from automarl.components.loggers.global_logger import globalWriteLine

# VALUE LOCALIZATION OPERATIONS --------------------------------------------------------------

def get_any(collection, default_value=None, non_exist_safe=False):

    if isinstance(collection, dict):

        if not non_exist_safe and len(collection) == 0:
            raise Exception(f"No value in collection: {collection}")

        return next(iter(collection.values()), default_value)
    
    elif isinstance(collection, (list, tuple)):

        if not non_exist_safe and len(collection) == 0:
            raise Exception(f"No value in collection: {collection}") 

        return collection[0] if collection else default_value
    
    else:

        if not non_exist_safe:
            raise Exception(f"Collection is of a non supported type: {type(collection)}")

        return default_value

def safe_get(collection_where_value_is : dict, index, default_value=None):

    '''gets the value in a collection'''

    try:

        if index == '__any__':
            return get_any(collection_where_value_is, default_value)

        return collection_where_value_is[index]
    
    except (IndexError, KeyError) as e:
            return default_value
    

def safe_general_remove(collection_where_value_is, index):
    
    try:
        return collection_where_value_is.pop(index)
    
    except (IndexError, KeyError) as e:
        return None



def get_last_collection_where_value_is(collection_where_value_is : dict, localization, default_value=None, non_exist_safe=False):

    '''Given a collection and a localization, returns the element before the last, that should be a collection'''


    return get_value_from_value_loc(collection_where_value_is, localization[:-1], default_value, non_exist_safe)


def get_value_from_value_loc(collection_where_value_is : dict, localization, default_value=None, non_exist_safe=False):

    '''Given a collection and a localization, returns the element in the localzation'''

    if not isinstance(localization, list):
        localization = [localization]

    current_value = collection_where_value_is

    for loc_index in localization:

        try:

            if loc_index == "__any__":
                current_value = get_any(current_value, default_value, non_exist_safe)

            else:
                current_value = current_value[loc_index]

        except (IndexError, KeyError) as e:
            if non_exist_safe:
                return default_value
            else:
                raise Exception(f"Error when getting collection before value, at index '{loc_index}', in localization: <{localization}> and collection {collection_where_value_is}") from e

    return current_value


# LOCALIZATION OPERATIONS -------------------------------------------------------------------- 


def get_source_component(component):
    '''Gets the source component, the one without parent'''
    
    current_component = component
    
    while True:
        
        if current_component.parent_component != None:
            current_component = current_component.parent_component
        else:
            return current_component




def get_child_by_name(component, name):
    """Gets a child component by its name using a breadth-first search."""

    queue = deque([component]) # to start the algorithm with the first component

    while queue:
        current_component = queue.popleft()

        # Check if current component matches
        if current_component.name == name:
            return current_component

        # Add all child components to the queue
        queue.extend(current_component.child_components)

    # If not found
    return None




def look_for_component_with_name(component, name):
    '''Looks a component with the specified name in the component tree'''

    source_component = get_source_component(component)

    component_with_name = get_child_by_name(source_component, name)

    return component_with_name


def component_has_correct_name(name : str, component):
    return component.name == name

def component_has_correct_type(type : type, component):
    return isinstance(component, type)

def relative_look_for_component(initial_component, is_correct_condition):

    current_component = initial_component
    already_looked_in = []

    components_to_look_for = []

    while current_component is not None:

        if is_correct_condition(current_component):
            return current_component
        
        already_looked_in.append(current_component)
            
        if current_component.parent_component is not None:
            if not current_component.parent_component in already_looked_in:
                components_to_look_for.append(current_component.parent_component)

        for child in current_component.child_components:
            if not child in already_looked_in:
                components_to_look_for.append(child)

        
        
        if len(components_to_look_for) > 0:
            current_component = components_to_look_for.pop(0)
        
        else:
            current_component = None

    return None




def get_parent_component(component):
    '''Returns the parent component of a component'''

    return component.parent_component





def get_next_component_by_tuple_operation(component, tuple_operation : tuple):

    if len(tuple_operation) != 2:
        raise Exception(f"Tuple operation in localization should be given by (operation_str, operation_parameters_dict), instead it was: {tuple_operation}")

    operation_str = tuple_operation[0]
    operation_parameters = tuple_operation[1]

    if operation_str == '__get_by_name__':
        to_return = relative_look_for_component(component, 
                                lambda x: component_has_correct_name(operation_parameters["name_of_component"],x)
                                )
        
        if to_return is None:
            raise Exception(f"Could not find component with name {operation_parameters['name_of_component']} from {component.name}")
        
        return to_return
    
    elif operation_str == '__get_by_type__':
        type_to_look = get_class_from(operation_parameters["type"])
        to_return = relative_look_for_component(component, 
                                lambda x: component_has_correct_type(type_to_look, x))
        
        if to_return is None:
            raise Exception(f"Could not find component with type {operation_parameters['type']} from {component.name}")
        
        return to_return
    
    elif operation_str == '__get_exposed_value__':

        try:
            return get_value_from_value_loc(component.values, operation_parameters["value_localization"])
        
        except Exception as e:
            raise Exception(f"Error when trying to get exposed value in component {component.name} with operation parameters {operation_parameters}:\n{e}") from e
    
    elif operation_str == '__get_input_value__':
        return get_value_from_value_loc(component.input, operation_parameters["value_localization"])
    
    else:
        raise Exception(f"Unkown tuple operation in localization: {tuple_operation}")





def get_next_component_by_str_operation(component, str_operation : str):

    if str_operation == '__up__':
        return get_parent_component(component)
    
    if str_operation == '__any__':
        component_children = component.child_components

        if len(component_children) == 0:
            raise Exception(f"Tried to get any children when component has none")
        
        else:
            return component_children[0]

    else:
        return get_child_by_name(component, str_operation)
    

    
def get_next_component_by_int_operation(component, integer : int):

    return component.child_components[integer]
    



def get_component_by_localization_list(component, localization : list):
    
    '''
    Gets child component or value by its location
    Note that an emty localization will return the component itself
    '''
    
    current_component  = component

    for operation in localization:

        if current_component == None:
            raise Exception(f"Tried to do operation on localization: <{operation}> when current component is None")
        
        if isinstance(operation, int):
            current_component = get_next_component_by_int_operation(current_component, operation)

        elif isinstance(operation, str):
            current_component = get_next_component_by_str_operation(current_component, operation)

        elif isinstance(operation, (tuple, list)):
            current_component = get_next_component_by_tuple_operation(current_component, operation)
        
        else:
            raise Exception(f"Invalid operation with type {type(operation)}, {operation} in localization {localization} for component {component.name}")
    


    return current_component




def get_component_by_localization(component, localization):

    localization_type, localization_list = interpret_localization(localization)

    try:

        if localization_type == ABSOLUTE_LOCALIZATION_STR:

            source_component = get_source_component(component)
            return get_component_by_localization_list(source_component, localization_list)

        elif localization_type == RELATIVE_LOCALIZATION_STR:

            return get_component_by_localization_list(component, localization_list)

        else:
            raise Exception("Unkown localization type")
        
    except Exception as e:

        raise Exception(f"Exception when getting value from localization <{localization}>, with localization type <{localization_type}> and localization list <{localization_list}>") from e

# CALCULATING LOCALIZATION -------------------------------------------------------------------- 


def get_index_localization(component, target_parent_components = [], accept_source_component_besides_targets=False):
        
    '''Gets localization of component, stopping the definition of localization when it finds a source component (without parent)'''
    
    current_component = component
    
    full_localization = [] # where we'll store the full localization of the component

    if not isinstance(target_parent_components, list):
        target_parent_components = [target_parent_components] #if it was only an element
                    
    while True: #while we have not reached the source component

        # if we reached a target parent component, we return the localization and whose it is from
        if current_component in target_parent_components:
            return full_localization, current_component 
                    
        elif current_component.parent_component != None:
            
            child_components_of_parent : list = current_component.parent_component.child_components
        
            index_of_this_in_parent = child_components_of_parent.index(current_component)
        
            full_localization.insert(0, index_of_this_in_parent) #inserts index
        
            current_component = current_component.parent_component
            
        else: # parent_commponent is None
            break #we reached the source component
        
    if target_parent_components != [] and not accept_source_component_besides_targets:
        raise Exception(f"Localization could not be computed from component {component.name} to one of target components {[targ_com.name for targ_com in target_parent_components]}")
    
    return full_localization, current_component 


# LOCALIZATION OBJECT DEFINItION -------------------------------------------------------


ABSOLUTE_LOCALIZATION_STR = 'absolute'
RELATIVE_LOCALIZATION_STR = 'relative'


def localization_list_from(localization):

    if isinstance(localization, list):
        return localization
    
    elif isinstance(localization, (int, str, tuple)):
        return [localization]
    
    else:
        raise Exception(f"Invalid type for creating a localization list: {type(localization)}")
    
    
def is_valid_tuple_format(localization_definition):

    return isinstance(localization_definition, (tuple, list)) and len(localization_definition) == 2

def is_valid_tuple_types(localization_definition):

    return isinstance(localization_definition[0], str) and isinstance(localization_definition[1], (list, str, int, tuple))

def is_valid_tuple_localization_definition(localization_definition):

    return is_valid_tuple_format(localization_definition) and is_valid_tuple_types(localization_definition)


def interpret_localization(localization_definition):

    if is_valid_tuple_localization_definition(localization_definition):

        localization_type = localization_definition[0]
        localization_list = localization_list_from(localization_definition[1])

    else: # if only localization definition was passed
        localization_type = ABSOLUTE_LOCALIZATION_STR
        localization_list = localization_list_from(localization_definition)

    return localization_type, localization_list
