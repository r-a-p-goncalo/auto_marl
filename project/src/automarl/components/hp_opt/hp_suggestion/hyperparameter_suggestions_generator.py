
from automarl.component import Component

from automarl.utils.json_utils.json_component_utils import decode_components_input_element, is_valid_component_tuple_definition

from automarl.utils.class_util import get_class_from
from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion


def generate_hyperparameter_suggestion_list_for_config_dict(config_dict):

    return _gen_hp_suggestions_for_collection(config_dict, current_localization=[])



def _gen_hp_suggestions_for_collection(collection, current_localization):

    if isinstance(collection, dict):
        return _gen_hp_suggestions_for_config(collection, current_localization)
    
    elif isinstance(collection, list):
        return _gen_hp_suggestions_for_list(collection, current_localization)
    
    else:
        raise Exception(f"Invalid type to generate hyperparameter suggestions")


def _gen_hp_suggestions_for_list(config_list : list, current_localization):

    to_return = []

    for i in range(len(config_list)):

        value = config_list[i]

        new_localization = [*current_localization, i]

        to_return = gen_hp_suggestions_for_new_localization_and_value(to_return, new_localization, value)

    return to_return


def _gen_hp_suggestions_for_config(config_dict : dict, current_localization):

    to_return = []

    for key, value in config_dict.items():

        new_localization = [*current_localization, key]

        to_return = gen_hp_suggestions_for_new_localization_and_value(to_return, new_localization, value)


    return to_return

def gen_hp_suggestions_for_new_localization_and_value(suggestions, new_localization, value):
        
        if isinstance(value, Component):
            return [ *suggestions,
                *gen_hp_suggestions_for_instanced_component(value, new_localization)]
            
        if is_valid_component_tuple_definition(value):
            return [ *suggestions,
                *gen_hp_suggestions_for_tuple_definition(value, new_localization)]

        if isinstance(value, dict) and "__type__" in value.keys():

            value_type = get_class_from(value["__type__"])
            
            if issubclass(value_type, Component):
                
                return [*suggestions, 
                        *gen_hp_suggestion_for_dict_definition(
                            value_type,
                            value,
                            new_localization
                        )
                ]

        if isinstance(value, (dict, list)):

            return [*suggestions, *_gen_hp_suggestions_for_collection(value, new_localization)]

        return suggestions

def gen_hp_suggestions_for_instanced_component(component : Component, current_localization):

    class_definition = type(component)
    input_of_component = component.input

    new_localization = [*current_localization, "__get_input_value__"]

    return [
            *gen_hp_suggestions_for_component_and_input(
                                class_definition, 
                                input_of_component,
                                new_localization
                            ),
            
             *_gen_hp_suggestions_for_collection(input_of_component, new_localization)
                            
            ]

def gen_hp_suggestion_for_dict_definition(class_definition : type[Component], dict_def : dict, current_localization):

    input_of_component = dict_def["input"]
    child_components = dict_def.get("child_components", {})


    suggestions_for_component = gen_hp_suggestions_for_component_and_input(
        class_definition,
        input_of_component,
        [*current_localization, "input"]
    )

    suggestions_following_input = _gen_hp_suggestions_for_collection(
        input_of_component,
        [*current_localization, "input"]
    )

    suggestions_following_child_components = _gen_hp_suggestions_for_collection(
        child_components,
        [*current_localization, "child_components"]
    )

    return [
        *suggestions_for_component,
        *suggestions_following_input,
        *suggestions_following_child_components
    ]



def gen_hp_suggestions_for_tuple_definition(tuple_definition, current_localization):

    (class_definition, input_of_component) = tuple_definition
    class_definition : type[Component] = get_class_from(class_definition)

    new_localization = [*current_localization, 1]


    return [
            *gen_hp_suggestions_for_component_and_input(
                                class_definition, 
                                input_of_component,
                                new_localization
                            ),
            
             *_gen_hp_suggestions_for_collection(input_of_component, new_localization)
                            
            ]



def gen_hp_suggestions_for_component_and_input(component_class : type[Component], input, current_localization):

    to_return = []

    for key, value in component_class.get_schema_parameters_signatures().items():

        if value.custom_dict != None:
        
            hyperparameter_suggestion_in_parameter_signature = value.custom_dict.get("hyperparameter_suggestion", None)
    
            if hyperparameter_suggestion_in_parameter_signature is not None:
                to_return.append(
                    gen_hp_suggestion_for_parameter_schema(hyperparameter_suggestion_in_parameter_signature, [*current_localization, key])
                )

    return to_return



def gen_hp_suggestion_for_parameter_schema(hyperparameter_suggestion, current_localization):

    if isinstance(hyperparameter_suggestion, HyperparameterSuggestion):

        to_return = hyperparameter_suggestion.clone()

        if to_return.get_localizations() == None: 
            to_return.base_hyperparameter_localizations = [current_localization]

        return to_return
    
    elif isinstance(hyperparameter_suggestion, dict) and ("__type__" in hyperparameter_suggestion.keys() or "name" in hyperparameter_suggestion.keys()):

        # guarantee hp suggestion has type
        hp_suggestion_type : type[HyperparameterSuggestion] = get_class_from(hyperparameter_suggestion.get("__type__", HyperparameterSuggestion))
        hyperparameter_suggestion["__type__"] = hp_suggestion_type

        # guarantee hp suggestion has name
        hp_suggestion_name : str = hyperparameter_suggestion.get("name", current_localization[len(current_localization) - 1])
        hyperparameter_suggestion["name"] = hp_suggestion_name

        return decode_components_input_element(hyperparameter_suggestion)
    
    else:
        hyperparameter_name = current_localization[len(current_localization) - 1]
        hyperparameter_localizations = [current_localization]
        value_suggestion = hyperparameter_suggestion

        return SingleHyperparameterSuggestion(name=hyperparameter_name, hyperparameter_localizations=hyperparameter_localizations, value_suggestion=value_suggestion)

