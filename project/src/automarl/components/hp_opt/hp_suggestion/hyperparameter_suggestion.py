
from typing import Union
from automarl.component import Component

from automarl.serialization.json_component_utils import decode_components_input_element, get_child_dict_from_localization
from automarl.serialization.custom_json_logic import CustomJsonLogic, register_custom_strategy
import optuna

from automarl.core.localizations import get_component_by_localization_list, get_last_collection_where_value_is, safe_get, safe_general_remove
from automarl.components.loggers.global_logger import globalWriteLine

class HyperparameterSuggestion(CustomJsonLogic):
    
    '''
    A class which defines a range of values a specific hyperparameter group can have

    It can have defined localizations, in which case it is able to set suggested values in those localizations or get already suggested values
    '''
    
    def __init__(self, name : str = '', hyperparameter_localizations=None):
        
        self.name = name
        self.base_name = name

        self.base_hyperparameter_localizations = hyperparameter_localizations

        if self.base_hyperparameter_localizations is not None and len(self.base_hyperparameter_localizations) > 0 and not isinstance(self.base_hyperparameter_localizations[0], list):
            self.base_hyperparameter_localizations = [self.base_hyperparameter_localizations] 

        self.hyperparameter_localizations = self.base_hyperparameter_localizations

        self.setup_names()

    # NAMES -----------------------------------------


    def get_name(self):
        '''Gets the current name of the Hyperparameter Suggestion, should be unique'''
        return self.name

    def get_base_name(self):
        return self.base_name

    def change_name(self, new_name):
        '''Changes the name and sets up any changes accross children'''
        self.name = new_name        
        self.setup_names()

    def setup_names(self):
        '''Sets up the names of child hyperparameter suggestions'''
        pass

    # SUGGESTIONS ------------------------------------

    def make_suggestion(self, trial : optuna.Trial):
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        try:
            return self._make_suggestion(trial)
        
        except Exception as e:
            raise Exception(f"Exception when making suggestion for hyperparameter with name {self.name}: \n{e}") from e


    def _make_suggestion(self, trial : optuna.Trial):
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''

    def already_has_suggestion_in_trial(self, trial : optuna.Trial):
        pass

    # LOCALIZATIONS ---------------------------------------

    def get_localizations(self):
        return self.hyperparameter_localizations
    
    def get_base_localizations(self):
        return self.base_hyperparameter_localizations
    
    def change_localizations(self, new_localizations):
        self.hyperparameter_localizations = new_localizations

    # SET VALUE IN LOCALIZATION -----------------------------------

    def set_suggested_value(self, suggested_value, component_definition : Union[Component, dict], localizations=None):
        
        '''Sets the suggested value in the component (or component input), using the localizations'''
        
        localizations = self.hyperparameter_localizations if localizations is None else localizations

        if localizations is None:
            raise Exception(f"No localizations specified in function call nor in object to set suggested value in component")

        return self._set_suggested_value(suggested_value, component_definition, localizations)


    def _set_suggested_value(self, suggested_value, component_definition : Union[Component, dict], localizations=None): 

        '''Sets the suggested value in passed localizations'''

        if isinstance(component_definition, Component):
            self._set_suggested_value_in_component(suggested_value, component_definition, localizations)
            
        elif isinstance(component_definition, (dict, list)):
            self._set_suggested_value_in_dict(suggested_value, component_definition, localizations)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}")
    
    
    def _set_suggested_value_in_component(self, suggested_value, component : Component, hyperparameter_localizations):
        
        '''Sets the suggested value in the component, using the localizations'''
    
        for hyperparameter_localizer in hyperparameter_localizations:
            
            # gets the component change 
            component_to_change : Component = get_component_by_localization_list(component, hyperparameter_localizer[:-1]) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{hyperparameter_localizer}> in {component.name}")   
            
            # we can do this because the hyperparameter is always the last element of the localization, and also we always change the whole value in the input
            component_to_change.pass_input({hyperparameter_localizer[-1] : suggested_value})



    def _set_suggested_value_in_dict(self, suggested_value, component_dict : dict, hyperparameter_localizations):
    
        '''Sets the suggested value in the dictionary representing a component, using the localizations'''

        for hyperparameter_localizer in hyperparameter_localizations:
            
            loc_where_value_is : dict = get_last_collection_where_value_is(component_dict, hyperparameter_localizer)

            loc_where_value_is[hyperparameter_localizer[-1]] = suggested_value
                


    # GET VALUE IN LOCALIZATION ---------------------------------------------------------

    def try_get_suggested_optuna_values(self, component_definition, localizations=None):

        '''Tries to get the already suggested value in a configuration'''

        localizations = self.hyperparameter_localizations if localizations is None else localizations

        if localizations is None:
            raise Exception(f"No localization specified in function call nor in object ({self.name}) to get suggested value in component")

        return self._try_get_suggested_optuna_values(component_definition, localizations)
    


    def _try_get_suggested_optuna_values(self, component_definition, localizations):

        '''Tries to get the already suggested value in a configuration'''

        value_to_return = self.try_get_suggested_value(component_definition, localizations)

        if value_to_return is None:
            return {} 
        
        return {self.get_name() : value_to_return}


    def try_get_suggested_value(self, component_definition : Union[Component, dict], localizations=None):
        
        '''Gets the suggested value in the component (or component input), using the localizations'''
        
        localizations = self.hyperparameter_localizations if localizations is None else localizations

        if localizations is None:
            raise Exception(f"No localization specified in function call nor in object ({self.name}) to get suggested value in component")

        return self._try_get_suggested_value(component_definition, localizations)


    def _try_get_suggested_value(self, component_definition : Union[Component, dict], localizations):
        
        '''Gets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            return self._try_get_already_suggested_value_in_component(component_definition, localizations)
            
        elif isinstance(component_definition, (dict, list)):
            return self._try_get_suggested_value_in_dict(component_definition, localizations)
            
        else:   
            raise Exception(f"Component definition is not a Component or a (dict | list), but {type(component_definition)}") 


    def _try_get_already_suggested_value_in_component(self, component : Component, hyperparameter_localizations):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for hyperparameter_localizer in hyperparameter_localizations:
            
            component_to_change : Component = get_component_by_localization_list(component, hyperparameter_localizer[:-1]) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{hyperparameter_localizer}> in {component.name}")   
            
            localization_suggested_value = component_to_change.get_input_value(hyperparameter_localizer[-1])

            if suggested_value != None and localization_suggested_value != suggested_value:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and localizations have different values. This is still not implemented, and the value will be treated as if it is non existent")
                return None

            if localization_suggested_value == None:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and one of localizations does not have any value. This is still not implemented, and the value will be treated as if it is non existent")
                return None
            
            if suggested_value == None:
                suggested_value = localization_suggested_value

            #if we reach here, localization suggested value exists and is equal to previous value

        return suggested_value
    


    
    def _try_get_suggested_value_in_dict(self, component_dict : dict, hyperparameter_localizations):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for hyperparameter_localizer in hyperparameter_localizations:
            
            colleciton_where_hyperparameter_is : dict = get_last_collection_where_value_is(component_dict, hyperparameter_localizer)

            localization_suggested_value = safe_get(colleciton_where_hyperparameter_is, hyperparameter_localizer[-1], None)

            if suggested_value != None and localization_suggested_value != suggested_value:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and localizations have different values. This is still not implemented, and the value will be treated as if it is non existent")
                return None

            if localization_suggested_value == None:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and one of localizations does not have any value. This is still not implemented, and the value will be treated as if it is non existent")
                return None
            
            if suggested_value == None:
                suggested_value = localization_suggested_value

            #if we reach here, localization suggested value exists and is equal to previous value

        return suggested_value



    # REMOVING VALUE --------------------------------

    def _try_remove_suggested_value(self, component_definition : Union[Component, dict], localization):
        
        '''Gets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            return self._try_remove_already_suggested_value_in_component(component_definition, localization)
            
        elif isinstance(component_definition, (dict, list)):
            return self._try_remove_suggested_value_in_dict(component_definition, localization)
            
        else:   
            raise Exception(f"Component definition is not a Component or a (dict | list), but {type(component_definition)}") 


    def _try_remove_already_suggested_value_in_component(self, component : Component, hyperparameter_localizations):
        
        '''Deletes if exists'''

        for hyperparameter_localizer in hyperparameter_localizations:
            
            component_to_change : Component = get_component_by_localization_list(component, hyperparameter_localizer[:-1]) 
            
            if component_to_change is not None:            
                component_to_change.remove_input(hyperparameter_localizer[-1])


    
    def _try_remove_suggested_value_in_dict(self, component_dict : dict, hyperparameter_localizations):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for hyperparameter_localizer in hyperparameter_localizations:
            
            colleciton_where_hyperparameter_is : dict = get_last_collection_where_value_is(component_dict, hyperparameter_localizer)

            if colleciton_where_hyperparameter_is is not None:
                safe_general_remove(colleciton_where_hyperparameter_is, hyperparameter_localizer[-1])

        
    # JSON ENCODING DECODING ------------------------------------------------------------------------

            
    def clone(self):
        return HyperparameterSuggestion(name=self.base_name, 
                                        hyperparameter_localizations=self.base_hyperparameter_localizations)

    def to_dict(self) -> dict:

        to_return = {"name" : self.base_name,
                     "__type__" : type(self)}

        if self.base_hyperparameter_localizations is not None:
            to_return["localizations"] = self.base_hyperparameter_localizations
        
        return to_return
                    
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
        return HyperparameterSuggestion(dict["name"], dict.get("localizations", None))
    

    def get_nested_hyperparameter_suggestions(self) -> list:
        '''Gets the hyperparameter suggestions that are nested in this one'''
        return [self]


register_custom_strategy(HyperparameterSuggestion, HyperparameterSuggestion)





        