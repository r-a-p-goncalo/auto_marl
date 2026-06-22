

import copy
from automarl.component import Component
from automarl.core.localizations import get_last_collection_where_value_is
from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
import optuna
from automarl.serialization.custom_json_logic import register_custom_strategy

class ComplexHpSuggestion(HyperparameterSuggestion):
    
    '''
        An Hyperparameter Suggestion that changes the structure before actually making a suggestion
    '''
    
    def __init__(self, name : str = '', structure_to_add : dict =None, hyperparameter_localizations=None, actual_hyperparameter_suggestion=None):
        
        self.actual_hyperparameter_suggestion : HyperparameterSuggestion = actual_hyperparameter_suggestion
        self.structure_to_add : dict = structure_to_add

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        if self.actual_hyperparameter_suggestion.hyperparameter_localizations is None:
            raise Exception(f"Actual hyperparameter suggestion for complex should have a localization defined")



    # SUGGESTIONS ------------------------------------    

    def setup_names(self):
        super().setup_names()

        suggestion_base_name = self.actual_hyperparameter_suggestion.get_base_name()

        self.actual_hyperparameter_suggestion.change_name(f"{self.get_name()}_{suggestion_base_name}")
                

    def _make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        to_return = self.actual_hyperparameter_suggestion._make_suggestion(trial)

        return to_return


    def _set_suggested_value(self, suggested_value, component_definition, localization=None): 

        structure_to_add = copy.deepcopy(self.structure_to_add)

        self.actual_hyperparameter_suggestion.set_suggested_value(suggested_value, structure_to_add)

        super()._set_suggested_value(structure_to_add, component_definition, localization)



    def _try_get_suggested_value(self, component_definition, localization):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = super()._try_get_suggested_value(component_definition, localization) # it is the current structure

        #note we don't pass the localization
        suggested_value_in_structure = self.actual_hyperparameter_suggestion.try_get_suggested_value(suggested_value)

        return suggested_value_in_structure


    
    def already_has_suggestion_in_trial(self, trial : optuna.Trial):
        return self.actual_hyperparameter_suggestion.already_has_suggestion_in_trial(trial)


    def _try_get_suggested_optuna_values(self, component_definition, localizations):

        suggested_value = super()._try_get_suggested_value(component_definition, localizations) # it is the current structure

        if suggested_value is None:
            return suggested_value

        return self.actual_hyperparameter_suggestion.try_get_suggested_optuna_values(suggested_value)

    # JSON ENCODING DECODING ------------------------------------------------------------------------

    def clone(self):
        return ComplexHpSuggestion(
            name=self.base_name,
            structure_to_add=self.structure_to_add,
            hyperparameter_localizations=self.base_hyperparameter_localizations,
            actual_hyperparameter_suggestion=self.actual_hyperparameter_suggestion
        )        

    def to_dict(self) -> dict:
        
        return {
                    **super().to_dict(),
                    "structure_to_add" : self.structure_to_add,
                    "actual_hyperparameter_suggestion" : self.actual_hyperparameter_suggestion
            }
                    
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
        return ComplexHpSuggestion(name=dict["name"], 
                                   hyperparameter_localizations=dict.get("localizations", None),
                                   structure_to_add=dict["structure_to_add"],
                                   actual_hyperparameter_suggestion=decode_elements_fun(source_component, dict["actual_hyperparameter_suggestion"])
                                   )
    

    def get_nested_hyperparameter_suggestions(self) -> list:
        '''Gets the hyperparameter suggestions that are nested in this one'''
        return [self, *self.actual_hyperparameter_suggestion.get_nested_hyperparameter_suggestions()]



register_custom_strategy(ComplexHpSuggestion, ComplexHpSuggestion)
