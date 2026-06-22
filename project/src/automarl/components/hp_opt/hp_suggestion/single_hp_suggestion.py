

from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
import optuna
from automarl.serialization.custom_json_logic import register_custom_strategy

class SingleHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''
        An Hyperparameter Suggestion that is a simple optuna wrapper
    '''
    
    def __init__(self, name : str = '', value_suggestion=None, hyperparameter_localizations=None):
        
        if value_suggestion is None:
            raise Exception("Value suggestion must be different than none")

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        self.value_suggestion = value_suggestion

    # SUGGESTIONS ------------------------------------    
        
    def _make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        (type_of_suggestion, kwargs) = self.value_suggestion
        
        if type_of_suggestion == 'float': 
            suggested_value = trial.suggest_float(self.name, **kwargs)
        
        elif type_of_suggestion == 'int':
            suggested_value = trial.suggest_int(self.name, **kwargs)
        
        elif type_of_suggestion == 'cat':
            suggested_value = trial.suggest_categorical(self.name, **kwargs)
        
        else:
            raise Exception(f"Invalid value suggestion with name {self.name}")            
                            
        return suggested_value


    def _try_get_suggested_value(self, component_definition, localization):

        gotten_suggestion_value = super()._try_get_suggested_value(component_definition, localization)

        if self._is_valid_suggestion_type(gotten_suggestion_value):
            return gotten_suggestion_value
        
        return None


    
    def _is_valid_suggestion_type(self, value):

        (type_of_suggestion, kwargs) = self.value_suggestion

        if type_of_suggestion == 'float' and isinstance(value, float):
            return True
        
        elif type_of_suggestion == 'int' and isinstance(value, int):
            return True
        
        elif type_of_suggestion == 'cat':
            choices = kwargs["choices"]

            return value in choices
                
        return False
    

    def already_has_suggestion_in_trial(self, trial : optuna.Trial):
        return self.get_name() in trial.params.keys()
        
    # JSON ENCODING DECODING ------------------------------------------------------------------------

    def clone(self):
        return SingleHyperparameterSuggestion(
            name=self.base_name,
            hyperparameter_localizations=self.base_hyperparameter_localizations,
            value_suggestion=self.value_suggestion
        )        

    def to_dict(self) -> dict:
        
        return {
                    **super().to_dict(),
                    "suggestion" : self.value_suggestion
            }
                    
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
        return SingleHyperparameterSuggestion(name=dict["name"], hyperparameter_localizations=dict.get("localizations", None), value_suggestion=dict["suggestion"])



register_custom_strategy(SingleHyperparameterSuggestion, SingleHyperparameterSuggestion)
