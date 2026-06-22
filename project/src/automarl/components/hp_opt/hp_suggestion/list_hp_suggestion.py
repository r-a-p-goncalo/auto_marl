from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automarl.serialization.custom_json_logic import register_custom_strategy
from automarl.serialization.json_component_utils import decode_components_input_element
import optuna


class DictHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str = '', hyperparameter_suggestions : dict[str, HyperparameterSuggestion] = None, hyperparameter_localizations=None):
        
        if hyperparameter_suggestions is None:
            raise Exception("Hyperparameter suggesiton for list must be defined")

        self.hyperparameter_suggestions : dict [str, HyperparameterSuggestion] = {}

        for key, hyperparameter_suggestion in hyperparameter_suggestions.items():

            if isinstance(hyperparameter_suggestion, dict):
                self.hyperparameter_suggestions[key] = decode_components_input_element(hyperparameter_suggestion)

            else:
                self.hyperparameter_suggestions[key] = hyperparameter_suggestion.clone()
                

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        

    def change_localizations(self, new_localizations):

        super().change_localizations(new_localizations)
        
        if self.hyperparameter_localizations != None:
            self.setup_localizations()



    def setup_localizations(self):

        for hyperparameter_suggestion in self.hyperparameter_suggestions.values():
                
            disjoint_hp_sugg_choice_locs = hyperparameter_suggestion.get_localizations()

            if hyperparameter_suggestion.hyperparameter_localizations is not None:
                    
                new_locs = []

                for this_loc in self.get_localizations():
                    for choice_loc in disjoint_hp_sugg_choice_locs:
                        new_locs.append([*this_loc, *choice_loc])

                hyperparameter_suggestion.change_localizations(new_locs)

            else:
                hyperparameter_suggestion.change_localizations(self.hyperparameter_localizations)




    def setup_names(self):
        super().setup_names()
        
        for hyperparameter_suggestion in self.hyperparameter_suggestions.values():
            
            suggestion_base_name = hyperparameter_suggestion.get_base_name()

            hyperparameter_suggestion.change_name(
                f"{self.name}_{suggestion_base_name}"
            )



    def _make_suggestion(self, trial : optuna.Trial) -> dict:
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        dict_to_return = {}

        for key, hyperparameter_suggestion in self.hyperparameter_suggestions.items():

            suggestion_to_put_in_list_base_name = hyperparameter_suggestion.get_base_name()
            hyperparameter_suggestion.change_name(f"{self.name}_{suggestion_to_put_in_list_base_name}_{key}")
    
            dict_to_return[key] = hyperparameter_suggestion.make_suggestion(trial)
                
            
        return dict_to_return
    

    def already_has_suggestion_in_trial(self, trial : optuna.Trial):

        for hyperparameter_suggestion in self.hyperparameter_suggestions.values(): 

            if hyperparameter_suggestion.already_has_suggestion_in_trial(trial):
                return True
        
        return False
    
    def _try_get_suggested_optuna_values(self, component_definition, localizations):

        value_to_return = {}

        for hyperparameter_suggestion in self.hyperparameter_suggestions.values():
            value_to_return = {**value_to_return, **hyperparameter_suggestion._try_get_suggested_optuna_values(component_definition, localizations)}
    
        return value_to_return

    # JSON ENCODING DECODING ------------------------------------------------------------------------

    def clone(self):
        
        return DictHyperparameterSuggestion(
            name=self.base_name,
            hyperparameter_suggestions=self.hyperparameter_suggestions,
            hyperparameter_localizations=self.base_hyperparameter_localizations
        )

            
    def to_dict(self) -> dict:

        hyperparameter_suggestions = {}

        for key, hyperparameter_suggestion in self.hyperparameter_suggestions.items():
            hyperparameter_suggestions[key] = hyperparameter_suggestion.to_dict()
        
        return  {
            **super().to_dict(),
            "hyperparameter_suggestions" : hyperparameter_suggestions
        }
                            
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        return DictHyperparameterSuggestion(
            name=dict["name"],
            hyperparameter_localizations=dict.get("localizations", None),
            hyperparameter_suggestions=decode_elements_fun(source_component, dict["hyperparameter_suggestions"])
        )
    
    def get_nested_hyperparameter_suggestions(self) -> list[HyperparameterSuggestion]:

        to_return = super().get_nested_hyperparameter_suggestions()

        for hyperparameter_suggestion in self.hyperparameter_suggestions.values():
            to_return = [*to_return, *hyperparameter_suggestion.get_nested_hyperparameter_suggestions()]

        return to_return

        
register_custom_strategy(DictHyperparameterSuggestion, DictHyperparameterSuggestion)