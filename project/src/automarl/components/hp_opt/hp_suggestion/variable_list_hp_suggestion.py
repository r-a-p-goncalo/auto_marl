from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automarl.serialization.custom_json_logic import register_custom_strategy
from automarl.serialization.json_component_utils import decode_components_input_element
import optuna


class VariableListHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str = '', min_len=1, max_len=None, hyperparameter_suggestion_for_list : HyperparameterSuggestion = None, hyperparameter_localizations=None):
        
        if max_len is None:
            raise Exception("Max len must be defined")
        
        if hyperparameter_suggestion_for_list is None:
            raise Exception("Hyperparameter suggesiton for list must be defined")

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        self.name_len_list = f"{self.name}_len"
        self.min_len = min_len
        self.max_len = max_len

        self.hyperparameter_suggestion_for_list : HyperparameterSuggestion =  hyperparameter_suggestion_for_list.clone()

        if isinstance(self.hyperparameter_suggestion_for_list, dict):
            self.hyperparameter_suggestion_for_list = decode_components_input_element(self.hyperparameter_suggestion_for_list)



    def _make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        list_len= trial.suggest_int(self.name_len_list, low=self.min_len, high=self.max_len)

        list_to_return = []

        for i in range(list_len):

            suggestion_to_put_in_list = self.hyperparameter_suggestion_for_list

            suggestion_to_put_in_list_base_name = suggestion_to_put_in_list.get_base_name()

            suggestion_to_put_in_list.change_name(f"{self.name}_{suggestion_to_put_in_list_base_name}_{i}")

            list_to_return.append(suggestion_to_put_in_list.make_suggestion(trial))
                
            
        return list_to_return
    

    def _try_get_suggested_optuna_values(self, component_definition, localizations):

        suggested_values = self.try_get_suggested_value(component_definition, localizations)

        if suggested_values is None:
            return {}

        to_return = {self.name_len_list : len(suggested_values)}

        for index in range(len(suggested_values)):

            suggestion_to_put_in_list_base_name = self.hyperparameter_suggestion_for_list.get_base_name()

            self.hyperparameter_suggestion_for_list.change_name(f"{self.name}_{suggestion_to_put_in_list_base_name}_{index}")

            to_return = {
                         **to_return,
                         **self.hyperparameter_suggestion_for_list.try_get_suggested_optuna_values(suggested_values, [[index]])
                        }      

        
        return to_return
    

    def already_has_suggestion_in_trial(self, trial : optuna.Trial):
        return self.name_len_list in trial.params.keys()


    
    # JSON ENCODING DECODING ------------------------------------------------------------------------

    def clone(self):
        
        return VariableListHyperparameterSuggestion(
            name=self.base_name,
            min_len=self.min_len,
            max_len=self.max_len,
            hyperparameter_suggestion_for_list=self.hyperparameter_suggestion_for_list,
            hyperparameter_localizations=self.base_hyperparameter_localizations
        )

            
    def to_dict(self) -> dict:
        
        return  {
            **super().to_dict(),
            "min_len" : self.min_len,
            "max_len" : self.max_len,
            "hyperparameter_suggestion_for_list" : self.hyperparameter_suggestion_for_list.to_dict()
        }
                            
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        return VariableListHyperparameterSuggestion(
            name=dict["name"],
            hyperparameter_localizations=dict.get("localizations", None),
            min_len=dict["min_len"],
            max_len=dict["max_len"],
            hyperparameter_suggestion_for_list=decode_elements_fun(source_component, dict["hyperparameter_suggestion_for_list"])

        )
    
    def get_nested_hyperparameter_suggestions(self) -> list[HyperparameterSuggestion]:

        return [*super().get_nested_hyperparameter_suggestions(), *self.hyperparameter_suggestion_for_list.get_nested_hyperparameter_suggestions()]


        
register_custom_strategy(VariableListHyperparameterSuggestion, VariableListHyperparameterSuggestion)