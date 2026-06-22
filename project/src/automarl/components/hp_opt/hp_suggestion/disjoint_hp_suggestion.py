

from automarl.component import Component
import optuna
from automarl.components.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automarl.serialization.custom_json_logic import register_custom_strategy


class DisjointHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str = '', 
                 disjoint_hyperparameter_suggestions : list[HyperparameterSuggestion]=None,
                 hyperparameter_localizations=None,
                 allow_none=False):
        
        if disjoint_hyperparameter_suggestions is None:
            raise Exception(f"Disjoint hyperparameter suggestions must not be none")
        
        self.disjoint_hyperparameter_suggestions : dict[str, HyperparameterSuggestion] = {}
        
        for hyperparameter_suggestion in disjoint_hyperparameter_suggestions:
            self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion.name] = hyperparameter_suggestion.clone()

        self.allow_none = allow_none

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        if self.hyperparameter_localizations != None:
            self.setup_localizations()

        self.setup_names()

        self._choices : list[str] = list(self.disjoint_hyperparameter_suggestions.keys())

        if self.allow_none:
            self._choices.append("none")



    def change_localizations(self, new_localizations):

        super().change_localizations(new_localizations)
        
        if self.hyperparameter_localizations != None:
            self.setup_localizations()



    def setup_localizations(self):

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
                
            disjoint_hp_sugg_choice_locs = hyperparameter_suggestion.get_localizations()

            if hyperparameter_suggestion.hyperparameter_localizations is not None:
                    
                new_locs = []

                for disjoint_loc in self.get_localizations():
                    for choice_loc in disjoint_hp_sugg_choice_locs:
                        new_locs.append([*disjoint_loc, *choice_loc])

                hyperparameter_suggestion.change_localizations(new_locs)

            else:
                hyperparameter_suggestion.change_localizations(self.hyperparameter_localizations)


    def setup_names(self):
        super().setup_names()
        
        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
                        
            suggestion_base_name = hyperparameter_suggestion.get_base_name()

            hyperparameter_suggestion.change_name(
                f"{self.name}_{suggestion_base_name}"
            )


    def _set_suggested_value(self, suggested_value, component_definition, localization=None) :

        (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use) = suggested_value

        if self.allow_none and hyperparameter_suggestion_to_use_name == 'none':
            self._try_remove_suggested_value(component_definition, localization)
            
        else:
            self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]._set_suggested_value(suggested_value_of_suggestion_to_use, component_definition, localization)

    

        
    def _make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        hyperparameter_suggestion_to_use_name = trial.suggest_categorical(self.get_name(), choices=self._choices)

        if hyperparameter_suggestion_to_use_name == 'none':
            return ('none', None)
        
        else:

            hyperparameter_suggestion_to_use = self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]

            suggested_value_of_suggestion_to_use = hyperparameter_suggestion_to_use.make_suggestion(trial)
                     
            return (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use)



    def _try_get_suggested_value(self, component_definition, localization):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            try:
                suggested_value = hyperparameter_suggestion._try_get_suggested_value(component_definition, localization)
            
            except Exception as e:
                suggested_value = None

            if suggested_value != None:
                suggested_value = (hyperparameter_suggestion.get_base_name(), suggested_value)
                break

        if self.allow_none and suggested_value is None:
            return ('none', None)
        
        else:
            return suggested_value # it is None, not found
        

    
    def _try_get_suggested_optuna_values(self, component_definition, localizations):

        value_to_return = self.try_get_suggested_value(component_definition, localizations)

        if value_to_return is None:
            return {} 
        
        (cat_choice, _) = value_to_return

        if cat_choice == "none":
            return {self.get_name() : "none"}
    
        else:
            return {self.get_name() : cat_choice, **self.disjoint_hyperparameter_suggestions[cat_choice].try_get_suggested_optuna_values(component_definition, localizations)}
    

    def already_has_suggestion_in_trial(self, trial : optuna.Trial):
        return self.get_name() in trial.params.keys()
    
    # JSON ENCODING DECODING ------------------------------------------------------------------------
            

    def clone(self):

        return DisjointHyperparameterSuggestion(
            name = self.base_name,
            disjoint_hyperparameter_suggestions = [disjoint_hyperparameter_suggestion.clone()  for disjoint_hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()],
            hyperparameter_localizations=self.hyperparameter_localizations,
            allow_none=self.allow_none
        )
            
    def to_dict(self) -> dict:
        
        
        return  {

            **super().to_dict(),
            "suggestions" : [
                hyperparameter_suggestion for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()
                ],
            "allow_none" : self.allow_none
            
        }
                            
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        suggestion_dicts =  [ hyperparameter_suggestion_dict for hyperparameter_suggestion_dict in dict["suggestions"] ]

        disjoint_to_return = DisjointHyperparameterSuggestion(
            name=dict["name"], 
            hyperparameter_localizations=dict.get("localizations", None),
            disjoint_hyperparameter_suggestions=decode_elements_fun(source_component, suggestion_dicts), # it is done this way to be able to deal with nested components
            allow_none=dict["allow_none"]
        )

        return disjoint_to_return
    
    def get_nested_hyperparameter_suggestions(self) -> list[HyperparameterSuggestion]:

        to_return = super().get_nested_hyperparameter_suggestions()

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            to_return = [*to_return, * hyperparameter_suggestion.get_nested_hyperparameter_suggestions()]

        return to_return
        
register_custom_strategy(DisjointHyperparameterSuggestion, DisjointHyperparameterSuggestion)