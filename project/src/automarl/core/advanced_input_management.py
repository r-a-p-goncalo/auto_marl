


'''This module provides advanced input management, importing from input_management and component'''
        

from automarl.component import Component
from automarl.core.input_management import ParameterSignature
from automarl.serialization.json_component_utils import gen_component_from
from automarl.core.localizations import get_component_by_localization


CHANGE_INPUT = True

class LookableParameterSignature(ParameterSignature):

    '''An input with functionality that allows it to look for its value when it is not of one of the explicitly defined types'''


    def get_value_from_input(self, component_with_input : Component, key, is_none_ok=True, accepted_types=()):
        
        value = super().get_value_from_input(component_with_input, key, is_none_ok)

        if value == None:
            return None

        elif isinstance(value, accepted_types):
            return value

        else: # is localization

            try:
                return get_component_by_localization(component_with_input, value)
            
            except Exception as e:

                raise Exception(f"Exception when trying to get value for key {key} for component {component_with_input.name}, with assumed localization {value}: \n{e}") from e
            

    def setup_default_values(self):
        super().setup_default_values()
    


class ComponentParameterSignature(ParameterSignature):
    
    '''Abstracts the passage of components in other components inputs'''

    
    possible_types = [Component, type, dict, str, tuple, list]


    @classmethod
    def process_value_in_input(cls, component_with_input, key, value, input_if_generated=None):

        '''
        Processes the value in the input to a component
        '''
        
        component = gen_component_from(value, component_with_input, input_if_generated)

        if CHANGE_INPUT:
            component_with_input.input[key] = component

        return component
    

    

    def get_value_from_input(self, component_with_input : Component, key, is_none_ok=True, input_if_generated=None):
        
        '''Returns a component from a ComponentParameterSignature passed value'''

        value = super().get_value_from_input(component_with_input, key, is_none_ok)

        if value is None:
            return value
        
        try:
        
            return ComponentParameterSignature.process_value_in_input(component_with_input, key, value, input_if_generated)
        
        except Exception as e:

            raise Exception(f"Error when generating component from component input signature, with value {value}: {e}") from e
    
    
    

    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
        
        if "possible_types" in kwargs.keys():
            kwargs["possible_types"] = [*ComponentParameterSignature.possible_types, *kwargs["possible_types"]]

        else:
            kwargs["possible_types"] = ComponentParameterSignature.possible_types

        if default_component_definition is not None:
        
            def generator(self : Component): # will return the component to be saved in 
                
                component = gen_component_from(default_component_definition)
                self.define_component_as_child(component)
                
                return component
            
            kwargs["generator"] = generator
            
            super().__init__(**kwargs)
        
        else:
            super().__init__(**kwargs)            

            
    def setup_default_values(self):
        super().setup_default_values()




class ComponentListParameterSignature(ParameterSignature):
    
    '''Abstracts the passage of component list in other components inputs'''

    @classmethod
    def get_value_from_input_class(cls, component_with_input : Component, key, is_none_ok=True):

        list_of_components = super().get_value_from_input_class(component_with_input, key, is_none_ok)

        if list_of_components is None:
            return list_of_components
        
        to_return : list[Component] = []

        for value in list_of_components:
        
            if isinstance(value, Component):
                to_return.append(value)

            else:

                component = gen_component_from(value, component_with_input)
                to_return.append(component)

        if CHANGE_INPUT:
            component_with_input.input[key] = to_return

        return to_return

    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    

        if default_component_definition is not None:
            
            raise NotImplementedError("Default component definition is not implemented for component lists")
        
        else:
            super().__init__(**kwargs)

    def setup_default_values(self):
        super().setup_default_values()
        

            
class ComponentDictParameterSignature(ParameterSignature):
    
    '''Abstracts the passage of component list in other components inputs'''

    def get_value_from_input(self, component_with_input : Component, key, is_none_ok=True):
        
        '''Returns a component list from a ComponentListParameterSignature passed value'''
        
        dict_of_components : dict = super().get_value_from_input(component_with_input, key, is_none_ok)
        
        if dict_of_components is None:
            dict_of_components = {}

        to_return : dict[Component] = {}

        for value_key, value in dict_of_components.items():
        
            if isinstance(value, Component):
                to_return[value_key] = value

            else:
            
                component = gen_component_from(value)
                component_with_input.define_component_as_child(component)

                to_return[value_key] = component

        if CHANGE_INPUT:
            component_with_input.input[key] = to_return

        return dict_of_components
    
    
    
    def __init__(self, default_component_definition : dict = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
            
            def generator(self : Component): # will return the component to be saved in 
                
                to_return : dict[Component] = {}

                for value_key, value in default_component_definition.items():
                
                    if isinstance(value, Component):
                        to_return[value_key] = value

                    else:
                    
                        component = gen_component_from(value)
                        self.define_component_as_child(component)

                        to_return[value_key] = component
                
                return to_return
            
            kwargs["generator"] = generator
            
            super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)


    def setup_default_values(self):
        super().setup_default_values()
        
            

        
        
        
    

            
        