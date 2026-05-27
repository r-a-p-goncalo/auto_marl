
from enum import Enum

from automarl.utils.smart_enum import SmartEnum

DEFAULT_GET_FROM_PARENT = False
DEFAULT_IGNORE_AT_SERIALIZATION = False
DEFAULT_PRIORITY = 50
DEFAULT_MANDATORY = True

class ParameterSignature():
    
    
    def __init__(self,
                 get_from_parent = None, 
                 default_value = None, 
                 generator = None, 
                 validity_verificator=[], 
                 possible_types =[], 
                 description=None, 
                 ignore_at_serialization=None, 
                 priority=None, 
                 on_pass=[], 
                 mandatory=None,
                 custom_dict=None):
        
        self.default_value = default_value
        self.generator = generator
        
        self.possible_types = possible_types if isinstance(possible_types, list) else [possible_types]
        
        self.description = description
        self.ignore_at_serialization = ignore_at_serialization
        self.priority = priority
        
        self.mandatory = mandatory
        self.get_from_parent = get_from_parent

        self.on_pass = on_pass if isinstance(on_pass, list) else [on_pass]
        self.validity_verificator = validity_verificator if isinstance(validity_verificator, list) else [validity_verificator]

        self.custom_dict = custom_dict

        self.child_parameter_signatures : list[ParameterSignature] = [] # a list with the input signatures that were fused with this one
        self.parent_parameter_signatures : list[ParameterSignature] = []

    
    def setup_default_values(self):
        '''
        This sets up the default values for the Input Signature, must be called right before it is ready for use

        The reason this is not done explicitly in __init__ is for the later possibility of delaying setting up default values to after all the fusions are done 
        '''

        if self.get_from_parent == None:
            self.get_from_parent = DEFAULT_GET_FROM_PARENT

        if self.ignore_at_serialization == None:
            self.ignore_at_serialization = DEFAULT_IGNORE_AT_SERIALIZATION

        if self.priority == None:
            self.priority = DEFAULT_PRIORITY

        if self.mandatory == None:
            self.mandatory = DEFAULT_MANDATORY

    def change_default_value(self, new_default_value):
        self.default_value = new_default_value

    @classmethod
    def get_value_from_input_class(cls, component_with_input, key, is_none_ok=True):

        '''Gets the value from input, returning None if it does not exist'''

        if is_none_ok:
            to_return = component_with_input.input.get(key, None)
        
        else:
            try:
                to_return = component_with_input.input[key]
            
            except KeyError as e:
                raise Exception(f"Component {component_with_input.name} tried to get mandatory input value for key '{key}' but it was not set in its input") from e


        return to_return

    def get_value_from_input(self, component_with_input, key, is_none_ok=True):

        '''Gets the value from input, returning None if it does not exist'''

        return type(self).get_value_from_input_class(component_with_input, key, is_none_ok)
    
    def fuse_with_new(self, other_input_signature):

        '''
        Creates a new ParameterSignature by fusing this one with another
        The other is treated as new, having some of its values as priority
        '''

        other_input_signature : ParameterSignature = other_input_signature

        new_default_value = self.default_value if other_input_signature.default_value == None else other_input_signature.default_value

        new_generator = self.generator if other_input_signature.generator == None else other_input_signature.generator

        new_ignore_at_serialization = self.ignore_at_serialization if other_input_signature.ignore_at_serialization == None else other_input_signature.ignore_at_serialization

        new_priority = self.priority if other_input_signature.priority == None else other_input_signature.priority


        new_mandatory = self.mandatory if other_input_signature.mandatory == None else other_input_signature.mandatory

        new_get_from_parent = self.get_from_parent if other_input_signature.get_from_parent == None else other_input_signature.get_from_parent


        new_validity_verificator = [*self.validity_verificator]

        for new_validity_verificator_fun in other_input_signature.validity_verificator:
            if not new_validity_verificator_fun in new_validity_verificator:
                new_validity_verificator.append(new_validity_verificator_fun)
            

        if other_input_signature.possible_types == None:
            new_possible_types = self.possible_types

        else:
            new_possible_types = [*self.possible_types, *other_input_signature.possible_types]
    
        if other_input_signature.description == None:
            new_description = self.description

        elif self.description == None:
            new_description = other_input_signature.description

        else: # if both have description
            new_description = f"{self.description}\n{other_input_signature.description}"


        new_on_pass = [*self.on_pass]

        for new_on_pass_fun in other_input_signature.on_pass:
            if not new_on_pass_fun in new_on_pass:
                new_on_pass.append(new_on_pass_fun)


        self.get_from_parent = new_get_from_parent
        self.default_value = new_default_value
        self.generator = new_generator
        self.validity_verificator= new_validity_verificator
        self.possible_types = new_possible_types
        self.description= new_description
        self.ignore_at_serialization= new_ignore_at_serialization
        self.priority= new_priority
        self.on_pass= new_on_pass
        self.mandatory= new_mandatory

        if self.custom_dict is not None and other_input_signature.custom_dict is not None:
            for key, value in other_input_signature.custom_dict.items():
                self.custom_dict[key] = value
        
        elif other_input_signature.custom_dict is not None:
            self.custom_dict = other_input_signature



        self.child_parameter_signatures.append(other_input_signature)
        other_input_signature.parent_parameter_signatures.append(self)


    
    def to_dict(self):

        to_dict_to_return = {

            "get_from_parent" : self.get_from_parent,
            "default_value" : self.default_value,
            "generator" : self.generator,
            "validity_verificator" : self.validity_verificator,
            "possible_types" : self.possible_types,
            "description" : self.description,
            "ignore_at_serialization" : self.ignore_at_serialization,
            "priority" : self.priority,
            "on_pass" : self.on_pass,
            "mandatory" : self.mandatory,
            "custom_dict" : self.custom_dict

        }

        return to_dict_to_return
    

    def to_str_dict(self):

        to_dict_to_return = self.to_dict()

        for key in to_dict_to_return.keys():
            to_dict_to_return[key] = str(to_dict_to_return[key])

        return to_dict_to_return
    

    def clone(self):
        return type(self)(**self.to_dict())
    

# TODO: there should be a priority parameter to define which came first
def fuse_input_signatures(first_input_signature : ParameterSignature, second_input_signature : ParameterSignature):

    if issubclass(type(first_input_signature), type(second_input_signature)):
        to_return = first_input_signature.clone()
        to_return.fuse_with_new(second_input_signature)
    
    elif issubclass(type(second_input_signature), type(first_input_signature)):
        to_return = second_input_signature.clone()
        to_return.fuse_with_new(first_input_signature)

    else:
        raise Exception(f"Tried to fuse input signatures that are not subclasses of eachother")

    return to_return



class ParameterMetaData():
    
    '''
    Represents the metadata of an input of a component
    
    Contains informations such as if it was passed, generated or is a default value
    
    '''
    
    class InputOrigin(SmartEnum):
        
        DEFAULT = 0
        GENERATED = 1
        PASSED = 2   
        
    
    def __init__(self, parameter_signature : ParameterSignature):
        
        self.origin = ParameterMetaData.InputOrigin.DEFAULT
        self.parameter_signature : ParameterSignature = parameter_signature

        self.ignore_at_serialization = parameter_signature.ignore_at_serialization
    
        self.value_got = False

    def custom_value_passed(self):
        self.origin = ParameterMetaData.InputOrigin.PASSED
        
    def custom_value_removed(self):
        self.origin = ParameterMetaData.InputOrigin.DEFAULT
        
    def default_value_was_set(self):
        self.origin = ParameterMetaData.InputOrigin.DEFAULT
        
    def generator_value_was_set(self):
        self.origin = ParameterMetaData.InputOrigin.DEFAULT
        
    def was_custom_value_passed(self):
        return self.origin == ParameterMetaData.InputOrigin.PASSED
    

    def value_was_got(self):
        self.value_got = True
    
    def was_value_got(self):
        return self.value_got
    
    def set_to_ignore_at_serialization(self, value : bool):
        self.ignore_at_serialization = value

