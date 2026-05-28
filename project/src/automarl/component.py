from automarl.core.input_management import ParameterMetaData, ParameterSignature
from types import FunctionType
import copy

from automarl.core.localizations import get_child_by_name, get_component_by_localization, get_index_localization, get_source_component

from automarl.utils.class_util import get_class_from
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.schema import Schema

# Reserved attributes: input, values, parameters_signature, exposed_values, output, _input_was_processed

def on_name_pass(self):
    self.name = self.input["name"] #sets the name of the component to the input name
    self._was_custom_name_set = True

def on_child_components_passed(self):
    self._process_passed_child_components()

INPUT_LOGGER_FILE='input_stuff.txt'

class Component(metaclass=Schema): # a component that receives and verifies input

    '''
    This is the basic Squema, all other Squemas should extend it
    All its defiined values are still processed by the metaclass Schema
    '''
    
    parameters_signature : dict[str, ParameterSignature] = {
        "name" : ParameterSignature(on_pass=on_name_pass, ignore_at_serialization=True, mandatory=False, priority=0), #the name of the component
        "child_components" : ParameterSignature(mandatory=False)
    }

    original_parameters_signature : dict[str, ParameterSignature] = {} # this is not for users to change, as the schema automatically stores here the original parameters_signature as it was defined
    fused_parameters_signature : dict[str, ParameterSignature] = {} # this is not for users to change, as the schema saves here the fused parameter signatures without default values

    is_debug_schema = False # meant to be True if a class is to be used as debug, this means it will always be counted last in the mro
    target_debug_class = None

    default_name = None

    #A dictionary { "value_name" -> initial_value }
    #it tells other components what are the values exposed by this component, useful when checking the validity of the program before running it
    #this does not need to be stored in the component, the only reason it is is to standardize this kind of exposure
    exposed_values : dict[str, any] = {}

    organized_parameters_signatures : dict[int, list[ParameterSignature]] = {} #ParameterSignatures organized by priorities
    
    
    # INITIALIZATION -------------------------------------------------------------------------
    
    def __init__(self, input : dict[str, any] = None): #we can immediatly receive the input
                        
        self.input : dict[str, any] = {} #the input will be a dictionary
        
        self.__input_meta : dict[str, ParameterMetaData] = {} # will store meta data according to the input
        self.__initialize_input_meta_data()
        
        self.values = {**type(self).exposed_values} #this is where the exposed values will be stored
        
        self.child_components : list[Component] = []
        self.parent_component : Component = None
        
        self.name = type(self).default_name
        self._was_custom_name_set = False
        
        self.output = {} #output, if any, will be a dictionary
        
        self.__input_was_processed = False #to track if the instance has had its input processing before any operations that needed it
        self.__input_is_being_processed = False
        self.__did_essential_input_processing = False

        self.__notes = [] #notes are a list of strings

        if input != None:
            self.pass_input(input) #passes the input but note that it does not process it


    
    # NAMING STUFF --------------------------------------------------------------------------------------------------

    def has_custom_name_passed(self):
        return self._was_custom_name_set
                
            
    # PASSING AND CHANGING INPUT --------------------------------------------------------------------------------------------------
    
    def pass_input(self, input: dict): # pass input to this component, may need verification of input
        '''Pass input to this component
           It is supposed to be called only before process_input and, if not, it mean process_input may be called a multitude of times
           '''
        
        if not isinstance(input, dict):
            raise Exception(f"In component {self.name} of type {type(self)}: Passed input is not of type dict")
           
        self.__input_was_processed = False #when we pass new input, it means that we need to process it again
        
        for passed_key in input.keys():
            
            parameter_signature = self.get_parameter_signature(passed_key)
            
            if parameter_signature != None:
                
                self.__verified_pass_input(passed_key, input[passed_key])
                
            else:
                globalWriteLine(f"WARNING: input with key {passed_key} passed to component {self.name} but not in its input signature, will be ignored", file=INPUT_LOGGER_FILE)
        

    def pass_input_if_no_value(self, key, value): # pass input to this component, may need verification of input
        '''
        Passed the value if no custom value was passed
        '''
        
        parameter_signature = self.get_parameter_signature(key)
            
        if parameter_signature != None:
            
            if not self.__input_meta[key].was_custom_value_passed():        
        
                self.__verified_pass_input(key, value)
                
        else:
            globalWriteLine(f"WARNING: input with key {key} passed to component {self.name} but not in its input signature, will be ignored", file=INPUT_LOGGER_FILE)
                
                   
        
                
    def setup_default_value_if_no_value(self, key):

        '''
        Internally sets up the default value for an input if it has no default value
        '''
        
        parameter_signature = self.get_parameter_signature(key)
            
        if parameter_signature != None:
            
            if not self.__input_meta[key].was_custom_value_passed():        
        
                self.__verified_setup_default_value(key)
                
        else:
            globalWriteLine(f"WARNING: input with key {key} passed to component {self.name} but not in its input signature, will be ignored", file=INPUT_LOGGER_FILE)
                
                   

    def setup_default_value(self, key):
                
        parameter_signature = self.get_parameter_signature(key)
            
        if parameter_signature != None:
                
            self.__verified_setup_default_value(key)
                
        else:
            globalWriteLine(f"WARNING: input with key {key} passed to component {self.name} but not in its input signature, will be ignored", file=INPUT_LOGGER_FILE)
                
    
    def remove_input(self, key):

        '''Removes input value from component'''
           
        self.__some_updated_input() #when we pass new input, it means that we need to process it again
        
        parameter_signature = self.get_parameter_signature(key)
        
        if parameter_signature != None:
            
            self.__verified_remove_input(key)
            
        else:
            globalWriteLine(f"WARNING: input with key {key} tried to remove from component {self.name} but not in its input signature, will be ignored", file=INPUT_LOGGER_FILE)
        
        
    def __some_updated_input(self): # some input was changed
        self.__input_was_processed = False
                
    def __verified_pass_input(self, key, value):
        
        '''for logic of passing the input, already verified'''
        
        self.input[key] = value

        try:
            self.__input_meta[key].custom_value_passed()
        
        except Exception as e:
            raise Exception(f"In component {self.name}, of type {type(self)}, when passing input: Tried to pass input for key {key}, with internal inconsistency, as it does not exist in the meta of inputs: {self.__input_meta.keys()} even if exists in parameter signature: {self.get_parameters_signatures()})") from e

        
        for on_pass in self.get_parameter_signature(key).on_pass:

            try:
                on_pass(self)

            except:
                raise Exception(f"In component {self.name}, of type {type(self)}, when passing input: Exception while using the on_pass for {key}, named {on_pass.__name__}")
               
               
    def __verified_setup_default_value(self, key): 
            
        parameter_signature = self.get_parameter_signature(key)
        
        if parameter_signature.default_value != None:
            self.input[key] = parameter_signature.default_value
            self.__input_meta[key].default_value_was_set()
        
        elif parameter_signature.generator != None:

            self.input[key] = parameter_signature.generator(self)
            self.__input_meta[key].generator_value_was_set()
            
        else:
            self.input[key] = None
            #raise Exception(f"In component {self.name}, of type {type(self)}, when setting default value for {key}: No default value or generator defined for this key")



    def __verified_remove_input(self, key):
        
        '''for logic of passing the input, already verified'''
        
        self.input[key] = None
        self.__input_meta[key].custom_value_removed()

    
    def _get_input_value_from_input(self, key, **kwargs):

        '''
        This is the method meant to get input from the component
        It generalizes using the method giving from the ParameterSignature of the specific key
        '''

        try:

            parameter_signature : ParameterSignature = self.get_parameter_signature(key)
           
            to_return = parameter_signature.get_value_from_input(self, key, **kwargs)

            self.__input_meta[key].value_was_got()

            return to_return
        
        except Exception as e:

            raise Exception(f"Error getting input value from component {self.name} with key '{key}' and parameters {kwargs}: {e}") from e


    def _try_look_input_in_attribute(self, input_key, attribute_name):

        if hasattr(self, attribute_name):
            return getattr(self, attribute_name)
        else:
            return None

    def _try_look_input_in_values(self, input_key, value_name):

        return self.values.get(value_name, None)
 


    def get_input_value(self, key, look_in_value_with_key=None, look_in_attribute_with_name=None, **kwargs):

        '''
        This is the method meant to get input from the component
        It generalizes using the method giving from the ParameterSignature of the specific key
        '''

        to_return = None

        if look_in_attribute_with_name != None:
            to_return = self._try_look_input_in_attribute(key, look_in_attribute_with_name)
            if to_return != None:
                return to_return

        if look_in_value_with_key != None:
            to_return = self._try_look_input_in_values(key, look_in_value_with_key)
            if to_return != None:
                return to_return

        return self._get_input_value_from_input(key, **kwargs)

    
    def set_input_to_be_ignored_at_serialization(self, key : str, value : bool):

        self.__input_meta[key].set_to_ignore_at_serialization(value)

    def was_custom_value_passed_for_input(self, key):

        return self.__input_meta[key].was_custom_value_passed()
    

    def get_attr_or_input(self, key):

        '''
        Gets an attribute with this name from a component or from the input if it does not have one
        '''

        if hasattr(self, key):
            return getattr(self, key)
        
        else:
            return self.get_input_value(self, key)


    # process INPUT ---------------------------------------------------------------------------------------


    def _process_input_internal(self): #verify the input to this component
        '''
        Verify validity of input and add default values, following initializing the attributes
        This can and should be extended by child Schemas
        ''' 
        
        self.__input_is_being_processed = True
        
        passed_keys = self.input.keys()
        
        for priority in type(self).parameters_signature_priorities:
                        
            self.__add_default_values_of_class_input(passed_keys, type(self).organized_parameters_signatures[priority])
            
            self.__verify_input(passed_keys, type(self).organized_parameters_signatures[priority])     

        self._process_passed_child_components()


    def _process_passed_child_components(self):

        passed_child_components = self.get_input_value("child_components")

        if passed_child_components is not None:
            from automarl.core.advanced_input_management import ComponentListParameterSignature
            passed_child_components = ComponentListParameterSignature.get_value_from_input_class(self, "child_components", is_none_ok=True)

            for passed_child_component in passed_child_components:
                if passed_child_component.parent_component != self:
                    self.define_component_as_child(passed_child_component) 
        
        self.input.pop("child_components", None)

        
    def process_input(self):
        '''
        This method is called by external parties
        Instead of extending it, extend process_input_internal
        '''
        self._process_input_internal()
        self.__input_was_processed = True
        self.__input_is_being_processed = False
        self._post_process_input()
        
    
    def _post_process_input(self):
        '''Called after the input was processed, to do any post processing necessary'''
        pass

    
    def input_was_processed(self):
        return self.__input_was_processed and not self.__input_is_being_processed
    
        
    def process_input_if_not_processed(self): 
           
        if not self.__input_was_processed:
            
            if self.__input_is_being_processed:
                raise Exception(f"In component {self.name}, of type {type(self)}, when cheking for the inputs: Input is already being processed, there is probably a recursive call to process_input")
            
            self.process_input()


    # OUTPUT ---------------------------------------------------------------------
    
    def get_output(self): #return output
        return self.output
    
    # CHILD AND PARENT COMPONENTS AND LOCALIZATION ---------------------------------
    
    def initialize_child_component(self, component_type : type, input : dict = None):
        
        '''Explicitly initializes a component of a certain type as a child component of this one'''
        
        component_type = get_class_from(component_type)
        
        initialized_component : Component = component_type(input)
        
        self.define_component_as_child(initialized_component)
        
        return initialized_component
    
    
    def define_component_as_child(self, new_child_component):
        
        '''Defines target component as being child component of this one'''

        if new_child_component.parent_component == self:
            raise Exception("Tried to add a child component that is already a child of this component")

        else:
            self.child_components.append(new_child_component)

            new_child_component.parent_component = self
        
            new_child_component._on_parent_component_defined()
        
    def _on_parent_component_defined(self):
        '''called after this component as another parent component defined'''
        pass
        
    
    def get_attr_from_parent(self, attr_name : str):
        '''Gets an attribute from a parent component, None if non existent'''
                
        if self.parent_component == None:
            return None
        
        else:
            value_in_parent_input = self.parent_component.input.get(attr_name)

            if value_in_parent_input is not None:
                return value_in_parent_input
            
            elif hasattr(self.parent_component, attr_name):
                return getattr(self.parent_component, attr_name)
            
            else:
                return self.parent_component.get_attr_from_parent(attr_name)
            
    
    def get_child_by_name(self, name):
        
        '''Gets child component, looking for it by its name'''
                
        return get_child_by_name(self, name)
    

    def get_child_by_localization(self, localization : list):
        
        '''
        Gets child component by its location
        Note that an emty localization will return the component itself
        '''
        
        return get_component_by_localization(self, localization)
                
    
    def get_index_localization(self, target_parent_components = [], accept_source_component_besides_targets=False):
        
        '''Gets localization of this component, stopping the definition of localization when it finds a source componen (without parent)'''
        
        return get_index_localization(self, target_parent_components, accept_source_component_besides_targets)
    
    
    def get_source_component(self):
        '''Gets the source component, the one without parent'''
        
        return get_source_component(self)
            

    # CLONING -------------------------------------------------

    def _input_to_clone(self):
        '''Defines what input this component clones into a new one'''
        return {**self.input}
    
    def _values_to_clone(self):
        '''Defines what exposed values this component clones into a new one'''
        return {**self.values}

    def _output_to_clone(self):
        '''Defines what output this component clones into a new one'''
        return {**self.output}
    
    def _clone(self, input_for_clone, is_deep_clone):
        '''Returns an instance with the values but non of the after process'''

        input_to_clone = {**self._input_to_clone(), **input_for_clone}
        values_to_clone = self._values_to_clone()
        output_to_clone = self._output_to_clone()

        to_return : Component = type(self)(input_to_clone)

        to_return.values = values_to_clone
        to_return.output = output_to_clone
        to_return.__notes = self.__notes # copy the notes

        return to_return
    
    def _after_clone(self, original, is_deep_clone):
        '''What to do after cloning the instance, this can have some input processing for example'''

    def clone(self, save_in_parent=True, input_for_clone=None, is_deep_clone=True):
        
        '''
        Creates a clone of this component, with the same input and exposed values.
        Note that the clone will not have the same parent component
        '''

        #self.process_input_if_not_processed()

        input_for_clone = {} if input_for_clone is None else input_for_clone    

        cloned_component : Component = self._clone(input_for_clone, is_deep_clone)

        if save_in_parent and self.parent_component != None:
            self.parent_component.define_component_as_child(cloned_component) # the cloned component is also child of same parent
        
        if input_for_clone != None:
            cloned_component.pass_input(input_for_clone)

        cloned_component._after_clone(self, is_deep_clone)

        return cloned_component
    
    # EXPOSED VALUES -------------------------------------------
    

    @classmethod
    def get_schema_exposed_values(cls) -> dict[str, ParameterSignature]:
        
        '''Returns the parameters signatures for this Schema'''

        return cls.exposed_values
    
    def set_value_with_initial(self, value_key):

        self.values[value_key] = self.exposed_values[value_key]

    # INPUT processING ---------------------------------------------      
    
    
    
    @classmethod
    def get_schema_parameter_signature(cls, key) -> ParameterSignature:
        
        '''
        Returns the parameter signature for a key for this Schema
        None if non existant
        '''

        return cls.parameters_signature.get(key, None)
    

    @classmethod
    def get_schema_parameters_signatures(cls) -> dict[str, ParameterSignature]:
        
        '''Returns the parameters signatures for this Schema'''

        return cls.parameters_signature



    
    

    def get_parameter_signature(self, key) -> ParameterSignature:
        
        '''Gets the parameter signature for a key for this component'''
        
        return type(self).get_schema_parameter_signature(key) #uses the class method to get the parameter signature for a key
    


    def get_parameters_signatures(self) -> dict[str, ParameterSignature]:
        
        '''Returns the parameters signatures for this component'''
        
        return type(self).get_schema_parameters_signatures() #uses the class method to get the parameter signature 
    

    
    def in_parameters_signature(self, key): #checks if the key is in input signature of component or its parents components
        
        return self.get_parameter_signature(key) != None
    
        
            
    # DEFAULT VALUES -------------------------------------------------
            
    def __add_default_values_of_class_input(self, passed_keys, list_of_signatures : dict[str, ParameterSignature]):
                        
        for input_key, parameter_signature in list_of_signatures.items():

            value_to_put = None
                             
            #if this values was not already defined
            if not input_key in passed_keys: 
                
                if parameter_signature.get_from_parent:
                    value_to_put = self.get_attr_from_parent(input_key)
                
                if value_to_put is None:
                                                                   
                    if parameter_signature.default_value is not None:
                        value_to_put = parameter_signature.default_value  #the value used will be the default value


                    elif parameter_signature.generator is not None:
                        try:
                            value_to_put = parameter_signature.generator(self) #generators have access to the instance        
                        
                        except Exception as e:
                            raise Exception(f"In component {self.name}, of type {type(self)}, when cheking for the inputs: Exception while using the generator for {input_key}, named {parameter_signature.generator.__name__}:\n{e}") from e
            
            if value_to_put is not None:
                self.input[input_key] = value_to_put
    
    # INPUT META DATA ----------------------------------------------------------------
    
    def __initialize_input_meta_data(self):
        
        '''Initializes the dictionary __input_meta'''
            
        for key in type(self).parameters_signature.keys():
            self.__input_meta[key] = ParameterMetaData(parameter_signature=type(self).parameters_signature[key])
        

            
            
    def get_input_meta(self) -> dict[str, ParameterMetaData]:
        return self.__input_meta
        
    # VALIDITY VERIFICATION ---------------------------------------------

    def __verify_input(self, passed_keys, list_of_signatures : dict[str, ParameterSignature]): #verify the input to this component and add default values, to the self.input dict

        for input_key, parameter_signature in list_of_signatures.items():
                                    
            if input_key in passed_keys: #if this value was in input
                
                input_value = self.input[input_key] #get the value passed

                if input_value is None and parameter_signature.mandatory == True:
                    raise Exception(f"In component {self.name}, of type {type(self)}, when cheking for the inputs: Did not set input for mandatory key '{input_key}' and has no default value nor generator\n but put for {passed_keys}")    
                
                elif input_value is not None:
                    self.verify_validity(input_key, input_value, parameter_signature.possible_types, parameter_signature.validity_verificator) #raises exceptions if input is not valid
                                    
            elif parameter_signature.mandatory: #if there was no specified value for this attribute in the input
                raise Exception(f"In component {self.name}, of type {type(self)}, when cheking for the inputs: Did not set input for mandatory key '{input_key}' and has no default value nor generator\n but put for {passed_keys}")     


    def verify_validity(self, input_key, input_value, possible_types, validity_verificator_list):

        if validity_verificator_list == None or validity_verificator_list == []: #if there was no specified validity_verification, we check the available types
            return self.verify_one_of_types(input_key, input_value, possible_types) 

        else:

            for validity_verificator in validity_verificator_list:
            
                try:
            
                    is_a_correct_type = validity_verificator(input_value) #use the verificator the Component has specified in its input signature

                    if not isinstance(is_a_correct_type, bool): #validity_verification must return bool
                        raise Exception(f"In component {self.name}, of type {type(self)}: Validity verification on key '{input_key}' returned a type other than bool")

                    elif is_a_correct_type == False:
                        raise Exception(f"In component {self.name}, of type {type(self)}: Value with key '{input_key}', of type {type(input_value)} did not pass Component specified validity verificator") 

                except Exception as e:
                    raise Exception(f"In component {self.name}, of type {type(self)}: Value with key '{input_key}', of type {type(input_value)} did not pass Component specified validity verificator due to reason: {e}") from e



    def verify_one_of_types(self, input_key, input_value, possible_types):
            
            if possible_types == None or possible_types == []:
                return #if there were no possible_types defined, there is no need to check the type                

            for possible_type in possible_types:

                if isinstance(input_value, possible_type):
                    return #break the loop and the functon, value is of one of the possible types
                
            #if we reach the end of the function, then the value is of none of the types
            raise Exception(f"In component {self.name}, of type {type(self)}: No validity verificator specified for key '{input_key}' and its value {input_value}, of type ({type(input_value)}) is of none of the possible types: {possible_types}\n{self.input.keys()}")
          

    # NOTES --------------------------------


    def write_line_to_notes(self, string : str = '', use_datetime=False):

        from datetime import datetime

        if use_datetime:
            string = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {string}"

        self.__notes.append(string)

    # TODO: maybe this should clone the notes
    def get_notes(self):
        return self.__notes

    def get_notes_as_text(self):
        return '\n'.join(self.__notes)

# VALIDITY VERIFICATION (static methods for validating input) -----------------------------          

def requires_input_process(func : FunctionType):
    '''
    An annotation that makes the input be processed, if it was not already, when a function is called
    
    Note that if a method has its super method with this annotation, adding it will be redundant
    '''
        
    def process_input_if_not_processed_wrapper(self : Component, *args, **kwargs):
        self.process_input_if_not_processed()
        return func(self, *args, **kwargs)
    
    return process_input_if_not_processed_wrapper


def uses_component_exception(func):
    '''A wrapper for functions so its errors have more information regarding the component they appeared in'''

    def wrapper(self : Component, *args, **kwargs):
        
        try:
            return func(self, *args, **kwargs)
        
        except Exception as e:
            original_args = e.args #arguments of an error
            e.args = (f"On Component {self.name} of type {type(self).__name__}:\n {original_args[0]}", *original_args[1:])
            raise e
            
    return wrapper




        


