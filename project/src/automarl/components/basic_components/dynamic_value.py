
from automarl.component import Component, requires_input_process
from automarl.core.input_management import ParameterSignature
from automarl.core.advanced_input_management import ComponentParameterSignature, LookableParameterSignature

from automarl.core.types import numeric_type
from automarl.components.loggers.global_logger import globalWriteLine, is_global_logger_active

def get_value_or_dynamic_value(value) -> any:

    '''A function that abstracts the possibility of being a set value or a DynamicValue'''


    if isinstance(value, DynamicValue):
        to_return = value.value()
        return to_return

    else:
        return value



class DynamicValue(Component):
    
    '''
    Implements value(), which generates a value that is variable
    '''

    def _calc_value(self):
        '''A method meant to be implemented, calculates with current state'''
        raise NotImplementedError()

    @requires_input_process
    def value(self) -> any:
        '''A method that can be reimplemented with regards to changing state after calculating value'''
        return self._calc_value()



class DynamicValueBasedOnIter(DynamicValue):
    
    '''
    Implements value(), which generates a value that is variable
    '''

    def _process_input_internal(self):
        super()._process_input_internal()

        self.values["iteration"] = 0

    exposed_values = {"iteration" : 0}

    def reset_iter(self):
        self.values["iteration"] = 0

    def iter_number(self):
        return self.values["iteration"]
    
    def value(self):

        to_return = super().value()
        self.values["iteration"] += 1
        return to_return


class DynamicValueBasedOnComponent(DynamicValue):
    
    '''
    Implements value(), which generates a value given on the context of the passed component
    '''

    def _process_input_internal(self):
        super()._process_input_internal()

    
    parameters_signature = {

        "input_component" : ComponentParameterSignature()

    }    

    def _process_input_internal(self):
        super()._process_input_internal()

        self._input_component : Component = self.get_input_value("input_component")




class DynamicLinearValueInRange(DynamicValue):
    
    '''
    Samples a Component
    '''
    
    parameters_signature = {

        "initial_value" : ParameterSignature(),
        "final_value" : ParameterSignature(),

        "input_for_fun_max_value" : LookableParameterSignature(),

    }    

    def _process_input_internal(self):
        super()._process_input_internal()

        self._initial_value = self.get_input_value("initial_value")
        self._final_value = self.get_input_value("final_value")

        self._input_for_fun_max_value = self.get_input_value("input_for_fun_max_value", accepted_types=numeric_type)


class DynamicLinearValueInRangeBasedOnComponent(DynamicLinearValueInRange, DynamicValueBasedOnComponent):
    
    '''
    Samples a Component
    '''
    
    parameters_signature = {

        "input_for_fun_key" : ParameterSignature(),
        "input_for_fun_initial_value" : ParameterSignature(default_value=0)

    }    

    def _process_input_internal(self):

        super()._process_input_internal()

        self._input_for_fun_key = self.get_input_value("input_for_fun_key")

        self._input_for_fun_initial_value = self.get_input_value("input_for_fun_initial_value")

        self._slope_of_fun = (self._final_value - self._initial_value) / (self._input_for_fun_max_value - self._input_for_fun_initial_value)

        if is_global_logger_active():

            input_string = f"[{self._input_for_fun_initial_value}, {self._input_for_fun_max_value}]"
            output_string = f"[{self._initial_value}, {self._final_value}]"

            globalWriteLine(f"Created Dynamic Linear Value with projection for function:\n    {input_string} -> {output_string}")
            globalWriteLine(f"Using function: {self._initial_value} + {self._slope_of_fun} * X ")

    @requires_input_process
    def _calc_value(self):
        
        value_to_return = self._initial_value + self._slope_of_fun * self._input_component.values[self._input_for_fun_key]

        #globalWriteLine(f"{self.name}: {self._input_component.values[self._input_for_fun_key]} -> {value_to_return}")

        return value_to_return

class DynamicLinearValueInRangeBasedOnIter(DynamicLinearValueInRange, DynamicValueBasedOnIter):
    
    '''
    Samples a Component
    '''
    
    parameters_signature = {

    }    

    def _process_input_internal(self):

        super()._process_input_internal()


        self._slope_of_fun = (self._final_value - self._initial_value) / (self._input_for_fun_max_value)


    def _calc_value(self):
        
        return self._initial_value + self._slope_of_fun * self.iter_number()