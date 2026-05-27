from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.loggers.global_logger import globalWriteLine


class ComponentDebug(ComponentWithLogging):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    is_debug_schema = True


    
    parameters_signature = {
                       }


    def _process_input_internal(self): #this is the best method to have initialization done right after

        globalWriteLine(f"{self.name}: Processing input:\n", file="__input_debug.txt")

        for key, value in self.input.items():
            self.__print_in_global_with_ident(key, value)
        
        super()._process_input_internal()

        globalWriteLine(f"{self.name}: Finished processing input\n", file="__input_debug.txt")



    def __print_in_global_with_ident(self, key, value, ident=1):

        if isinstance(value, (list, tuple)):

            globalWriteLine(f"{'   ' * ident}{key}:", file="__input_debug.txt")
            
            for i in range(len(value)):
                self.__print_in_global_with_ident(f"{i}", value[i], ident=ident + 1)

        elif isinstance(value, dict):

            globalWriteLine(f"{'   ' * ident}{key}:", file="__input_debug.txt")

            for k, v in value.items():
                self.__print_in_global_with_ident(k, v, ident=ident + 1)

        else:
            globalWriteLine(f"{'   ' * ident}{key}: {value}", file="__input_debug.txt")



    def pass_input(self, input):

        globalWriteLine(f"{self.name}: Received input:\n", file="__input_debug.txt")

        for key, value in self.input.items():
            self.__print_in_global_with_ident(key, value)

        super().pass_input(input)

        globalWriteLine(f"{self.name}: Finished processing passed input\n", file="__input_debug.txt")


        

    def _try_look_input_in_attribute(self, input_key, attribute_name):

        globalWriteLine(f"{self.name}: Trying to look for attribute '{attribute_name}' for input '{input_key}'", file="__input_debug.txt")

        to_return = super()._try_look_input_in_attribute(input_key, attribute_name)

        if to_return == None:
            globalWriteLine(f"{self.name}: Did not have attribute '{attribute_name}' for input '{input_key}'", file="__input_debug.txt")

        else:
            globalWriteLine(f"{self.name}: Attribute '{attribute_name}' for input '{input_key}' was found", file="__input_debug.txt")

        return to_return
    

    def _try_look_input_in_values(self, input_key, value_name):

        globalWriteLine(f"{self.name}: Trying to look for value '{value_name}' for input '{input_key}'", file="__input_debug.txt")

        to_return = super()._try_look_input_in_values(input_key, value_name)

        if to_return == None:
            globalWriteLine(f"{self.name}: Did not have value '{value_name}' for input '{input_key}'", file="__input_debug.txt")

        else:
            globalWriteLine(f"{self.name}: Value '{value_name}' for input '{input_key}' was found", file="__input_debug.txt") 

        return to_return
    

    def _on_parent_component_defined(self):
        super()._on_parent_component_defined()

        globalWriteLine(f"{self.name}: New parent component was defined: {self.parent_component}", file="__input_debug.txt") 


    def get_attr_from_parent(self, attr_name : str):
        '''Gets an attribute from a parent component, None if non existent'''
        
        globalWriteLine(f"{self.name}: Trying to get value with key {attr_name} from parent component")
        globalWriteLine(f"{self.name}: Parent component is: {self.parent_component if self.parent_component is None else self.parent_component.name}")

        to_return = super().get_attr_from_parent(attr_name)

        globalWriteLine(f"{self.name}: Value got from parent: {to_return}")

        return to_return


    def clone(self, *args, **kwargs):

        self.lg.writeLine(f"Cloning component...")

        return super().clone(*args, **kwargs)
