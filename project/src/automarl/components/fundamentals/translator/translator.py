

from automarl.component import Component
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.core.input_management import ParameterSignature


class Translator(Component):

    '''A strategy that translates data with a preditermined type and shape to another'''

    parameters_signature = {

        "original_shape" : ParameterSignature(mandatory=False),
        "in_place_translation" : ParameterSignature(default_value=False),
        "buffered_operations" : ParameterSignature(default_value=False),
        
        
        "device" : ParameterSignature(default_value="cpu", ignore_at_serialization=True, get_from_parent=True)

    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.in_place_translation = self.get_input_value("in_place_translation")

        self.buffered_operations = self.get_input_value("buffered_operations")

        self.device = self.get_input_value("device")

        self.original_shape = self.get_input_value("original_shape")
        self._setup_shape_cache()


    def _setup_shape_cache(self):

        if self.original_shape is not None:
            self.new_shape = self.get_shape(self.original_shape)

        else:
            self.new_shape = None

    def translate_state(self, state):
        raise NotImplementedError()
    

    def _get_shape(self, original_shape=None):
        raise NotImplementedError()
    

    def get_shape(self, original_shape=None):

        '''Gets the shape this translator will return'''
        
        if original_shape is None and self.new_shape is not None:
        
            return self.new_shape
        
        else:
            return self._get_shape(original_shape)

    


class TranslatorSequence(Translator):

    '''A strategy that translates data with a preditermined type and shape to another'''

    parameters_signature = {
        "translators_sequence" : ComponentListParameterSignature()
    }

    def _process_input_internal(self):

        super()._process_input_internal()

        self.translators_sequence : list[Translator] = self.get_input_value("translators_sequence")

        self._setup_translators_in_sequence_input()

        self._setup_sequence_state_cache()

        for translator in self.translators_sequence:
            translator.process_input_if_not_processed()

    
    def _setup_translators_in_sequence_input(self):
        

        if self.was_custom_value_passed_for_input("in_place_translation"):

            for translator in self.translators_sequence:
                translator.pass_input_if_no_value("in_place_translation", self.in_place_translation)


        if self.was_custom_value_passed_for_input("buffered_operations"):

            for translator in self.translators_sequence:
                translator.pass_input_if_no_value("buffered_operations", self.buffered_operations)



    def _setup_shape_cache(self): # we overwrite this method to do nothing as we want to have control over how the cache is processed
        pass

    
    def _setup_sequence_state_cache(self):

        if self.original_shape is not None:

            current_shape = self.original_shape

            for translator in self.translators_sequence:

                translator.pass_input({"original_shape" : current_shape})
                translator.process_input_if_not_processed()
                current_shape = translator.get_shape()

            self.new_shape = current_shape

        else:
            self.new_shape = None



    def translate_state(self, state):
        
        current_state = state

        for translator in self.translators_sequence:
            current_state = translator.translate_state(current_state)

        return current_state
    
    
    def _get_shape(self, original_shape):

        current_shape = original_shape

        for translator in self.translators_sequence:
            current_shape = translator.get_shape(current_shape)

        return current_shape