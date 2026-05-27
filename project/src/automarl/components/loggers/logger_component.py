from datetime import datetime
import pickle
from automarl.component import ParameterSignature, Component, requires_input_process

from automarl.utils.files_utils import  saveDataframe

from enum import Enum

from automarl.components.basic_components.artifact_management import ArtifactComponent

import os

import pandas as pd

from automarl.utils.smart_enum import SmartEnum
from automarl.utils.json_utils.json_component_utils import json_string_of_component
from automarl.components.loggers.global_logger import globalWriteLine


class DEBUG_LEVEL(SmartEnum):
    NONE = -1
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


IDENT_SPACE = '    '

DEFAULT_LOGGER_LEVEL = DEBUG_LEVEL.DEBUG
DEFAULT_WRITE_TO_FILE_WHEN = 1000

def change_default_logger_level(new_value):

    global DEFAULT_LOGGER_LEVEL

    DEFAULT_LOGGER_LEVEL = new_value

    LoggerSchema.get_schema_parameter_signature("necessary_logger_level").change_default_value(new_value)

# LOGGING SCHEMA  -------------------------------------------------------------------------------------------------   

class LoggerSchema(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''

    
    
    parameters_signature = {

                       "necessary_logger_level" : ParameterSignature(
                            generator= lambda _ : DEFAULT_LOGGER_LEVEL, # this is so we can swith the default logger level
                            ignore_at_serialization=True),

                       "default_print" : ParameterSignature(default_value=False, ignore_at_serialization=True),

                       "write_to_file_when_text_lines_over" : ParameterSignature(
                           default_value=DEFAULT_WRITE_TO_FILE_WHEN,
                           mandatory=False),

                       "artifact_relative_directory" : ParameterSignature(
                                priority=1,
                                default_value="log"
                                ),
                                              
                       "user_timestamp_in_logs" : ParameterSignature(default_value=True, ignore_at_serialization=True),
                       
                       "log_text_file" : ParameterSignature(default_value='log.txt', ignore_at_serialization=True, description="The name of the log text file, if it is not specified, it will be created as 'log.txt' in the log directory"),
                    
                        "object_with_name" : ParameterSignature(mandatory=False, ignore_at_serialization=True, description="The object that will be used to create the profile for the logger, if it is not specified, the logger will not have a profile"),

                        "create_new_directory" : ParameterSignature(default_value=False), # this is to overwrite behavior defined in artifact component
                    }
        

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                        
    
    # INITIALIZATION --------------------------------------------------------

    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()
            
        self.necessary_logger_level = DEBUG_LEVEL.from_value(self.get_input_value("necessary_logger_level"))  
        
        self.default_print = self.get_input_value("default_print")
        
        self.default_use_timestamp = self.get_input_value("user_timestamp_in_logs")
        
        self.default_log_text_file = self.get_input_value("log_text_file")
        
        self.object_with_name = self.get_input_value("object_with_name")
        
        self.write_to_file_when_text_lines_over = self.get_input_value("write_to_file_when_text_lines_over")

        # if it was negative, revert it back to None
        self.write_to_file_when_text_lines_over = None if self.write_to_file_when_text_lines_over != None and self.write_to_file_when_text_lines_over <= 0 else self.write_to_file_when_text_lines_over

        if self.write_to_file_when_text_lines_over != None:
            self.text_buffer : dict[str, list] = self.text_buffer if hasattr(self, "text_buffer") else {}
            self.text_buffer_counts : dict[str, int] = self.text_buffer_counts if hasattr(self, "text_buffer_counts") else {}


    # LOGGING -----------------------------------------------------------------------------        

    @requires_input_process
    def writeLine(self, string : str = "", file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
    
        '''
        This writes a line to the logger of the component, meant to be called by outside of the scope of the component
        '''

        return self._writeLine(string, file, level, toPrint, use_time_stamp, str_before, ident_level)


    @requires_input_process
    def change_logger_level(self, new_level : DEBUG_LEVEL):

        self.necessary_logger_level = new_level
    
            
    def _writeLine(self, string : str, file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
        
        '''
        This writes a line to the logger of the component, meant to be called inside the scope of the component
        It does not require the component to have its input processed
        '''
        

        #print(f"passed value: ({level}, {level.value}), necessary value: ({self.necessary_logger_level}, {self.necessary_logger_level.value})")
        if self.necessary_logger_level.value >= level.value: #if the level of the message is lower than the default level, we write it (more important than what was asked)

            if toPrint == None:
                toPrint = self.default_print
            
            if use_time_stamp == None: #if it was not specified, we use the default value
                use_time_stamp = self.default_use_timestamp
                
            if ident_level > 0:
                for _ in range(ident_level):
                    string = IDENT_SPACE + string

            if use_time_stamp: #if we want to use the timestamp, we add it to the string
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                string = f'[{timestamp}] {string}'
                
            string = f"{str_before}{string}"

            if self.object_with_name is not None: #if we have an object with a name, we add it to the string
                string = f'{self.object_with_name.name}: {string}'

            if file is None:
                file = self.default_log_text_file
            
            string = f'{string}'
            
            if toPrint:
                print(string)

            if self.write_to_file_when_text_lines_over == None:

                path_to_write = os.path.join(self.get_artifact_directory(), file)
                directory = os.path.dirname(path_to_write)

                os.makedirs(directory, exist_ok=True)

                fd = open(path_to_write, 'a')
                fd.write(f'{string}\n')
                fd.close()
            
            else:
                self.write_to_buffer_file(filename=file, text=f"{string}\n")

        
        
    # note this does not necessary needs all input to be processed
    def saveFile(self, data, directory='', filename='data'): #saves a file using the directory of this log object as a point of reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)
        
        fd = open(os.path.join(self.get_artifact_directory(), directory, filename), 'wb') 
        pickle.dump(data, fd)
        fd.close()
    
    # note this does not necessary needs all input to be processed
    def saveDataframe(self, df, directory='', filename='dataframe.csv'): #saves a dataframe using this log object as a reference
        
        '''
        Saves dataframe in artifact directory
        This does not trigger input processing
        '''
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)

        saveDataframe(df, os.path.join(self.get_artifact_directory(), directory), filename)

    def loadDataframe(self, directory='', filename='dataframe.csv'):
        '''
        Loads dataframe in artifact directory
        This does not trigger input processing
        '''
        return pd.read_csv(os.path.join(self.get_artifact_directory(), directory, filename))
    
                
    def createDirIfNotExistent(self, dir): #creates a dir if it does no exist
        
        dir = os.path.join(self.get_artifact_directory(), dir)
        
        try:
            os.listdir(dir)
        
        except:
            os.makedirs(dir)
            
    @requires_input_process
    def openFile(self, fileRelativePath): #reads and returns a file
        fd = open(os.path.join(self.get_artifact_directory(), fileRelativePath), 'rb')
        toReturn = pickle.load(fd)
        fd.close()
        return toReturn
    
    @requires_input_process
    def createProfile(self, name : str = '', object_with_name = None):            
        
        '''Creates a new logger object with the only difference of this one being the used name'''
        
        input_of_copy = {**self.input} #copy the input to avoid modifying the original one
        
        if object_with_name is not None:
            input_of_copy['object_with_name'] = object_with_name
        
        elif name != '':
            object_with_name = object() #create simple object with a name attribute
            object_with_name.name = name
            input_of_copy["object_with_name"] = object_with_name
        else:
            raise ValueError("You must provide a name or an object with a name to create a profile")

        return LoggerSchema(input=input_of_copy)
    
    
    def create_buffer_for_file(self, filename):

        self.text_buffer[filename] = []
        self.text_buffer_counts[filename] = 0


    def write_to_buffer_file(self, filename, text):

        buffer_for_file = self.text_buffer.get(filename, None)

        if buffer_for_file == None:
            self.create_buffer_for_file(filename)


        self.text_buffer[filename].append(text)
        self.text_buffer_counts[filename] += 1


        if self.text_buffer_counts[filename] >= self.write_to_file_when_text_lines_over:

            self.flush_buffer_of_file(filename)


    def flush_buffer_of_file(self, filename):
            
            path_to_write = os.path.join(self.get_artifact_directory(), filename)
            directory = os.path.dirname(path_to_write)

            os.makedirs(directory, exist_ok=True)

            fd = open(path_to_write, 'a')

            fd.write("".join(self.text_buffer[filename]))
            fd.close()
                    
            self.text_buffer[filename].clear()
            self.text_buffer_counts[filename] = 0


    @requires_input_process
    def flush_text(self):

        if self.write_to_file_when_text_lines_over != None:

            for filename in self.text_buffer.keys():
                self.flush_buffer_of_file(filename)

    
    
def flush_text_of_all_loggers_and_children(component : Component):

    if isinstance(component, LoggerSchema):
        component.flush_text()

    elif isinstance(component, ComponentWithLogging) and component.has_logger_object_defined():
        component.lg.flush_text()

    for child_component in component.child_components:
        flush_text_of_all_loggers_and_children(child_component)



# COMPONENT WITH LOGGING -------------------------------------------------------------------------------------------------    
    
# CONTEXT STUFF -----------------------------------------------------

from contextlib import contextmanager

from contextvars import ContextVar

_current_logger: ContextVar["LoggerSchema | None"] = ContextVar(
    "current_logger",
    default=None
)

_first_logger_override: ContextVar["LoggerSchema | None"] = ContextVar(
    "first_logger_override",
    default=None
)

_component_that_got_first_logger_override: ContextVar["LoggerSchema | None"] = ContextVar(
    "component_that_got_first_logger_override",
    default=None
)

@contextmanager
def use_logger(logger: "LoggerSchema"):
    token = _current_logger.set(logger)
    try:
        yield logger
    finally:
        _current_logger.reset(token)

@contextmanager
def override_first_logger(logger):
    token1 = _first_logger_override.set(logger)
    token2 = _component_that_got_first_logger_override.set(None)
    try:
        yield logger
    finally:
        _first_logger_override.reset(token1)
        _component_that_got_first_logger_override.reset(token2)


# ACTUAL COMPONENT WITH LOGGING ---------------------------------------------------------

def on_log_pass(self : Component):
    self.define_new_logger_object(self.get_input_value("logger_object"))
    
def generate_logger_for_component(self : ArtifactComponent):
    return self.initialize_child_component(LoggerSchema, input={
            "create_new_directory" : False,
            "base_directory" : self, 
            "artifact_relative_directory" : ""}
        )

class ComponentWithLogging(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    
    parameters_signature = {
                                                                                
                       "logger_object" : ParameterSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = generate_logger_for_component , 
                                                        on_pass=on_log_pass),

                        "logger_input" : ParameterSignature(ignore_at_serialization=True, default_value={}),

                       }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._lg = None

    def _process_input_internal(self): #this is the best method to have initialization done right after
            
        super()._process_input_internal()
        
        self._lg : LoggerSchema = self.get_input_value("logger_object") if not self.has_logger_object_defined() else self._lg 
        
        self._lg.pass_input(self.get_input_value("logger_input"))

    def define_new_logger_object(self, new_logger : LoggerSchema):
        self._lg = new_logger

    def has_logger_object_defined(self):
        hasattr(self, "_lg") and self._lg is not None

    @property
    def lg(self) -> LoggerSchema:

        first_override = _first_logger_override.get()

        if first_override is not None:
            
            component_that_got_first_override = _component_that_got_first_logger_override.get()

            if component_that_got_first_override is None:
                _component_that_got_first_logger_override.set(self)
                component_that_got_first_override = self

            if component_that_got_first_override is self:
                return first_override

        ctx_logger = _current_logger.get()
        if ctx_logger is not None:
            return ctx_logger

        return self._lg


    def change_logger_level(self, new_level : DEBUG_LEVEL):

        if not self.has_logger_object_defined():
            self.pass_input({"logger_object" : generate_logger_for_component(self)})

        self.lg.pass_input({"necessary_logger_level" : new_level})

    


    @requires_input_process
    def write_configuration_to_relative_file(self, filename : str, level : DEBUG_LEVEL = DEBUG_LEVEL.INFO, save_exposed_values=False, ignore_defaults=True, respect_ignore_order=False):

        self_json_str = json_string_of_component(self, save_exposed_values=save_exposed_values, ignore_defaults=ignore_defaults, respect_ignore_order=respect_ignore_order)

        self.lg.writeLine(string=self_json_str, file=filename, level=level, use_time_stamp=False)
    

