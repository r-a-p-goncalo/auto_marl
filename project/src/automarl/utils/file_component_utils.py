
    

import os
import pickle

from automarl.components.basic_components.state_management import load_component_from_folder
from automarl.component import Component
from automarl.consts import CONFIGURATION_FILE_NAME, LOADED_COMPONENT_FILE_NAME
from automarl.utils.json_utils.json_component_utils import component_from_json_string
from automarl.components.loggers.global_logger import globalWriteLine


def gen_component_from_path(path, parent_component_for_generated : Component = None) -> Component:

    if not os.path.exists(path):
        raise Exception(f"Path does not exist: {path}")
    
    elif os.path.isdir(path):
        generated_component =  gen_component_in_directory(path, parent_component_for_generated)
    
    elif os.path.isfile(path):
        generated_component =  gen_component_in_file_path(path)
    
    else:
        raise ValueError(f"Path '{path}' is neither a file nor a directory.")
    
    return generated_component
    
    

def gen_component_in_directory(dir_path, parent_component_for_generated : Component = None) -> Component:
    
    configuration_file = os.path.join(dir_path, CONFIGURATION_FILE_NAME)

    if os.path.exists(configuration_file):
        return load_component_from_folder(dir_path, parent_component_to_be=parent_component_for_generated)
    
    component_loaded_file = os.path.join(dir_path, LOADED_COMPONENT_FILE_NAME)
    
    if os.path.exists(component_loaded_file):
        return gen_component_in_file_path(component_loaded_file)

    raise Exception(f"No component defined in folder: {dir_path}")



def gen_component_in_file_path(file_path) -> Component:

    component_to_return = None
    
    if file_path.endswith('.json'):

        globalWriteLine(f"WARNING: Generating component from file {file_path}, if you want to load the enterity of its state, you should use the base folder instead of the configuration path")
        
        with open(file_path, 'r') as f:
            str_to_gen_from = f.read()
            component_to_return = component_from_json_string(str_to_gen_from)

    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            component_to_return =  pickle.load(f)

    else:
        raise Exception("Not supported file to generate component from")

    component_to_return.write_line_to_notes(f"Component was generated from path {file_path}", use_datetime=True)

    return component_to_return
