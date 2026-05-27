
    

import os
from automarl.component import Component
from automarl.consts import CONFIGURATION_FILE_NAME
from automarl.utils.files_utils import write_text_to_file
from automarl.utils.json_utils.json_component_utils import json_string_of_component


def save_configuration(component : Component, config_directory, config_filename=CONFIGURATION_FILE_NAME, save_exposed_values=False, ignore_defaults=True, respect_ignore_order=True):
        
        json_str = json_string_of_component(component, save_exposed_values=save_exposed_values, ignore_defaults=ignore_defaults, respect_ignore_order=respect_ignore_order)
        
        path_to_save_configuration = os.path.join(config_directory, config_filename)

        write_text_to_file(filename=path_to_save_configuration, text=json_str, create_new=True)  