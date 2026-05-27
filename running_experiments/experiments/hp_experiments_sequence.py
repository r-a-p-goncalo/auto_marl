


import os
import shutil

from automarl.utils.files_utils import open_or_create_folder
from automarl.utils.json_utils.json_component_utils import dict_from_json_string, json_string_of_component_dict
from experiments.paths import HP_SCRIPT_PATH, SCRIPT_PATH


BASE_COMMAND = [
    SCRIPT_PATH,
    "--COMMAND",
    "python",
    HP_SCRIPT_PATH
]

'''
The specification of a command to do hyperparameter optimization defined as a dictionary of the parameters
'''

    

# COMMON FUNCTIONS FOR COMMANDS ------------------------------------------------------------------------------

def copy_configurations_to_exp_paths(command_dict_collection : list[list[dict]]):

    '''Instead of relying on the configurations in the original paths, copies them immediatly to the target path of the experiment'''

    if not isinstance(command_dict_collection, list):
            raise Exception("Must be list")
    
    if isinstance(command_dict_collection[0], list):

        for command_dict_collection_element in command_dict_collection:
            copy_configurations_to_exp_paths(command_dict_collection_element)

    elif isinstance(command_dict_collection[0], dict):

        for command_dict_collection_element in command_dict_collection:

            command_dict_collection_element : dict = command_dict_collection_element

            path_to_store_experiment = command_dict_collection_element.get("path_to_store_experiment", None)
            experiment_relative_path = command_dict_collection_element.get("experiment_relative_path", None)
            hp_configuration_path = command_dict_collection_element.get("hp_configuration_path", None)

            if not None in [path_to_store_experiment, experiment_relative_path, hp_configuration_path]:

                full_path_of_experiment = os.path.join(path_to_store_experiment, experiment_relative_path)
                new_path_of_configuration = os.path.join(full_path_of_experiment, "configuration.json")
                
                new_path_dir = os.path.dirname(new_path_of_configuration)
                os.makedirs(new_path_dir, exist_ok=True)

                shutil.copy(hp_configuration_path, new_path_of_configuration)

                command_dict_collection_element["hp_configuration_path"] = new_path_of_configuration    
    
    else:
        raise Exception("Invalid element type")




def guarantee_same_path_in_commands(command_dict_sequence : list[dict]):
        
        '''
        Guarantees that in command sequences, the same relevant directories are being referenced
        This modifies the sequence in place, without returning a new one
        '''

        if not isinstance(command_dict_sequence, list):
            raise Exception("Must be list")
        

        elif len(command_dict_sequence) == 0:
            return
        
        elif isinstance(command_dict_sequence[0], list):

            for command_dict_sequence_element in command_dict_sequence:
                guarantee_same_path_in_commands(command_dict_sequence_element)

        elif isinstance(command_dict_sequence[0], dict):

            first_command : dict = command_dict_sequence[0]

            path_to_store_experiment = first_command["path_to_store_experiment"]
            experiment_relative_path = first_command["experiment_relative_path"]

            to_optimize_config_path = first_command.get("to_optimize_configuration_path", None)
            hp_configuration_path = first_command.get("hp_configuration_path", None)

            for command_dict in command_dict_sequence[1:]:

                command_dict["path_to_store_experiment"] = path_to_store_experiment
                command_dict["experiment_relative_path"] = experiment_relative_path
                command_dict["create_new_directory"] = False

                if to_optimize_config_path != None:
                    command_dict["to_optimize_configuration_path"] = to_optimize_config_path

                if to_optimize_config_path != None:
                    command_dict["hp_configuration_path"] = hp_configuration_path

                

        else:

            raise Exception("Invalid type")


def make_command_list_string(parameter_list : list):


    return ' '.join([str(parameter_list_element) for parameter_list_element in parameter_list])



def make_command_collection_string(parameter_collection, complete_commands=False):


    if isinstance(parameter_collection, dict):
        
        command_sequence = hp_opt_command_sequence(parameter_collection) if complete_commands else hp_opt_command_sequence_from_parameters(parameter_collection)

        return make_command_list_string(command_sequence)

    if isinstance(parameter_collection, str):
        return parameter_collection

    elif isinstance(parameter_collection, list):
        make_command_list_string(parameter_collection)

    else:
        raise Exception("Invalid type")



def hp_opt_command_sequence_from_parameters(
                                 parameter_dict : dict,
                                 command_args_list : list = None
                            ):
    
    '''Transforms a given parameter dict in an actual command list for running hyperparameter optimization problems'''
    
    command_args_list = [] if command_args_list == None else command_args_list

    for key, value in parameter_dict.items():
        command_args_list.append(f"--{key}")
        command_args_list.append(str(value))

    return command_args_list

def hp_opt_command_sequence(
                                 parameter_dict : dict,
                            ):
    
    '''Transforms a given parameter dict in an actual command list for running hyperparameter optimization problems'''
    
    return hp_opt_command_sequence_from_parameters(parameter_dict, [*BASE_COMMAND])


def make_command_dicts_command_strings(command_dicts, complete_commands=True):

    '''Returns a collection with the same shape as the passed one, but command_dicts are transformed into correct strings'''

    if isinstance(command_dicts, list): # if is list of commands

        # we recursively call the method for each of those sublists, mantaining shape
        return [make_command_dicts_command_strings(command_dict_element) for command_dict_element in command_dicts]

    # if we called it for a single command
    elif isinstance(command_dicts, dict):

        return make_command_collection_string(command_dicts, complete_commands)
    

    else:
        raise Exception(f"Invalid type when making commands")
    


def print_commands(command_dicts_str, ident_level=0):
    
    ident_str = '    ' * ident_level

    if isinstance(command_dicts_str, list):
        
        if len(command_dicts_str) == 0: # if list is empty we do nothing
            pass
        
        elif isinstance(command_dicts_str[0], list): # if is list of lists we ident accordingly

            print(ident_str + "----\n")

            for command_dicts_element in command_dicts_str:

                print_commands(command_dicts_element, ident_level + 1)
            
            print()
            print(ident_str + "----\n")

        else: # if is list of parameters we make it a command string

            print(ident_str + "[----\n")

            for command_dicts_element in command_dicts_str:
                print(ident_str + f"{make_command_collection_string(command_dicts_element)}\n")

            print(ident_str + "----]\n")

    elif isinstance(command_dicts_str, str):

        print(ident_str + f"{command_dicts_str}\n")

    else:
    
        raise Exception("Invalid type")


# MAKE EXPERIMENTS FOR VALUES ------------------------------------------------------------------------------12


def change_command_for_value_change(command_dict : dict, 
                                    value_to_put, 
                                    localization_to_change,
                                    name_of_experiment: str, 
                                    directory_to_store_definitions : str):
        
        '''
        Changes value in command dict by changing its to_optimize_configuration_path, 
        its path_to_store_experiment and experiment_relative_path
        '''
        
        if "to_optimize_configuration_path" in command_dict.keys() and "path_to_store_experiment" in command_dict.keys() and  "experiment_relative_path" in command_dict.keys():
        
            # the current to optimize configuration
            base_to_opt_config_path = command_dict["to_optimize_configuration_path"]

            base_to_opt_name = (os.path.basename(base_to_opt_config_path).split('.')[0])

            fd = open(base_to_opt_config_path, 'r') 
            base_to_opt_config_str = fd.read() # reads the base configuration str
            fd.close()

            # modify configuration to use desired value
            base_to_opt_config_dict = dict_from_json_string(base_to_opt_config_str)

            current_value_localizer = base_to_opt_config_dict
            for index_loc_i in range(len(localization_to_change) - 1):
                current_value_localizer = current_value_localizer[localization_to_change[index_loc_i]]

            current_value_localizer[localization_to_change[len(localization_to_change) - 1]] = value_to_put

            # save changed configuration
            json_str_to_opt = json_string_of_component_dict(base_to_opt_config_dict, respect_ignore_order = False)

            to_optimize_config_path = os.path.join(directory_to_store_definitions, f'{base_to_opt_name}_{name_of_experiment}.json')

            fd = open(to_optimize_config_path, 'w')
            fd.write(json_str_to_opt)
            fd.close()

            # SETUP EXPERIMENT COMMAND SEQUENCE

            current_path_to_store_experiment = command_dict["path_to_store_experiment"]
            current_experiment_relative_path = command_dict["experiment_relative_path"]

            current_path_to_store_experiment = f"{current_path_to_store_experiment}\\{current_experiment_relative_path}"
            current_experiment_relative_path = name_of_experiment

            command_dict_to_return = {
                **command_dict,
                "path_to_store_experiment" : current_path_to_store_experiment,
                "experiment_relative_path" : current_experiment_relative_path,
                "to_optimize_configuration_path" : to_optimize_config_path
                }
            


            return command_dict_to_return

        else: # if not all necessary keys were in command
            return {**command_dict}

def process_original_without_value_change(command_dict : dict, 
                                    directory_to_store_definitions : str):
                

        if "to_optimize_configuration_path" in command_dict.keys() and "path_to_store_experiment" in command_dict.keys() and  "experiment_relative_path" in command_dict.keys():
        
            # the current to optimize configuration
            base_to_opt_config_path = command_dict["to_optimize_configuration_path"]

            base_to_opt_name = (os.path.basename(base_to_opt_config_path).split('.')[0])

            fd = open(base_to_opt_config_path, 'r') 
            base_to_opt_config_str = fd.read() # reads the base configuration str
            fd.close()

            # modify configuration to use desired value
            base_to_opt_config_dict = dict_from_json_string(base_to_opt_config_str)

            # save non changed configuration
            json_str_to_opt = json_string_of_component_dict(base_to_opt_config_dict, respect_ignore_order = False)

            to_optimize_config_path = os.path.join(directory_to_store_definitions, f'{base_to_opt_name}_original.json')

            fd = open(to_optimize_config_path, 'w')
            fd.write(json_str_to_opt)
            fd.close()

            # SETUP EXPERIMENT COMMAND SEQUENCE

            current_path_to_store_experiment = command_dict["path_to_store_experiment"]
            current_experiment_relative_path = command_dict["experiment_relative_path"]

            current_path_to_store_experiment = f"{current_path_to_store_experiment}\\{current_experiment_relative_path}"
            current_experiment_relative_path = 'original'

            command_dict_to_return = {
                **command_dict,
                "path_to_store_experiment" : current_path_to_store_experiment,
                "experiment_relative_path" : current_experiment_relative_path,
                "to_optimize_configuration_path" : to_optimize_config_path
                }


            return command_dict_to_return

        else: # if not all necessary keys were in command
            return {**command_dict}





def expand_commands_for_each_value_change(command_dicts_list, value_changes, loc_of_value, experiment_names, directory_to_store_definitions, mantain_original=False):

    '''
    Given a collection (that can have nested lists) of command dicts lists (sequences of commands), expands those for each of the given value changes
    '''


    if isinstance(command_dicts_list, dict):

        raise Exception("Expected list of commands")

    elif len(command_dicts_list) == 0:
        return command_dicts_list # if it is empty, we do nothing


    # if it is a list of lists
    # in this case we expect list[list[dict]] -> list[list[dict]], in whici list[dict] elements are added for the list
    elif isinstance(command_dicts_list[0], list):

        to_return = []

        # for each list[dict] element
        for command_dicts_list_element in command_dicts_list:

            # this makes list[dict] -> list[list[dict]]
            expanded_value = expand_commands_for_each_value_change(
                command_dicts_list=command_dicts_list_element,
                value_changes=value_changes,
                loc_of_value=loc_of_value,
                experiment_names=experiment_names,
                directory_to_store_definitions=directory_to_store_definitions,
                mantain_original=mantain_original)

            to_return.append(expanded_value)

        return to_return

    # if it is a list of dicts, we add to it the changed dictionaries, changing [command_1, command_2] -> [[command_1, command_2], [altered_1, altered_2]]
    elif isinstance(command_dicts_list[0], dict):

        # in this case we should add expanded list[dict] to here
        to_return : list[list[dict]] = []

        if mantain_original: # if we are to mantain the original

            new_command_list : list[dict] = []
        
            # for each original command
            for command_dicts_element in command_dicts_list: # for each command, we change it

                new_command_list.append(
                    process_original_without_value_change(
                        command_dict=command_dicts_element,
                        directory_to_store_definitions= directory_to_store_definitions
                    ))
                
            to_return.append(new_command_list)


        for value_change_i in range(len(value_changes)):

            new_command_list = []
        
            for command_dicts_element in command_dicts_list: # for each command, we change it

                new_command_list.append(
                    change_command_for_value_change(
                        command_dict=command_dicts_element,
                        value_to_put=value_changes[value_change_i],
                        localization_to_change=loc_of_value,
                        name_of_experiment=experiment_names[value_change_i],
                        directory_to_store_definitions= directory_to_store_definitions
                    ))
                
            to_return.append(new_command_list)

        # return list[list[dict]]
        return to_return
    

def expand_commands_for_each_path_in_directory(command_dicts, localization, directory_of_paths, directory_to_store_definitions, mantain_original=False, paths_to_test=None):

    '''Given a collection (that can have nested lists) of command dicts, expands those for each of the given models'''

    if paths_to_test == None:
        # get model paths
        model_paths = [
            os.path.join(directory_of_paths, f)
            for f in os.listdir(directory_of_paths)
        ]

    else:
        model_paths = [
            os.path.join(directory_of_paths, f)
            for f in os.listdir(directory_of_paths)
            if f in paths_to_test
        ]   

    experiment_names = [os.path.basename(path) for path in model_paths]

    return expand_commands_for_each_value_change(
            command_dicts_list=command_dicts, 
            value_changes=model_paths,
            loc_of_value=localization,
            experiment_names=experiment_names,
            directory_to_store_definitions=directory_to_store_definitions,
            mantain_original=mantain_original
        )

def unfold_sequences_to_correct_format(commands_collection_element : list) -> list[list[dict]]:


    if not isinstance(commands_collection_element, list):
        raise Exception(f"Commands collection element must be of type list[dict|str] but was {type(commands_collection_element)}")
    
    elif len(commands_collection_element) == 0:
        return []
    
    # if commands_collection type is list[dic | str] we have to turn it into a list
    elif isinstance(commands_collection_element[0], (dict, str)):
        return [commands_collection_element]
    
    
    #if commands_collection_element type is list[list[?]], we must unfold it
    elif isinstance(commands_collection_element[0], list):

        to_return : list[list[dict]] = []


        # each element is of list[?]
        for element_in_collection_element in commands_collection_element:

            unfolded_list_element : list[list[dict]] = unfold_sequences_to_correct_format(element_in_collection_element)

            to_return = [
                    *to_return, *unfolded_list_element
                ]


        return to_return


    else:
        raise Exception(f"Element in commands collection must be of type list[str | dict] but was list[{type(commands_collection_element[0])}]")

        


# MAKE EXPERIMENTS FOR MODELS ------------------------------------------------------------------------------


def expand_commands_for_each_model(command_dicts, directory_of_models, directory_to_store_definitions, mantain_original=False, models_to_test=None): 

    '''Given a collection (that can have nested lists) of command dicts, expands those for each of the given models'''

    return expand_commands_for_each_path_in_directory(
        command_dicts=command_dicts,
        localization=["input", "agents_input", "policy", 1, "model"],
        directory_of_paths=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions,
        mantain_original=mantain_original,
        paths_to_test=models_to_test
    )



def expand_commands_for_each_critic_model(command_dicts, directory_of_models, directory_to_store_definitions, mantain_original=False, models_to_test=None):

    '''Given a collection (that can have nested lists) of command dicts, expands those for each of the given models'''

    return expand_commands_for_each_path_in_directory(
        command_dicts=command_dicts,
        localization=["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "critic_model"],
        directory_of_paths=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions,
        mantain_original=mantain_original,
        paths_to_test=models_to_test
    )

    
def setup_experiment_directories(
                                 directory_to_store_experiment,
                                 experiment_name,
                                 ):
    
    '''
    Generate directories necessary to start an experiment
    
    '''

    directory_to_store_experiment = open_or_create_folder(directory_to_store_experiment, folder_name=f"{experiment_name}")

    directory_to_store_definition = open_or_create_folder(directory_to_store_experiment, "definitions", create_new=False)

    directory_to_store_experiments = os.path.join(directory_to_store_experiment, "experiments")
    open_or_create_folder(directory_to_store_experiments) # we do this because the directory is shared, and so there is a possibility of multiple processes trying to create it at the same time

    directory_to_store_logs = os.path.join(directory_to_store_experiment, "logs")

    return directory_to_store_experiment, directory_to_store_definition, directory_to_store_experiments, directory_to_store_logs    