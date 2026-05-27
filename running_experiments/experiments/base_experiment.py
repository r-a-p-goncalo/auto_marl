


from experiments import hp_experiments_sequence

def experiment_base_commands(directory_to_store_experiment,
                 base_to_opt_config_path,
                 hp_opt_config_path, 
                 experiment_name,
                 base_commands=None,
                 base_command=None):


    base_commands, directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = experiment_base_commands_and_info(directory_to_store_experiment,
                 base_to_opt_config_path,
                 hp_opt_config_path, 
                 experiment_name,
                 base_commands,
                 base_command)
    
    return base_commands


def experiment_base_commands_and_info(directory_to_store_experiment,
                 base_to_opt_config_path,
                 hp_opt_config_path, 
                 experiment_name,
                 base_commands=None,
                 base_command : dict = None):
    

    directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = hp_experiments_sequence.setup_experiment_directories(
        directory_to_store_experiment=directory_to_store_experiment,
        experiment_name=experiment_name
    )

    base_command = {} if base_command == None else base_command

    # Basic parameters all commands will have
    base_command = {
            **base_command,
            "create_new_directory" : "False",
            "path_to_store_experiment" : directory_to_store_experiments,
        }
    

    base_command.setdefault("experiment_relative_path", ".")    

    if base_commands == None:

        base_commands = [
            {
                    "num_trials" : 50,
                    "sampler" : "Random", # this is to gain some knowledge first
                    "hp_configuration_path" : hp_opt_config_path,
                    "to_optimize_configuration_path" : base_to_opt_config_path,
                    "num_steps" : 2,

             },

            {
                    "num_trials" : 150,
                    "sampler": "TreeParzen",
                    "num_steps" : 2

            },
        ]

    for base_command_setuped in base_commands:

        for base_command_key, base_command_value in base_command.items():
            base_command_setuped[base_command_key] = base_command_value

    return base_commands, directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs
