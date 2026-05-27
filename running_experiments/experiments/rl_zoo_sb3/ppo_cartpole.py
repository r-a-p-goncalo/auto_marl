


from experiments import hp_experiments_sequence
from experiments.base_experiment import experiment_base_commands_and_info



def experiment_for_poo_actors(directory_to_store_experiment = "C:\\rgoncalo\\experiments",
                 base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json",
                 hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration_3.json", 
                 mantain_original=False,
                 directory_of_models="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models",
                 models_to_test=None,
                 experiment_name="sb3_zoo_dqn_cartpole_hp_opt_mult_samplers_pruners",
                 base_commands=None,
                 base_command : dict = None):


    base_commands, directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = experiment_base_commands_and_info(
        directory_to_store_experiment,
                 base_to_opt_config_path,
                 hp_opt_config_path, 
                 experiment_name,
                 base_commands)

    expanded_commands_for_models = hp_experiments_sequence.expand_commands_for_each_model(
        command_dicts=base_commands,
        directory_of_models=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions,
        mantain_original=mantain_original,
        models_to_test=models_to_test
    )

    hp_experiments_sequence.guarantee_same_path_in_commands(expanded_commands_for_models)

    return expanded_commands_for_models




def experiment_for_poo_actors_and_critics(directory_to_store_experiment = "C:\\rgoncalo\\experiments",
                 base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json",
                 hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration_3.json", 
                 mantain_original=False,
                 directory_of_models="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models",
                 models_to_test=None,
                 directory_of_critics="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models",
                 critics_to_test=None,
                 mantain_critic_original=True,
                 experiment_name="sb3_zoo_dqn_cartpole_hp_opt_mult_samplers_pruners",
                 base_commands : list[dict] =None,
                 base_command : dict = None):


    base_commands, directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = experiment_base_commands_and_info(
        directory_to_store_experiment,
                 base_to_opt_config_path,
                 hp_opt_config_path, 
                 experiment_name,
                 base_commands,
                 base_command)

    print("\nCOMMANDS FOR MODELS:\n")

    expanded_commands_for_models = hp_experiments_sequence.expand_commands_for_each_model(
        command_dicts=base_commands,
        directory_of_models=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions,
        mantain_original=mantain_original,
        models_to_test=models_to_test
    )

    hp_experiments_sequence.print_commands(expanded_commands_for_models)

    print("\nEND OF COMMANDS FOR MODELS:\n")

    expanded_commands_strs_for_critics = hp_experiments_sequence.expand_commands_for_each_critic_model(
        command_dicts=expanded_commands_for_models,
        directory_of_models=directory_of_critics,
        directory_to_store_definitions=directory_to_store_definitions,
        models_to_test=critics_to_test,
        mantain_original=mantain_critic_original
    )

    hp_experiments_sequence.guarantee_same_path_in_commands(expanded_commands_strs_for_critics)

    return expanded_commands_strs_for_critics