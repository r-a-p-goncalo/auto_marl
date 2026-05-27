import os


def gen_default_hp_config_path(path_to_store_experiment, experiment_relative_path):

    print(f"Generating default hp configuration path...")

    to_return = os.path.join(path_to_store_experiment, experiment_relative_path, "configuration.json")

    print(f"Generated path: {to_return}")

    if not os.path.exists(to_return):
        to_return = "configuration.json"
        print(f"Path does not exist, new path: {to_return}")

    if not os.path.exists(to_return):
        raise Exception(f"In generating hp default configuration path, could not generate a path that exists")

    return to_return


def main(component_configuration_path=None, path_to_store_experiment='.\\data\\experiments', create_new_directory=None, experiment_relative_path=None, global_logger_level=None, global_logger_path=None, default_logger_level=None):
    

    from automarl.utils.json_utils.json_component_utils import gen_component_from_path
    from automarl.components.loggers.logger_component import DEBUG_LEVEL, change_default_logger_level
    from automarl.components.loggers.global_logger import activate_global_logger

    from automarl.components.loggers.logger_component import LoggerSchema

    exec_component_input = {}

    limit_text_input = LoggerSchema.get_schema_parameter_signature("write_to_file_when_text_lines_over")
    limit_text_input.change_default_value(1000) # this is because we don't want every single thing logging and taking up resources

    exec_component_input["logger_input"] = {"write_to_file_when_text_lines_over" : -1}
    exec_component_input["logger_input"]["necessary_logger_level"] = "INFO"
    
    exec_component_input["base_directory"] = path_to_store_experiment

    if experiment_relative_path is not None:
        exec_component_input["artifact_relative_directory"] = experiment_relative_path

    if create_new_directory != None:
        if isinstance(create_new_directory, str):
            create_new_directory = True if create_new_directory.lower() in ["true", "yes", "t", "y"] else False
        exec_component_input["create_new_directory"] = create_new_directory

    

    component_configuration_path = gen_default_hp_config_path(path_to_store_experiment, experiment_relative_path) if component_configuration_path == None else component_configuration_path
    

    from automarl.components.basic_components.exec_component import ExecComponent

    # generate hp optimization pipeline component
    exec_component : ExecComponent = gen_component_from_path(component_configuration_path)


    from automarl.components.basic_components.seeded_component import SeededComponent

    if isinstance(exec_component, SeededComponent):
        exec_component_input.pass_input_if_no_value("do_full_setup_of_seed", True)
    
    #pass the defined input
    exec_component.pass_input(exec_component_input)    

    from automarl.components.basic_components.artifact_management import ArtifactComponent

    if global_logger_level != None or global_logger_path != None:

        global_logger_input = {}

        if global_logger_path == None:

            if isinstance(exec_component, ArtifactComponent):
                global_logger_path = exec_component.get_artifact_directory() 

            else:
                raise Exception(f"Could not compute globalGlobal path")

        if global_logger_level != None:
            global_logger_input["necessary_logger_level"] = global_logger_level

      
        activate_global_logger(global_logger_path, global_logger_input=global_logger_input)

    



    if default_logger_level != None:
      
        change_default_logger_level(default_logger_level)


    exec_component.run()
    


if __name__ == "__main__":

    print(f"\Executing executable component, processing input...")

    import argparse
    parser = argparse.ArgumentParser(description="Executes a runnable component")
    
    parser.add_argument("--create_new_directory", type=str, default=None, help="If a component already exists in the path, should we create a new one")

    parser.add_argument("--component_base_path", type=str, default='.', help="Directory to save results.")
    parser.add_argument("--component_relative_path", type=str, default=None, help="Relative directory to save results.")

    parser.add_argument("--component_configuration_path", type=str, default=None, help="Path of config")

    parser.add_argument("--global_logger_level", type=str, default=None, help="If to activate the global logger and with what level")
    parser.add_argument("--global_logger_path", type=str, default=None, help="If to activate the global logger and with what path")
    parser.add_argument("--default_logger_level", type=str, default=None, help="If to change the default logger level")
    

    args = parser.parse_args()

    print(f"\nStopped processing input")

    main(
         create_new_directory=args.create_new_directory,
         component_base_path=args.component_base_path, 
         component_relative_path=args.component_relative_path,
         component_configuration_path = args.component_configuration_path,
         global_logger_level = args.global_logger_level,
         global_logger_path = args.global_logger_path,
         default_logger_level=args.default_logger_level
         )
    