from automarl.utils.files_utils import write_text_to_file
from automarl.utils.type_utils import str_to_bool, str_to_number



def generate_path(component_path, target_path_dir='', target_path_name=None):

    import shutil
    from automarl.components.basic_components.artifact_management import open_or_create_folder

    target_path_dir = '' if target_path_dir is None else target_path_dir

    experiment_path = open_or_create_folder(target_path_dir, target_path_name, create_new=True) if target_path_name is not None else target_path_dir

    shutil.copytree(component_path, experiment_path, dirs_exist_ok=True)

    return experiment_path


def load_component(component_path):

    from automarl.serialization.json_component_utils import gen_component_from_path

    component = gen_component_from_path(component_path)

    return component

def run_component(component):

    #component.pass_input({"logger_input" : {"necessary_logger_level" : "INFO"}})
    component.run()


def main(component_path, target_dir=None, target_dir_name=None, 
         global_logger_level=None, logger_input_level=None, 
         lines_until_logger_write=None, save_results_on_log=None):
    
    # LOADING THE COMPONENT

    if target_dir is not None or target_dir_name is not None:
        component_path = generate_path(component_path, target_dir, target_dir_name)
    
    # ACTIVATING GLOBAL LOGGER BEFORE LOADING

    if global_logger_level != None:
        try:
            from automarl.components.loggers.global_logger import activate_global_logger, globalWriteLine
            activate_global_logger(component_path, global_logger_input={"necessary_logger_level" : global_logger_level})
        
        except Exception as e:
            print(f"Error trying to activate global logger: {e}")
            write_text_to_file(component_path, filename="error_global.txt", text=str(e))

    # FINISHING COMPONENT LOADING

    try:
        component = load_component(component_path)

    except Exception as e:
        from automarl.components.loggers.global_logger import get_global_logger

        global_logger = get_global_logger()

        if global_logger is not None:

            from automarl.core.exceptions import common_exception_handling
            common_exception_handling(global_logger, e, "on_load_error.txt")

        raise e
    
    # COMPONENT INPUT PRE PROCESSING

    try:

        logger_input = {}

        if logger_input_level is not None:
            logger_input["necessary_logger_level"] = logger_input_level

        if lines_until_logger_write is not None:
            logger_input["write_to_file_when_text_lines_over"] = str_to_number(lines_until_logger_write)

        if logger_input != {}:
            prev_logger_input = component.input.get("logger_input", {})
            logger_input = {**prev_logger_input, **logger_input}

            component.pass_input({"logger_input" : logger_input})

        if save_results_on_log is not None:
            prev_results_logger_input = component.input.get("results_logger_input", {})

            component.pass_input({"results_logger_input": {**prev_results_logger_input, "save_results_on_log" : str_to_bool(save_results_on_log)}})

    except Exception as e:
        from automarl.components.loggers.global_logger import get_global_logger

        global_logger = get_global_logger()

        if global_logger is not None:

            from automarl.core.exceptions import common_exception_handling
            common_exception_handling(global_logger, e, "on_input_handling_error.txt")

        raise e

    # RUNNING THE COMPONENT

    try:
        run_component(component)

    except Exception as e:
        from automarl.components.loggers.global_logger import get_global_logger

        global_logger = get_global_logger()

        if global_logger is not None:

            from automarl.core.exceptions import common_exception_handling
            common_exception_handling(global_logger, e, "on_run_error.txt")

        raise e        


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization pipeline.")

    
    parser.add_argument("--component_path", type=str, default='.', help="Path of experiment")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory where to put the component run")
    parser.add_argument("--target_dir_name", type=str, default=None, help="Last folder where to put the run")
    parser.add_argument("--global_logger_level", type=str, default=None, help="If to activate global logger and with what level")
    
    args = parser.parse_args()

    main(
        component_path=args.component_path,
        target_dir=args.target_dir,
        target_dir_name=args.target_dir_name,
        global_logger_level=args.global_logger_level
         )

    
