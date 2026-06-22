


import os
import shutil
import subprocess
import time
from automarl.components.basic_components.artifact_management import ArtifactComponent
from automarl.component import ParameterSignature, requires_input_process
from automarl.components.basic_components.exec_component import ExecComponent
from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.components.basic_components.evaluator_component import ComponentWithEvaluator, EvaluatorComponent
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.components.loggers.component_with_results import ComponentWithResults, DEFAULT_RESULTS_LOGGER_KEY
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.loggers.result_logger import ResultLogger
from automarl.serialization.json_component_utils import gen_component_from_dict, gen_component_from

from typing import Union

from automarl.components.basic_components.state_management import StatefulComponent
from automarl.components.basic_components.stateful_component_loder import StatefulComponentLoader

AGGREGATE_RESULTS_KEY = "aggregate_results"

Component_in_group_type = Union[ExecComponent, StatefulComponent, SeededComponent, ComponentWithResults]

class RunnableComponentGroup(SeededComponent, StatefulComponent, ComponentWithLogging, ComponentWithResults, EvaluatorComponent):

    '''A component which represents a group of runnable seeded components which can be run and evaluated'''
    
    parameters_signature = {
        
        "components_loaders_in_group" : ComponentListParameterSignature(mandatory=False)
    
    }
    
    results_columns = {
        
        DEFAULT_RESULTS_LOGGER_KEY : [],
        AGGREGATE_RESULTS_KEY : ["index", "times_ran"]
        
    }
    
    results_loggers_names = [DEFAULT_RESULTS_LOGGER_KEY, AGGREGATE_RESULTS_KEY]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.components_loaders_in_group = []

    
    def _process_input_internal(self):
        super()._process_input_internal()

        self.components_loaders_in_group : list[StatefulComponentLoader] = self.get_input_value("components_loaders_in_group")


    @requires_input_process
    def get_component(self, index) -> Component_in_group_type:
        
        '''Returns the component at the given index'''

        component_loader : StatefulComponentLoader = self.components_loaders_in_group[index]
            
        return component_loader.get_component()
    
    @requires_input_process
    def get_loader(self, index) -> StatefulComponentLoader:
         
        return self.components_loaders_in_group[index]
    
    @requires_input_process
    def unload_all_components(self):
         for loader in self.components_loaders_in_group:
              loader.unload_if_loaded()
    
    @requires_input_process
    def unload_all_components_with_retries(self, number_of_times_to_try_unload=3, time_secs_to_wait=30, lg=None):
         
        exceptions = []

        for i in reversed(range(number_of_times_to_try_unload)):
             
            exceptions = []

            for l in range(len(self.components_loaders_in_group)):
                loader = self.components_loaders_in_group[l]
                try:
                    loader.unload_if_loaded()
                except Exception as e:
                    exceptions.append((l, e))


            if len(exceptions) > 0:

                if lg is not None:

                    for (l, e) in exceptions:
                        lg.writeLine(f"Exception in loader in index {l}: {e}")

                    lg.writeLine(f"Will try to unload components {i} more times")

                time.sleep(time_secs_to_wait)

            else:
                break

        if len(exceptions) > 0:
             (l, e) = exceptions[0]
             raise Exception(f"Error unloading component {l}") from e
            
    
    # RUNNING COMPONENTS -------------------------------------------------
    
    @requires_input_process
    def detach_run_all_components(self, number_of_threads = None, global_logger_level = None):

        if number_of_threads is None:
             number_of_threads = len(self.components_loaders_in_group)

        if number_of_threads > len(self.components_loaders_in_group):
             raise Exception(f"Number of threads higher than components, {number_of_threads} > {len(self.components_loaders_in_group)}")
        
        loaders : list[StatefulComponentLoader] = list(self.components_loaders_in_group)

        running_processes: list[subprocess.Popen] = []
        active_slots = number_of_threads
        next_index = 0

        #while we still have components to run or there are running processes
        while next_index < len(loaders) or running_processes:

            # Spawn new processes while slots are available
            while active_slots > 0 and next_index < len(loaders):
                loader = loaders[next_index]

                popen_process = loader.detach_run_component(
                    to_wait=False,
                    global_logger_level=global_logger_level
                )

                running_processes.append(popen_process)
                active_slots -= 1
                next_index += 1

            # Poll running processes
            still_running = []
            for p in running_processes:
                if p.poll() is None:
                    still_running.append(p)
                else:
                    active_slots += 1  # free a slot

            running_processes = still_running

            # Avoid busy waiting
            if running_processes:
                time.sleep(10)




# CUSTOM INITIALIZATION -------------------------------------------------------------

def _create_loader_for_component(component_to_opt, loader_name):

        '''Creates a loader for a component'''

        component_saver_loader = StatefulComponentLoader({"name" : loader_name})
        component_saver_loader.define_component_to_save_load(component_to_opt)

        return component_saver_loader


def _create_loader_for_run_using_path(base_name, original_path, run_id : int, base_directory, setup_seed_of_testing):

        '''Create a loader using an existent path, copying the original component to a new path (which will be where the new loading will happen)'''

        run_name = f"{run_id}"
        component_name = f"{base_name}_{run_name}"

        component_saver_loader = StatefulComponentLoader({
            "name" : f"loader_{component_name}",
            "base_directory" : base_directory,
            "artifact_relative_directory" : run_name,
            "create_new_directory" : False})
        
        destination_path = component_saver_loader.get_artifact_directory()
        
        shutil.copytree(
            original_path,
            destination_path,
            dirs_exist_ok=True
        )                
        component_to_opt : ArtifactComponent = component_saver_loader.get_component()

        component_to_opt.clean_artifact_directory()
        
        component_to_opt.pass_input({
            "name" : component_name,
            "base_directory" : base_directory,
            "artifact_relative_directory" : run_name,
            "create_new_directory" : False
            })
        
        if setup_seed_of_testing and isinstance(component_to_opt, SeededComponent):
            component_to_opt.generate_and_setup_input_seed()
    
        elif setup_seed_of_testing:
            globalWriteLine(f"WARNING: component to test {component_to_opt.name} is not a seeded component, will not setup any stochastic environment for it")

        del component_to_opt

        component_saver_loader.save_state()

        try:
            component_saver_loader.unload_component()
        except:
             pass

        return component_saver_loader


def setup_component_group(number_of_components, group_directory, base_name, base_component_definition, setup_seed_of_testing):

        first_component : StatefulComponent = gen_component_from(base_component_definition)

        loaders : list[StatefulComponentLoader] = []
        to_return = RunnableComponentGroup({"components_loaders_in_group" : loaders, "artifact_relative_directory" : '', "base_directory" : group_directory, "create_new_directory" : False})

        run_name = f"0"
        component_name = f"{base_name}_{run_name}"
        first_component.clean_artifact_directory()
        first_component.pass_input({"base_directory" : str(group_directory), "artifact_relative_directory" : str(run_name), "name" : component_name, "create_new_directory" : False})

        if setup_seed_of_testing and isinstance(first_component, SeededComponent):
            first_component.generate_and_setup_input_seed(to_do_full_setup_of_seed=True)
    
        elif setup_seed_of_testing:
            globalWriteLine(f"WARNING: component to test is not a seeded component, will not setup any stochastic environment for it")

        loader = _create_loader_for_component(first_component, f"loader_{component_name}")
        to_return.define_component_as_child(loader)
        loader.save_component()
        loaders.append(loader)

        completed_component_directory = loader.get_artifact_directory()

        for i in range(1, number_of_components):

            loader : StatefulComponentLoader = _create_loader_for_run_using_path(base_name, completed_component_directory, i, group_directory, setup_seed_of_testing)
            to_return.define_component_as_child(loader)
            loaders.append(loader)


        return to_return
