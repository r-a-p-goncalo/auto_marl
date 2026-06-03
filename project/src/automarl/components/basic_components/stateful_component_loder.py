


import shutil
import subprocess
import time
from typing import final
from automarl.component import Component, requires_input_process
from automarl.utils.json_utils.json_component_utils import  gen_component_from
from automarl.utils.files_utils import write_text_to_file, read_text_from_file
from automarl.components.basic_components.artifact_management import ArtifactComponent, find_artifact_component_first_parent_directory
import os

from automarl.consts import CONFIGURATION_FILE_NAME

import weakref
import gc
import inspect

from automarl.components.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.components.loggers.logger_component import flush_text_of_all_loggers_and_children

from automarl.components.basic_components.exec_component import State
from automarl.components.basic_components.state_management import StatefulComponent, load_component_from_folder, save_state
import torch
                
import sys

from automarl.core.global_class_registry import get_registered_classes_generators, has_registered_classes_generators, serialize_registered_classes, load_custom_classes
   
            

class StatefulComponentLoader(StatefulComponent):
    
    '''A component with the capability of storing its state in its respective directory and later load it'''
    
    def define_component_to_save_load(self, component : ArtifactComponent):
        
        '''
        Defines an Artifact Component as the component to be saved and loaded
        That component has to already have its artifact directory generated
        '''
        
        if not isinstance(component, ArtifactComponent):
            raise Exception(f"Tried to define component to save / load that is not a ArtifactComponent, but a {type(component)}")
        
        self.component_to_save_load = component
        self.component_to_save_type = type(component)
        
        self.input["artifact_relative_directory"] = ''
        self.input["base_directory"] = str(component.get_artifact_directory())
        self.input["create_new_directory"] = False        

        
    def _process_input_internal(self):
        
        super()._process_input_internal()    
        
        if hasattr(self, 'component_to_save_load'):
        
            self.input["artifact_relative_directory"] = str(self.component_to_save_load.input["artifact_relative_directory"])
            self.input["base_directory"] = str(self.component_to_save_load.input["base_directory"])
            self.input["create_new_directory"] = False   
        

    def _save_state_internal(self):
        super()._save_state_internal()
        self.save_component()

    @requires_input_process
    def save_component(self):
        '''Saves component to its folder'''
        save_state(self.component_to_save_load, save_definition=True)

    @requires_input_process
    def unload_if_loaded(self):
        if hasattr(self, 'component_to_save_load'):
            self.unload_component() 

    def __deal_with_unwanted_unloaded_component(self, weak_ref : weakref.ReferenceType):

        leaked_obj = weak_ref()

        referrers = gc.get_referrers(leaked_obj)

        debug_lines = [
            "Component was not fully unloaded, still referenced elsewhere.",
            f"Leaked object id: {id(leaked_obj)}",
            f"Number of referrers: {len(referrers)}",
            "Referrers detail:"
        ]

        for i, ref in enumerate(referrers):
            try:
                ref_type = type(ref)
                location = ""

                # Try to locate where this referrer lives
                if hasattr(ref, "__class__"):
                    location = inspect.getmodule(ref.__class__)

                elif inspect.ismodule(ref):
                    location = ref

                debug_lines.append(
                    f"[{i}] type={ref_type}, repr={repr(ref)[:200]}, module={location}"
                )

            except Exception as e:
                debug_lines.append(f"[{i}] <error inspecting referrer: {e}>")

        raise Exception("\n".join(debug_lines))
    
    @requires_input_process
    def unload_component(self):

        '''Unloads component, note that it does not implicitly save it first'''

        # if this or any children components had dataframes, results, or something, we save them
        save_all_dataframes_of_component_and_children(self.component_to_save_load)
        flush_text_of_all_loggers_and_children(self.component_to_save_load)
                
        weak_ref = weakref.ref(self.component_to_save_load)

        del self.component_to_save_load

        gc.collect()

        if weak_ref() is not None:
            self.__deal_with_unwanted_unloaded_component(weak_ref)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Get memory before
            #before = torch.cuda.memory_allocated(device)
            
            #print(f"Memory allocated before freeing: {before} bytes, {before / (1024 * 1024)} MB")
    
            # Clean memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Optional: Collect unused IPC memor
            
            #after = torch.cuda.memory_allocated(device)
            
            #print(f"Memory allocated freed: {before - after} bytes, {(before - after) / (1024 * 1024)} MB")


    def unload_component_if_loaded_with_retries(self, number_of_times_to_try_unload=3, time_secs_to_wait=30, lg=None):

        exception = None

        for i in reversed(range(number_of_times_to_try_unload)):

            try:
                self.unload_if_loaded()
                exception = None
                break

            except Exception as e:
                exception = e
                if lg is not None:
                    lg.writeLine(f"Exception {e} when trying to unload component, will try {i} more times")
                time.sleep(time_secs_to_wait)

        if exception is not None:
            raise exception
            
        
    @requires_input_process
    def save_and_onload_component(self):
        
        self.save_component()
        self.unload_component()

        
    @requires_input_process
    def get_component(self) -> ArtifactComponent:
        '''Gets the component, if not loaded yet, it is loaded'''
        
        if not hasattr(self, 'component_to_save_load'):
            self.load_component() 
        
        return self.component_to_save_load
    
    def get_loaded_component_state(self):
        return State.clone(self.component_to_save_load.values["running_state"])
    
    @requires_input_process
    def detach_run_component(self, to_wait = False, global_logger_level = None):
        '''
        If the component is loaded, saves it and unloads it
        It then uses the load_and_run_component command to start a subprocess to run the component
        '''
        artifact_dir = self.get_artifact_directory()

        if hasattr(self, 'component_to_save_load'):
            self.save_component()
            self.unload_component()



        cmd = [
            sys.executable,
            "-m",
            "automarl.cli.commands.load_run_component",
            "--component_path",
            artifact_dir,
        ]

        if global_logger_level is not None:
            cmd = [*cmd, "--global_logger_level", global_logger_level]

        popen_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # fully detached (POSIX)
        )

        if to_wait:
            return_code = popen_process.wait()

        return popen_process
    
    @requires_input_process
    def load_component(self):
        '''
        Loads the component from the artifact directory and returns it
        '''
        
        if hasattr(self, 'component_to_save_load'):
            raise Exception("Component is already loaded, cannot load it again")
        
        self.component_to_save_load = load_component_from_folder(self.get_artifact_directory())
        
        return self.component_to_save_load