
import shutil
from automarl.component import Component

from automarl.utils.files_utils import new_path_if_exists, open_or_create_folder

from automarl.core.input_management import ParameterSignature

import os

from automarl.utils.configuration_component_utils import save_configuration
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.consts import CONFIGURATION_FILE_NAME

def on_artifact_directory_change(self : Component):
        
    if self.has_artifact_directory_defined_or_created(): #if there is enough to create an artifact directory   

        
        if hasattr(self, "artifact_directory"):
            globalWriteLine(f"Warning: Artifact directory was changed after it already being generated: {self.artifact_directory}, will not automatically change it again")
            globalWriteLine(f"Warning: Now base directory is {self.base_directory} and relative directory is {self.artifact_relative_directory}")
        
        #else:
        #    print(f"Generating artifact directory due to change in input, instead of inpput procesing")
        #    self.get_artifact_directory()

        
def _get_artifact_parent_component(current_parent_component : Component):
                
        while current_parent_component != None: #looks for a parent component which is an Artifact Component and sets its directory based on it
            
            if isinstance(current_parent_component, ArtifactComponent):
                return current_parent_component
            
            current_parent_component = current_parent_component.parent_component

        return 0 

def define_base_directory_with_parent(self : Component):

        return _get_artifact_parent_component(self.parent_component)



def generate_target_directory(input_of_component : dict, parent_of_component=None):
    '''Generates a directory for the data of a component, without it being initialized'''

    base_directory = input_of_component.get("base_directory", '')

    if base_directory == '' and parent_of_component is not None:
        
            base_directory = _get_artifact_parent_component(self)

            if base_directory == 0:
                base_directory = ''

            
    artifact_relative_directory = input_of_component.get("artifact_relative_directory", '')

    if isinstance(base_directory, ArtifactComponent):
                base_directory = base_directory.get_artifact_directory()

    if base_directory == '' and artifact_relative_directory == '':
        return None
    
    else:
        return os.path.join(base_directory, artifact_relative_directory)

class ArtifactComponent(Component):

    '''A component which has a directory associated with it'''
    
    
    parameters_signature = {
        
                        "create_new_directory" : ParameterSignature(
                            priority=1, default_value=True, ignore_at_serialization=True,
                            description="If it is supposed to create a new directory if existent"),
        
                        "artifact_relative_directory" : ParameterSignature(
                                priority=2,
                                generator= lambda self : self.name, #the default value for the name of the artifact folder is its name
                                on_pass=on_artifact_directory_change
                                ),
                        
                        "base_directory" : ParameterSignature(
                            priority=2,
                            ignore_at_serialization=True,
                            generator=define_base_directory_with_parent,
                            on_pass=on_artifact_directory_change,
                            description='This path is used as basis to calculate the artifact directory of a component'
                        )
                       }
  
    _artifact_directory_parameters = ["artifact_relative_directory", "base_directory", "create_new_directory"]
            
            
    def _on_parent_component_defined(self):
        '''Artifact Components try to get the directory of a parent component to use as a base directory'''
        super()._on_parent_component_defined()

        if not "base_directory" in self.input.keys() and not hasattr(self, "base_directory"):
                
            new_base_directory = define_base_directory_with_parent(self)

            if new_base_directory != 0:
                self.pass_input({"base_directory" : new_base_directory})


    def clean_artifact_directory(self):

        self.remove_input("base_directory")
        self.remove_input("create_new_directory")
        self.remove_input("artifact_relative_directory") 

        if hasattr(self, "base_directory"):
            del self.base_directory
        
        if hasattr(self, "artifact_relative_directory"):
            del self.artifact_relative_directory
        
        if hasattr(self, "create_new_directory"):
            del self.create_new_directory
        
        if hasattr(self, "artifact_directory"):
            del self.artifact_directory
        
            
    
    def _ensure_artifact_directory_inputs(self):

        for key in self._artifact_directory_parameters:
            if key not in self.input:
                self.setup_default_value(key)


    def _generate_artifact_directory(self):

        necessary_parameters = self._artifact_directory_parameters

        self._ensure_artifact_directory_inputs()
        
        if not all(key in self.input.keys() for key in necessary_parameters):
            raise Exception(f'Artifact {self.name} with type {type(self)} trying to create a directory without the necessary parameters, needs {necessary_parameters} and has {self.input.keys()}')
        
        self.artifact_relative_directory = self.get_input_value("artifact_relative_directory")
        
        self.base_directory = self.get_input_value("base_directory")
        
        if self.base_directory == 0:
            self.base_directory = ''

        elif isinstance(self.base_directory, Component):
            if not isinstance(self.base_directory, ArtifactComponent):
                raise Exception(f"Passed component as base directory but is not artifact component, as it is of type: {type(self.base_directory)}")
            
            self.base_directory = self.base_directory.get_artifact_directory()


        
        self.create_new_directory = self.get_input_value("create_new_directory")
        
        if self.base_directory == '' and self.artifact_relative_directory == '':
            raise Exception("No path specification for artifact directory")
        
        try:
            full_path = os.path.join(self.base_directory, self.artifact_relative_directory)
            self.artifact_directory = open_or_create_folder(full_path, create_new=self.create_new_directory)
            
        except Exception as e:
            raise Exception(f"Component {self.name} could not open or create folder with base directory \'{self.base_directory}\' and artifact relative directory \'{self.artifact_relative_directory}\', full directory {full_path} due to exception:\n{e}") from e
        
        self.input["create_new_directory"] = False # if the component generated its directory, it won't need to create when it is loaded
                
    def _force_generate_artifact_directory(self):
        self._generate_artifact_directory()

    def __generate_base_directory(self):

        self.base_directory = self.get_input_value("base_directory")
        
        if isinstance(self.base_directory, Component):
            if not isinstance(self.base_directory, ArtifactComponent):
                raise Exception(f"Passed component as base directory but is not artifact component, as it is of type: {type(self.base_directory)}")
            
            self.base_directory = self.base_directory.get_artifact_directory()


    def get_base_directory(self):

        if not hasattr(self, "base_directory"):
            self.__generate_base_directory()

        return self.base_directory


    def generate_artifact_directory(self):
        
        '''Forces generation of artifact directory for this component and raises an exception if it already has it generated'''
        
        if hasattr(self, "artifact_directory"):
            raise Exception(f"Component {self.name} already had an artifact directory generated, \'{self.artifact_directory}\'")
            
        self._generate_artifact_directory()
        
    
    def has_artifact_directory_defined_or_created(self) -> bool:
        return self.has_artifact_directory() or all(key in self.input.keys() for key in ["artifact_relative_directory", "base_directory", "create_new_directory"])

    def has_artifact_directory(self) -> bool:
        return hasattr(self, "artifact_directory")
    
                
    def get_artifact_directory(self):
        '''Gets (and sets if needed) the artifact directory'''      
        
        if not self.has_artifact_directory():
            self._generate_artifact_directory()
            
        return self.artifact_directory
    

    
    def open_or_create_relative_folder(self, filename, sub_dir='', create_new_if_exists=True):

        '''Creates a new relative folder'''

        full_dir_path = os.path.join(self.get_artifact_directory(), sub_dir)

        returned_path = open_or_create_folder(full_dir_path, filename, create_new_if_exists)

        return os.path.relpath(returned_path, start=self.get_artifact_directory())
    


    def new_relative_path_if_exists(self, specific_path, dir = ''):

        '''Generates a string for a path, the specific path is what is used to version the path'''

        full_dir_path = os.path.join(self.get_artifact_directory(), dir)

        returned_path = new_path_if_exists(specific_path, dir=full_dir_path)

        return os.path.relpath(returned_path, start=self.get_artifact_directory())
    
    # CHANGE ARTIFACT DIRECTORY -------------------------------------------------

    def change_to_new_artifact_directory(self, new_folder_path):

        # check if new_folder_path is empty, if not, raise error
        if os.path.exists(new_folder_path) and os.listdir(new_folder_path):
            raise ValueError(f"Target folder '{new_folder_path}' is not empty.")
        
        # create new_folder in path if it does not exist
        os.makedirs(new_folder_path, exist_ok=True)

        self._change_to_new_artifact_directory_internal(new_folder_path)

        self.pass_input({"base_directory" : new_folder_path, 
                         "artifact_relative_directory" : '', 
                         "create_new_directory" : False})

        self._force_generate_artifact_directory()

        self.write_line_to_notes(f"Component had its path changed to {new_folder_path}", use_datetime=True)


    def _change_to_new_artifact_directory_internal(self, new_folder_path):
        '''What happens between the creation of the folder and the actual change of the directory'''
        os.makedirs(new_folder_path, exist_ok=True)


    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()
        
        # self.get_artifact_directory()

        
    def copy_relative_file_to(self, relative_filename, new_relative_filename, new_exists_ok=True, create_new_if_exists=False):

        old_path = os.path.join(self.get_artifact_directory(), relative_filename)

        if not os.path.exists(old_path):
            raise Exception(f"Relative filename {relative_filename} does not exist, with path {old_path}")
        
        new_path = os.join(self.get_artifact_directory(), new_relative_filename)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
        if os.path.exists(new_path):
        
            if not new_exists_ok and not create_new_if_exists:
                raise FileExistsError(f"Destination file {new_relative_filename} already exists at {new_path}")
            
            elif create_new_if_exists:
                new_path = self.new_relative_path_if_exists(new_relative_filename) 
    
        # Perform the copy
        shutil.copy2(old_path, new_path)
    
        return new_path


    def save_configuration(self, save_exposed_values=False, ignore_defaults=True, respect_ignore_order=True, config_filename=CONFIGURATION_FILE_NAME):
        
        save_configuration(self, self.get_artifact_directory(), save_exposed_values=save_exposed_values, ignore_defaults=ignore_defaults, respect_ignore_order=respect_ignore_order, config_filename=config_filename)



        

def find_artifact_component_first_parent_directory(component : Component):
    
    current_parent_component = component
    
    while current_parent_component != None: #looks for a parent component which is an Artifact Component and sets its directory based on it
        
        if isinstance(current_parent_component, ArtifactComponent):
            return current_parent_component.get_artifact_directory()
        
        current_parent_component = current_parent_component.parent_component
        
    return None
