

import os
import pandas
from automarl.components.loggers.global_logger import globalWriteLine



def open_or_create_folder(dir, folder_name='', create_new=True):
    
    '''
    If create_new == True, we make a new folder in the case it exists
    '''

    if folder_name == '':
        dir, folder_name = os.path.split(dir) #if the last folder/file name is not defined, extract it from dir
    
    full_path = os.path.join(dir, folder_name)        

    dir = os.path.dirname(full_path)
    folder_name = os.path.basename(full_path)

    try:
        folders_in_dir = os.listdir(dir) #is the dir already created? if not, we create it
        
    except:
        try:
            os.makedirs(dir)
        except:
            globalWriteLine(f"When creating directory {dir}, error, while it did not exist before. If it exists now, this means badly written parallel code that is trying to create the same directory")
        
        folders_in_dir = os.listdir(dir)
    
    folder_exists = os.path.exists(full_path) and os.path.isdir(full_path)

    full_path = os.path.join(dir, folder_name)
    
    if folder_exists and create_new: 
            
        number_of_versioned_folders = len([l for l in folders_in_dir if l.startswith(f"{folder_name}_") and os.path.isdir(os.path.join(dir, l))]) #counts the number of dirs that start with specified name
    
        folder_name = f"{folder_name}_{number_of_versioned_folders}"
    
        full_path = f"{dir}\\{folder_name}"
    
        os.makedirs(full_path)

    elif not folder_exists:

        os.makedirs(full_path)

    # else folder exists and create_new is False
    
    return full_path

def new_path_if_exists(specific_path, dir = ''):

    '''Generates a string for a path, the specific path is what is used to version the path'''

    full_path = os.path.join(dir, specific_path)

    dir = os.path.dirname(full_path)
    specific_path = os.path.basename(full_path)
    

    if os.path.exists(full_path): #file with that name already existed

        paths_in_dir = os.listdir(dir)

        if os.path.isfile(full_path):
            
            filename, ext = os.path.splitext(specific_path)

            # find existing versions like filename_1.ext, filename_2.ext, ...
            existing_versions = [
                f for f in paths_in_dir
                if f.startswith(filename + "_") and f.endswith(ext)
            ]

            version_number = len(existing_versions) + 1
            specific_path = f"{filename}_{version_number}{ext}"
            full_path = os.path.join(dir, specific_path)

        else: # if is dir
        
            number_of_versioned_paths = len([l for l in paths_in_dir if l.startswith(f"{specific_path}_")])

            specific_path = f"{specific_path}_{number_of_versioned_paths}"

            full_path = os.path.join(dir, specific_path)

    return full_path




def write_text_to_file(dir = '', filename = '', text : str = '', create_new=True):
    
    full_path = os.path.join(dir, filename)
    
    dir = os.path.dirname(full_path)
    
    os.makedirs(dir, exist_ok=True)
    
    # If the file exists and create_new is True, delete old and write new
    if os.path.exists(full_path):
        
        if create_new:
                        
            with open(full_path, 'w') as f:
                f.write(text)
        
        else:
            
            with open(full_path, 'a') as f:
                f.write(text)

    else:
        with open(full_path, 'w') as f: # write to file
            f.write(text)


    
def read_text_from_file(dir='', filename=''):
    
    if dir == '' and '':
        raise Exception()
    
    if filename != '':
        full_path = os.path.join(dir, filename)

    else:
        full_path = dir

    # Append the text to the file
    with open(full_path, 'r') as f:
        return f.read()


def saveDataframe(df : pandas.DataFrame, directory='', filename='dataframe.csv'): 
                
        if(directory != ''):
            open_or_create_folder(directory, create_new=False)

        df.to_csv(os.path.join(directory, filename), index=False)


def loadDataframe(directory='', filename='dataframe.csv') -> pandas.DataFrame:
        
    # Build full path
    full_path = os.path.join(directory, filename)

    # Safety: check existence
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataframe file not found: {full_path}")

    # Read CSV
    return pandas.read_csv(full_path)



def get_first_path_with_name(base_path, name):
    """
    Returns the full path of the first directory or file with this exact name
    that is a subdirectory or file inside base_path (recursive search).
    If not found, returns None.
    """

    # Safety: if base_path does not exist
    if not os.path.exists(base_path):
        return None

    # Walk the directory tree
    for root, dirs, files in os.walk(base_path):
        
        # Check directories
        if name in dirs:
            return os.path.join(root, name)

        # Check files
        if name in files:
            return os.path.join(root, name)

    # Nothing found
    return None