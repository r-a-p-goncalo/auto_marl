import importlib
import os

def get_class_from(class_definition) -> type:
    
    if isinstance(class_definition, type):
        return class_definition
    
    elif isinstance(class_definition, str):
        return get_class_from_string(class_definition)
    
    else:
        raise Exception(f"Class definition {class_definition} is neither of type str nor type")
    

def is_valid_str_class_definition(class_string : str):

    return class_string.startswith("<class '") and class_string.endswith("'>")
    

def get_class_from_string(class_string: str):
    
    """
    Returns the class type from its string representation.

    Args:
        class_string (str): A string like "<class 'module.submodule.ClassName'>".

    Returns:
        type: The class type if found, otherwise raises an error.
    """
    # Extract the full path of the class
    if not is_valid_str_class_definition(class_string):
        raise ValueError(f"{class_string} has invalid class string format")
    
    full_path = class_string[len("<class '"):-len("'>")]
    
    # Split into module path and class name
    *module_parts, class_name = full_path.split('.')
    module_name = '.'.join(module_parts)
    
    # Dynamically import the module and get the class
    try:
        
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Unable to locate module '{module_name}' in class '{class_string}': {e}")
    
    except Exception as e:
        raise Exception(f"Problem when getting class with string {class_string}: {e}")
    
        
def super_class_from_list_of_class(list_of_superclasses : list[type], class_to_look_for : type) -> type:

    for super_class in list_of_superclasses:

        if issubclass(class_to_look_for, super_class):
            return super_class
        
    return None


def generate_folder_import(cls):
    """
    Given a class, generates an import statement that imports
    ALL modules from the same folder as the class's module.
    """
    module_name = cls.__module__
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return f"Could not import module {module_name}"
    
    # Get the file path of the module
    mod_file = getattr(mod, '__file__', None)
    if mod_file is None:
        return f"Module {module_name} has no __file__ attribute (possibly built-in)."
    
    folder_path = os.path.dirname(mod_file)
    folder_name = os.path.basename(folder_path)
    
    # List all .py files except __init__.py
    files = []
    for f in os.listdir(folder_path):
        if f.endswith(".py") and f != "__init__.py":
            module_base = os.path.splitext(f)[0]
            files.append(module_base)
    
    files.sort()
    
    if not files:
        return f"No modules found in {folder_name}"
    
    module_list = ", ".join(files)
    
    import_statement = f"from {folder_name} import {module_list}"

    print(f"Import statement generated:\n{import_statement}")
    eval(import_statement)


def __get_all_subclasses_recursive(cls):

    direct_subs = cls.__subclasses__()
    return direct_subs + [g for s in direct_subs for g in get_all_subclasses(s)]
    

def get_all_subclasses(cls):
    
    '''Returns a list of all subclasses of a class, including indirect ones.
    
    '''
    
    return [cls, *__get_all_subclasses_recursive(cls)]


def make_classes_in_collection_strings(collection):

    if isinstance(collection, type):
        return str(collection)
    
    elif isinstance(collection, dict):

        to_return = {}

        for key, value in collection.items():

            to_return[key] = make_classes_in_collection_strings(value)

        return to_return
    
    elif isinstance(collection, list):

        return [make_classes_in_collection_strings(element) for element in collection]
    
    elif isinstance(collection, tuple):

        return tuple(make_classes_in_collection_strings(element) for element in collection)
    
    else:
        return collection
    

def organize_collection_from_subclass_to_super_class(subclasses_list : list[type]):

    for i in range(len(subclasses_list) - 1):

        for u in range(i, len(subclasses_list) - 1):

            current_class = subclasses_list[u]
            next_class = subclasses_list[u + 1]

            if issubclass(next_class, current_class):
                subclasses_list[u] = next_class
                subclasses_list[u + 1] = current_class
    
def substitute_classes_by_subclasses(collection, subclasses_list : list[type]):

    organize_collection_from_subclass_to_super_class(subclasses_list)

    return __substitute_classes_by_subclasses(collection, subclasses_list)


def __substitute_classes_by_subclasses(collection, subclasses_list : list[type]):

    if isinstance(collection, type):
        
        for subclass in subclasses_list:
            if issubclass(subclass, collection):
                return subclass
        
        return collection
    
    elif isinstance(collection, dict):

        to_return = {}

        for key, value in collection.items():

            to_return[key] = __substitute_classes_by_subclasses(value)

        return to_return
    
    elif isinstance(collection, list):

        return [__substitute_classes_by_subclasses(element) for element in collection]
    
    elif isinstance(collection, tuple):

        return tuple(__substitute_classes_by_subclasses(element) for element in collection)
    
    else:
        return collection

    
