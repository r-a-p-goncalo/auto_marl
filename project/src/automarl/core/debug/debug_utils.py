
from automarl.component import Schema, get_class_from
from automarlcore.global_class_registry import register_custom_class
from automarl.components.fundamentals.translator.translator import Component
from automarl.utils.class_util import is_valid_str_class_definition, organize_collection_from_subclass_to_super_class

def get_non_debug_schema_of_schema(schema : type[Component]):
    
    if not schema.is_debug_schema:
        return None

    elif schema.target_debug_class is not None:
        return schema.target_debug_class
    
    else:

        schema_mro = schema.__mro__

        for schema_superclass_index in range(1, len(schema_mro)):
            schema_superclass = schema_mro[schema_superclass_index]

            if not issubclass(schema_superclass, Component):
                return None
            
            elif not schema_superclass.is_debug_schema:
                return schema_superclass
                
        return None
    


def generate_pairs_class_debug_class(debugclasses_list : list[type[Component]]) -> list[tuple[type[Component], type[Component]]]:

    debug_classes_and_classes : list[tuple[type[Component], type[Component]]] = []

    for debug_class in debugclasses_list:

        class_of_debug_class : type[Component] = get_non_debug_schema_of_schema(debug_class)

        if class_of_debug_class is not None:
            debug_classes_and_classes.append((class_of_debug_class, debug_class))
            print(f"debug -> base_class: {debug_class} -> {class_of_debug_class}")

        else:
            print(f"Could not find a base class for debug class : {debug_class}")

    return debug_classes_and_classes



def substitute_classes_by_debug_classes(collection, debugclasses_list : list[Component]):

    for debug_class in debugclasses_list:
        if not debug_class.is_debug_schema:
            raise Exception(f"Class {debug_class} passed as debug class, but is not registered as such")

    debug_classes_and_classes = generate_pairs_class_debug_class(debugclasses_list)

    return __substitute_classes_by_debugclasses(collection, debug_classes_and_classes)




def __substitute_classes_by_debugclasses(collection, class_debugclasses_pairs : list[tuple[type[Component], type[Component]]]):

    if isinstance(collection, str): # translate into direct type if is type in string
        if is_valid_str_class_definition(collection):
            collection = get_class_from(collection)

    if isinstance(collection, type):
        
        for (subclass, debug_class) in class_debugclasses_pairs:

            if subclass == collection:

                print(f"\nFound class {collection}, substituting it by debugclass {debug_class}")
                
                return debug_class

            elif issubclass(collection, subclass):

                new_debug_class : type[Component] = register_custom_class(name=f"{debug_class.__name__}_{collection.__name__}",bases=(debug_class, collection))
                                
                print(f"\nFound class {collection}, substituting it by debugclass {new_debug_class}, using debug class {debug_class}")
                print(f"    parameters: {[key for key in new_debug_class.get_schema_parameters_signatures().keys()]}")
                print(f"    exp values: {[key for key in new_debug_class.get_schema_exposed_values().keys()]}")
                print(f"    mro:        {new_debug_class.__mro__}")
                
                return new_debug_class
                    
        return collection
    
    elif isinstance(collection, dict):

        to_return = {}

        for key, value in collection.items():

            to_return[key] = __substitute_classes_by_debugclasses(value, class_debugclasses_pairs)

        return to_return
    
    elif isinstance(collection, list):

        return [__substitute_classes_by_debugclasses(element, class_debugclasses_pairs) for element in collection]
    
    elif isinstance(collection, tuple):

        return tuple(__substitute_classes_by_debugclasses(element, class_debugclasses_pairs) for element in collection)
    
    else:
        return collection