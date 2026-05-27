
from automarl.component import Component

def __recursive_add_components_to_return_tree(component : Component, to_return : list):

    if not component in to_return:
        to_return.append(component)

    for child_component in component:
        __recursive_add_components_to_return_tree(child_component, to_return)

def components_in_tree(component : Component):

    to_return = []

    __recursive_add_components_to_return_tree(component, to_return)

    return to_return
