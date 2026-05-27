def substitute_value_in_dict(dict_with_value : dict, key, new_value):
    print(f"Old value for key '{key}': {dict_with_value.get(key, None)}, new value: {new_value}")
    dict_with_value[key] = new_value

def remove_value_in_dict(dict_with_value : dict, key, new_value):
    print(f"Old value for key '{key}': {dict_with_value.get(key, None)}, to be removed...")
    dict_with_value.pop(key, None)



def substitute_tuple_value_in_dict(dict_with_tuple : dict, key, tuple_index, new_value):

    tuple_value : tuple = dict_with_tuple[key]

    print(f"Old value for tuple pos {tuple_index}: {tuple_value[tuple_index]}, new value: {new_value}")
    new_tuple_value = tuple( new_value if tuple_index == i else tuple_value[i] for i in range(len(tuple_value)) )

    dict_with_tuple[key] = new_tuple_value


def substitute_list_value_in_dict(dict_with_tuple : dict, key, tuple_index, new_value):

    tuple_value : tuple = dict_with_tuple[key]

    print(f"Old value for tuple pos {tuple_index}: {tuple_value[tuple_index]}, new value: {new_value}")
    new_tuple_value = [ new_value if tuple_index == i else tuple_value[i] for i in range(len(tuple_value)) ]

    dict_with_tuple[key] = new_tuple_value