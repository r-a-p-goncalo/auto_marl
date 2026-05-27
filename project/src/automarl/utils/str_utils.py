
def generate_unique_versioned_str(base_str : str, current_collection : str):

    if base_str in current_collection:

        i = 1
        str_to_return = f"{base_str}_{i}"

        while str_to_return in current_collection:
            i += 1
            str_to_return = f"{base_str}_{i}"

        return str_to_return
    
    else:
        return base_str
    


def generate_str_fixed_chars(base_str: str, n_chars: int = 35) -> str:
    """Return a string with exactly `n_chars` characters.

    If the input is shorter, it is right-padded with spaces.
    If it is longer, it is truncated in the middle using ' ... '.
    """

    if not isinstance(base_str, str):
        base_str = str(base_str)

    if n_chars < 6:
        raise ValueError("n_chars must be at least 6")

    if len(base_str) == n_chars:
        return base_str

    if len(base_str) < n_chars:
        return base_str + " " * (n_chars - len(base_str))

    separator = " ... "
    remaining = n_chars - len(separator)

    left = remaining // 2
    right = remaining - left

    return base_str[:left] + separator + base_str[-right:]