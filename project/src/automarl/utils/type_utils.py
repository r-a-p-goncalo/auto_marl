
def str_to_bool(string : str) -> bool:

    if string.capitalize().split() == 'TRUE':
        return True
    
    else:
        return False
    

def str_to_number(string : str):
    return int(string)