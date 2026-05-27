
import random
import numpy
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.components.loggers.logger_component import DEBUG_LEVEL
import torch

SEED_GLOBAL_LOGGER = "seed_logging.txt"

is_full_seed_setup_done = False


def generate_and_setup_a_seed():

    '''
    Generates a seed and sets up the whole stochastic process with it
    '''
    
    seed = generate_seed()

    do_full_setup_of_single_seed(seed)
    
    return seed



def generate_seed():
    return random.randint(0, 2**32 - 1)


def generate_seed_configuration(seed_configuration : dict = None):

    seed_configuration = {} if seed_configuration == None else {**seed_configuration}

    if seed_configuration.get("python", None) == None:
        seed_configuration["python"] = generate_seed()

    if seed_configuration.get("torch", None) == None:
        seed_configuration["torch"] = generate_seed()

    if seed_configuration.get("numpy", None) == None:
        seed_configuration["numpy"] = generate_seed()

    return seed_configuration


def setup_seed_from_dict_configuration(seed_configuration : dict = None, force_new = False):

    '''
    Sets up seed for other libraries using the respective keys
    '''

    global is_full_seed_setup_done

    if not is_full_seed_setup_done or force_new:

        seed_configuration = generate_seed_configuration(seed_configuration)

        globalWriteLine(f"Setting up seeds:", file=SEED_GLOBAL_LOGGER)

        setup_python_seed(seed_configuration["python"])

        setup_torch_seed(seed_configuration["torch"])

        setup_numpy_seed(seed_configuration["numpy"])

        is_full_seed_setup_done = True

        return seed_configuration

    else:
        globalWriteLine(f"Tried to setup seeds when seeds were already setup, ignoring it...", file=SEED_GLOBAL_LOGGER)
                     
        return None    

def do_full_setup_of_single_seed(seed, force_new=False):

    '''
    Does full setup with a single seed
    '''

    setup_python_seed(seed)
    
    return setup_seed_from_dict_configuration(
        {
            "python" : seed
        },
        force_new=force_new
    )


def setup_python_seed(seed):
    globalWriteLine(f"    Python seed {seed}", file=SEED_GLOBAL_LOGGER)
    random.seed(seed)    


def setup_numpy_seed(seed):
    globalWriteLine(f"    Numpy seed {seed}", file=SEED_GLOBAL_LOGGER)
    numpy.random.seed(seed)     # NumPy    


def setup_torch_seed(seed):

    globalWriteLine(f"    Torch (cuda included) seed {seed}", file=SEED_GLOBAL_LOGGER)
    
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state() -> str:

    '''
    Returns a string representing the whole random state
    '''
    
    python_state_str = str(random.getstate())
    numpy_state_str = str(numpy.random.get_state())
    torch_state_str = str(torch.get_rng_state())
    torch_cuda_state = str(torch.cuda.get_rng_state_all())

    return f"Python:\n{python_state_str}\n\nNummpy:\n{numpy_state_str}\n\nTorch:\n{torch_state_str}\n\nTorch cuda:\n{torch_cuda_state}"