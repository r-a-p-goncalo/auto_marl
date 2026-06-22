import json
import numpy as np
import gymnasium as gym
from automarl.serialization.custom_json_logic import CustomJsonLogic, register_custom_strategy

'''
Has custom class for encoding and decoding states
Importing this has the inherit effect of adding the encoder / decoder strategy to the registry
'''


def compress_bound(bound):

    '''Compresses an array of the same value into itself'''

    if isinstance(bound, np.ndarray):
        if np.all(bound == bound.flat[0]):  # constant array, return only its value
            return float(bound.flat[0])
        return bound.tolist()               # true per-dimension bounds
    return float(bound)

def expand(bound, shape):
    if np.isscalar(bound):
        return np.full(shape, bound, dtype=np.float32)
    return np.array(bound, dtype=np.float32)


class CustomSpaceJsonEncoderDecoder(CustomJsonLogic):


    class CustomSpaceEncoder(json.JSONEncoder):
        
        '''Encodes elements of a component input or exposed value, which can be a component (defined by its localization) or a primitive type'''
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
                    
        def default(self, space):

            if isinstance(space, gym.spaces.Discrete):
                space_dictionary =  {
                    "space_type": "Discrete",
                    "n": int(space.n)
                }

            elif isinstance(space, gym.spaces.Box):

                space_dictionary = {
                    "space_type": "Box",
                    "shape": space.shape,
                    "low": compress_bound(space.low),
                    "high": compress_bound(space.high),
                    "dtype": str(space.dtype)
                }


            elif isinstance(space, gym.spaces.MultiBinary):
                space_dictionary =  {
                    "space_type": "MultiBinary",
                    "n": int(space.n)
                }

            elif isinstance(space, gym.spaces.MultiDiscrete):
                space_dictionary =  {
                    "space_type": "MultiDiscrete",
                    "nvec": space.nvec.tolist()
                }

            else:
                return super().default(space) # this is for the case where there are elements which should be serialized using the common strategy, such as a dictionary of spaces


            return space_dictionary



    def to_dict(space):
        
        encoder = CustomSpaceJsonEncoderDecoder.CustomSpaceEncoder()
        
        return encoder.default(space) # we call explicitly our default method so we have full control over the logic of serialization



    def from_dict(dict, element_type, decode_elements_fun, source_component):

        """Reconstruct a Gym/Gymnasium space from its JSON dict."""
        
        type = dict["space_type"]

        space_to_return = None

        if type == "Discrete":
            space_to_return =  gym.spaces.Discrete(dict["n"])
        
        elif type == "Box":

            low  = expand(dict["low"],  tuple(dict["shape"]))
            high = expand(dict["high"], tuple(dict["shape"]))

            space_to_return = gym.spaces.Box(
                low=low,
                high=high,
                dtype=np.dtype(dict["dtype"])
            )
            
        elif type == "MultiBinary":
            space_to_return =  gym.spaces.MultiBinary(dict["n"])
        
        elif type == "MultiDiscrete":
            space_to_return =  gym.spaces.MultiDiscrete(np.array(dict["nvec"]))
        
        else:
            raise ValueError(f"Unknown space type: {type}")
        
        return space_to_return
    

register_custom_strategy(gym.spaces.Space, CustomSpaceJsonEncoderDecoder)