

from automarl.components.fundamentals.translator.translator import Translator
from automarl.core.input_management import ParameterSignature
from automarl.utils.shapes_util import torch_shape_from_space
import torch

class ImageReverter(Translator):

    parameters_signature = {
        }    

    def _process_input_internal(self):
        
        super()._process_input_internal()

    def _setup_shape_cache(self):

        super()._setup_shape_cache()

        if self.buffered_operations: # if we are to initialize buffer
            self._out_buffer = torch.empty(self.new_shape, dtype=torch.float32, device=self.device)
        
        else:
            self._out_buffer = None


    def translate_state(self, state : torch.Tensor):

        """
        Transforms a pixel observation (H,W,C) -> torch(C,H,W)
        """

        if state is None:
            return None
        
        with torch.no_grad(): 
        
            if self._out_buffer == None:
                return state.permute(2, 0, 1)
            
            else:

                # tmp is (H, W, C). We need (C, H, W) in buffer.
                self._out_buffer.copy_(state.permute(2, 0, 1))

                return self._out_buffer


    def _get_shape(self, original_shape):

        shape = torch_shape_from_space(original_shape)  # torch.Size([H, W, C])

        if len(shape) != 3:
            raise ValueError(f"ImageReverter expected a 3D shape (H,W,C), got: {shape}")

        H, W, C = shape
        return torch.Size([C, H, W])
    

class ImageSingleChannel(Translator):

    parameters_signature = {
        }    

    def _process_input_internal(self):
        
        super()._process_input_internal()


    def translate_state(self, state : torch.Tensor):

        """
        Only uses the first of 3 channels
        """

        if state is None:
            return None

        return state[0]
    

    def _get_shape(self, original_shape):

        shape = torch_shape_from_space(original_shape)  # torch.Size([H, W, C])

        if len(shape) != 3:
            raise ValueError(f"ImageSingleChannel expected a 3D shape (C,H, W), got: {shape}")

        C, H, W= shape
        return torch.Size([H, W])
    


class ImageNormalizer(Translator):

    parameters_signature = {
        }    



    def translate_state(self, state : torch.Tensor):

        """
        Transforms a pixel observation (H,W,C) -> torch(C,H,W) normalized.
        """


        if state is None:
            return None
        
        with torch.no_grad(): 
        
            if self.in_place_translation:
                state.div_(255.0)
                return state

            else:
                return state / 255.0 

        
    def _get_shape(self, original_shape):
        return original_shape
    
    

class ImageReverterToSingleChannel(Translator):

    '''Directly reverts an image to single channel'''

    parameters_signature = {
        }    
    

    def _process_input_internal(self):
        super()._process_input_internal()


    def _setup_shape_cache(self):
        super()._setup_shape_cache()

        if self.buffered_operations: # if we are to initialize buffer
            self._out_buffer = torch.empty(self.new_shape, dtype=torch.float32, device=self.device)
        
        else:
            self._out_buffer = None

    
    def translate_state(self, state : torch.Tensor):
    
        if self._out_buffer == None:
            return state[:, :, 0]
        
        else:
            self._out_buffer._copy(state[:, :, 0])
            return self._out_buffer


    
    def _get_shape(self, original_shape):
        """
        Expected input shape: (H, W, C)
        Output shape: (H, W)
        """

        shape = torch_shape_from_space(original_shape)  # torch.Size([H, W, C])

        if len(shape) != 3:
            raise ValueError(f"ImageReverterToSingleChannel expected a 3D shape (H,W,C), got: {shape}")

        H, W, C = shape
        return torch.Size([H, W])