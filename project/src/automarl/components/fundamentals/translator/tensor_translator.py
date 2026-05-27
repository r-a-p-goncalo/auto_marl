

from automarl.components.fundamentals.translator.translator import Translator
from automarlcore.input_management import ParameterSignature
from automarl.utils.shapes_util import torch_shape_from_space
import torch
import numpy


class ToTorchTranslator(Translator):

    parameters_signature = {
        }    

    def _process_input_internal(self):
        
        super()._process_input_internal()


    def translate_state(self, state):

        """
        Transforms a pixel observation (H,W,C) -> torch(C,H,W)
        """

        if state is None:
            return None
        
        return torch.tensor(state, dtype=torch.float32, device=self.device)
        

    def _get_shape(self, original_shape):

        shape = torch_shape_from_space(original_shape)  # torch.Size([H, W, C])

        return shape
    