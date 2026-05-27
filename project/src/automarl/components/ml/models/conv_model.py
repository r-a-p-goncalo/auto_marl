


from automarl.components.hp_opt.hp_suggestion.list_hp_suggestion import DictHyperparameterSuggestion
from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion
from automarl.components.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn

from ...component import ParameterSignature
import torch
import random
import math
import numpy as nn

from automarl.components.ml.models.model_components import ModelComponent
import torch
import torch.nn as nn
import torch.nn.functional as F

from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.component import ParameterSignature
from automarl.utils.shapes_util import  discrete_output_layer_size_of_space


class ConvModel(TorchModelComponent):

    class Model_Class(nn.Module):

        def __init__(self, input_shape, output_size):
            """
            input_shape: (C, H, W)
            """
            super().__init__()

            C, H, W = input_shape
            self.input_shape = input_shape
            self.output_size = output_size


            self.conv1 = nn.Conv2d(C, 32, kernel_size=8, stride=4)   # 280→70, 480→120
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 70→35, 120→60
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # 35→33, 60→58

            self.res1 = self._res_block(128)
            self.res2 = self._res_block(128)

            dummy = torch.zeros(1, *input_shape)
            with torch.no_grad():
                n_flat = self._forward_conv(dummy).view(1, -1).shape[1]

            # ----------------------------------------------------
            # Final fully connected head
            # ----------------------------------------------------
            self.fc = nn.Linear(n_flat, 512)
            self.policy = nn.Linear(512, output_size)  # supports discrete action spaces

        # --------------------------------------------------------
        # Residual block (2x conv)
        # --------------------------------------------------------
        def _res_block(self, channels):
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1)
            )

        # --------------------------------------------------------
        # Forward CNN feature extractor only
        # --------------------------------------------------------
        def _forward_conv(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.res1(x) + x)
            x = F.relu(self.res2(x) + x)
            return x
        
    # ----------------------------------------------------------
    # INPUT SIGNATURE
    # ----------------------------------------------------------

    parameters_signature = {

    }

    # ----------------------------------------------------------

    def _setup_values(self):

        super()._setup_values()

        if self.input_shape is None:
            raise Exception(f"{type(self)} requires input_shape")

        if self.output_shape is None:
            raise Exception(f"{type(self)} requires output_shape")

        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)



    def _initialize_mininum_model_architecture(self):
        super()._initialize_mininum_model_architecture()

        self._setup_values()

        # Create minimal CNN
        self.model = type(self).Model_Class(
            input_shape=self.input_shape,
            output_size=self.output_size,
            cnn_layers=self.cnn_layers
        )

    def _initialize_model(self):
        super()._initialize_model()

        # Same as minimum but new weights
        self.model = type(self).Model_Class(
            input_shape=self.input_shape,
            output_size=self.output_size
            )

    def _is_model_well_formed(self):
        super()._is_model_well_formed()
        # You could add conv shape checks here
