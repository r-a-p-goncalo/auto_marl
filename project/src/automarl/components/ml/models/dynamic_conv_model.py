


from automarl.components.hp_opt.hp_suggestion.list_hp_suggestion import DictHyperparameterSuggestion
from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion
from automarl.components.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn

from automarl.components.ml.models.model_components import ModelComponent

from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.component import ParameterSignature
from automarl.utils.shapes_util import  discrete_output_layer_size_of_space


class DynamicConvModelSchema(TorchModelComponent):

    class Model_Class(nn.Module):

        def __init__(self, input_shape, cnn_layers, output_size=None):
            """
            input_shape: (C, H, W)
            cnn_layers: list of dicts with out_channels, kernel_size, stride
            """

            super().__init__()

            C, H, W = input_shape

            self.input_shape = [C, H, W]

            layers = []
            in_channels = C

            # Build CNN layers
            for layer_cfg in cnn_layers:
                k = layer_cfg["kernel_size"]
                s = layer_cfg["stride"]
                oc = layer_cfg["out_channels"]

                layers.append(nn.Conv2d(in_channels, oc, k, s))
                layers.append(nn.ReLU())

                # compute next H, W
                H = (H - k) // s + 1
                W = (W - k) // s + 1

                in_channels = oc

            self.feature_extractor = nn.Sequential(*layers)

            self.last_H = H
            self.last_W = W
            self.last_out_channels = oc

            # Flatten size after CNN
            flatten_size = in_channels * H * W
            self.flatten_size = flatten_size

            self.output_size = output_size
            if output_size != None:
                self.output_layer = nn.Linear(flatten_size, output_size)

            else:
                self.output_layer = None

        def forward(self, x):
            x = self.feature_extractor(x)

            if self.output_layer is not None:
                x = torch.flatten(x, 1)
                return self.output_layer(x)
            
            return x
        
        def get_input_shape(self):
           return self.input_shape
       
        def get_output_shape(self):
           if self.output_size is not None:
               return self.output_size
           else:
               return [self.last_out_channels, self.last_H, self.last_W]

    # ----------------------------------------------------------
    # INPUT SIGNATURE
    # ----------------------------------------------------------

    parameters_signature = {

        "cnn_layers": ParameterSignature(
            custom_dict={"hyperparameter_suggestion": 
                VariableListHyperparameterSuggestion(
                    name='cnn_layers',
                    min_len=2,
                    max_len=4,
                    hyperparameter_suggestion_for_list=DictHyperparameterSuggestion(
                        hyperparameter_suggestions={
                            "out_channels" : SingleHyperparameterSuggestion(name="out_channels",
                                                           value_suggestion=("cat", {"choices": [16, 32, 64, 128]})
                            ),
                            "kernel_size" : SingleHyperparameterSuggestion(name="kernel_size",
                                                           value_suggestion=("cat", {"choices": [3, 5, 7]})
                            ),
                            "stride" : SingleHyperparameterSuggestion(name="stride",
                                                           value_suggestion=("cat", {"choices": [1, 2]})
                            )
                        }
                    )
                )
            }  
        ),

    }

    # ----------------------------------------------------------

    def _setup_values(self):
        super()._setup_values()

        if self.input_shape is None:
            raise Exception(f"{type(self)} requires input_shape")

        if self.output_shape is not None:
            self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)

        else:
            self.output_size = None

        # CNN layers list (list of dicts)
        self.cnn_layers = self.get_input_value("cnn_layers")


        self.lg.writeLine(f"Passed CNN layers:")

        for layer in self.cnn_layers:
            self.lg.writeLine(layer)

        self._correct_cnn_layers()

        self.lg.writeLine(f"Corrected CNN layers:")

        for layer in self.cnn_layers:
            self.lg.writeLine(f"    {layer}")

    
    def _correct_cnn_layers(self):

        C, H, W = self.input_shape
        layers = self.cnn_layers

        # We will compute the output sizes backwards
        # desired minimum valid size at the final layer output is 1×1
        min_h = 1
        min_w = 1

        corrected = []

        # --- PASS 1: Backward validation ---
        # Start from the last layer and go backwards, correcting invalid parameters
        for cfg in reversed(layers):
            k = cfg["kernel_size"]
            s = cfg["stride"]

            # Ensure kernel never exceeds current expected minimum spatial size
            if k > H:   k = H
            if k > W:   k = W
            if k < 1:   k = 1

            # Ensure stride cannot exceed kernel size
            if s > k: s = k

            # Recompute the required input sizes for this layer:
            # H_in = s*(H_out - 1) + k
            H_in = s * (min_h - 1) + k
            W_in = s * (min_w - 1) + k

            # Prepare corrected layer
            corrected.append({
                "out_channels": cfg["out_channels"],
                "kernel_size":  k,
                "stride":       s,
            })

            # Update required minimum input size for next iteration
            min_h, min_w = H_in, W_in

        # Reverse back to original order
        corrected.reverse()

        # --- PASS 2: Forward check to ensure feasible from actual input ---
        # Clip layers if they produce invalid shapes (too small)
        Htest, Wtest = H, W
        final_layers = []

        for cfg in corrected:
            k = cfg["kernel_size"]
            s = cfg["stride"]

            # Ensure valid convolution
            if Htest < k or Wtest < k:
                # Stop adding more layers
                break

            Hnext = (Htest - k) // s + 1
            Wnext = (Wtest - k) // s + 1

            if Hnext < 1 or Wnext < 1:
                break

            final_layers.append(cfg)
            Htest, Wtest = Hnext, Wnext

        # Store corrected
        self.cnn_layers = final_layers


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
            output_size=self.output_size,
            cnn_layers=self.cnn_layers
        )

    def _is_model_well_formed(self):
        super()._is_model_well_formed()
        # You could add conv shape checks here
