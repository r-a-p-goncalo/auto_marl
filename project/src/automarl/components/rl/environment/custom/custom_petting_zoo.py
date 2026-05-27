
from automarl.schema import ParameterSignature
import numpy as np
from pettingzoo.butterfly.cooperative_pong.cooperative_pong import CooperativePong
from automarl.components.rl.environment.parallel_environment import ParallelEnvironmentComponent
import torch

class TorchCooperativePong(CooperativePong, ParallelEnvironmentComponent):

    raise NotImplementedError()

    parameters_signature = {
        "device" : ParameterSignature(get_from_parent=True)
    }


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.render_mode == 'human':
            raise Exception(f"TensorCoopeativePong does not support human render mode")

        self.device = self.get_input_value("device")

        obs_h = int(self.s_height * 2 / self.kernel_window_length)
        obs_w = int(self.s_width * 2 / self.kernel_window_length)

        # complete this, create torch buffer
        self.area = None
        self.screen = None

        self._np_obs_buffer = np.zeros((obs_h, obs_w, 3), dtype=np.uint8)
        self._np_state_buffer = np.zeros((self.s_height, self.s_width, 3), dtype=np.uint8)

        # torch tensors sharing memory with numpy (zero-copy)
        self.obs_tensor = torch.from_numpy(self._np_obs_buffer).to(self.device)
        self.state_tensor = torch.from_numpy(self._np_state_buffer).to(self.device)

    def observe(self):
        # return torch tensor directly
        return self.obs_tensor

    def state(self):
        # return torch tensor directly
        return self.state_tensor

