'''Useful for defining custom types'''

from typing import Union
import torch

numeric_type = Union[
    int,
    float,
    torch.Tensor
]