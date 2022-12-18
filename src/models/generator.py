import torch
import torch.nn.functional as F

from typing import List

from src.config import GeneratorConfig
from src.utils import dilation2padding


class ResBlock(torch.nn.Module):
    def __init__(self, n_channels: int, dilations: List[List[int]], kernel_size: int, lrelu_slope: float):
        self.blocks = torch.nn.ModuleList()
        for continual_dilations in dilations:
            layers = []
            for dilation in continual_dilations:
                layers += [
                    torch.nn.LeakyReLU(negative_slope=lrelu_slope),
                    torch.nn.Conv1d(
                        in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation2padding(kernel_size, dilation)
                    )
                ]
            self.blocks.append(
                torch.nn.Sequential(layers)
            )

    def forward(self, x):
        for block in self.blocks:
            x += block(x)
        
        return x


class MRFLayer(torch.nn.Module):
    def __init__(self, config: GeneratorConfig):
        self.config = config 


class Generator(torch.nn.Module):
    def __init__(self, config: GeneratorConfig):
        self.config = config
