import torch
import torch.nn.functional as F

from typing import List

from src.config import GeneratorConfig
from src.models.utils import dilation2padding


class ResBlock(torch.nn.Module):
    def __init__(self, n_channels: int, dilations: List[List[int]], kernel_size: int, lrelu_slope: float):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        self.lrelu_slope = lrelu_slope
        for continual_dilations in dilations:
            layers = torch.nn.ModuleList()
            for dilation in continual_dilations:
                layers.append(
                    torch.nn.Conv1d(
                        in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation2padding(kernel_size, dilation)
                    )
                )
            self.blocks.append(layers)

    def forward(self, x):
        for block in self.blocks:
            xb = x.clone()
            for conv in block:
                xb = F.leaky_relu(xb, negative_slope=self.lrelu_slope, inplace=False)
                xb = conv(xb)

            x += xb
        
        return x


class MRFLayer(torch.nn.Module):
    def __init__(self, config: GeneratorConfig, n_channels: int):
        super().__init__()
        self.config = config 
        self.resblocks = torch.nn.ModuleList()
        for kernel_size, dilations in zip(config.mrf_kernel_sizes, config.mrf_dilations):
            self.resblocks.append(
                ResBlock(
                    n_channels=n_channels,
                    dilations=dilations,
                    kernel_size=kernel_size,
                    lrelu_slope=config.leaky_relu_slope
                )
            )

    def forward(self, x):
        result = 0
        for block in self.resblocks:
            result += block(x)
        
        return result / len(self.resblocks)


class Generator(torch.nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.blocks = torch.nn.Sequential()

        self.pre_conv = torch.nn.Conv1d(
            in_channels=config.mel_dimension,
            out_channels=config.upsampling_hidden_dim,
            kernel_size=config.pre_post_kernel_size,
            padding=config.pre_post_kernel_size // 2
        )
        
        hidden_dim = config.upsampling_hidden_dim
        for kernel_size, stride in zip(config.upsampling_kernels, config.upsampling_strides):
            self.blocks.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // 2,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - stride) // 2
                    ),
                    MRFLayer(config, hidden_dim // 2)
                )
            )
            hidden_dim //= 2

        self.post_conv = torch.nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=config.pre_post_kernel_size,
            padding=config.pre_post_kernel_size // 2
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.blocks(x)
        x = self.post_conv(x).squeeze(1)
        return torch.tanh(x)