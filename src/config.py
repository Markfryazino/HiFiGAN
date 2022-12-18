from dataclasses import dataclass
from typing import List


@dataclass
class GeneratorConfig:
    leaky_relu_slope: float = 0.1

    mrf_dilations: List[List[List[int]]] = [[[1, 1], [1, 3], [1, 5]]] * 3
    mrf_kernel_sizes: List[int] = [3, 7, 11]

    upsampling_strides: List[int] = [8, 8, 2, 2]
    upsampling_kernels: List[int] = [16, 16, 4, 4]
    upsampling_hidden_dim: int = 512

    pre_post_kernel_size: int = 7

    mel_dimension: int = 80


@dataclass
class DiscriminatorConfig:
    leaky_relu_slope: float = 0.1
    mpd_kernel_size: int = 5
    mpd_stride: int = 3
    mpd_n_blocks: int = 4
    mpd_post_conv_kernel_size: int = 3
    mpd_periods: List[int] = [2, 3, 5, 7, 11]

    msd_channels: List[int] = [1, 128, 128, 256, 512, 1024, 1024, 1024, 1]
    msd_kernel_sizes: List[int] = [15, 41, 41, 41, 41, 41, 5, 3]
    msd_strides: List[int] = [1, 2, 2, 4, 4, 1, 1, 1]
    msd_paddings: List[int] = [7, 20, 20, 20, 20, 20, 2, 1]
    msd_groups: List[int] = [1, 4, 16, 16, 16, 16, 2, 1]

    pool_kernel: int = 4
    pool_stride: int = 2
    pool_padding: int = 2