from dataclasses import dataclass
from typing import List


@dataclass
class GeneratorConfig:
    leaky_relu_slope: float = 0.1

    tr_conv_kernel_sizes: List[int] = None
    resblock_dilations: List[List[List[int]]] = None
