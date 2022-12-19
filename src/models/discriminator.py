import torch

from torch.nn.utils import weight_norm, spectral_norm

from src.config import DiscriminatorConfig
from src.models.utils import dilation2padding


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period: int, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.period = period

        self.layers = torch.nn.ModuleList()
        self.leaky_relu = torch.nn.LeakyReLU(config.leaky_relu_slope)
        last_dim = 1
        for i in range(config.mpd_n_blocks):
            self.layers.append(
                weight_norm(torch.nn.Conv2d(
                    in_channels=last_dim,
                    out_channels=2 ** (5 + i),
                    kernel_size=(config.mpd_kernel_size, 1),
                    stride=(config.mpd_stride, 1),
                    padding=(dilation2padding(config.mpd_kernel_size, 1), 0)
                ))
            )
            last_dim = 2 ** (5 + i)

        self.layers.extend([
            weight_norm(torch.nn.Conv2d(
                in_channels=last_dim,
                out_channels=last_dim,
                kernel_size=(config.mpd_kernel_size, 1),
                stride=(config.mpd_stride, 1),
                padding=(dilation2padding(config.mpd_kernel_size, 1), 0)
            )),
            weight_norm(torch.nn.Conv2d(
                in_channels=last_dim,
                out_channels=1,
                kernel_size=(config.mpd_post_conv_kernel_size, 1),
                padding=(dilation2padding(config.mpd_post_conv_kernel_size, 1), 1)
            ))
        ])
        
    def _reshape_input(self, x):
        batch_size, timesteps = x.size()
        if timesteps % self.period > 0:
            x = torch.nn.functional.pad(x, (0, self.period - timesteps % self.period), "reflect")
        return x.view(batch_size, 1, timesteps // self.period + int(timesteps % self.period > 0), self.period)

    def forward(self, x):
        x = self._reshape_input(x)

        hiddens = []
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.leaky_relu(x)
            hiddens.append(torch.flatten(torch.clone(x), 1))

        return torch.flatten(self.layers[-1](x), 1), torch.cat(hiddens, 1)


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig, norm_func):
        super().__init__()
        self.config = config

        self.leaky_relu = torch.nn.LeakyReLU(config.leaky_relu_slope)
        self.blocks = torch.nn.ModuleList()

        for in_chans, out_chans, kernel, stride, padding, groups in zip(
            config.msd_channels[:-1], config.msd_channels[1:],
            config.msd_kernel_sizes, config.msd_strides,
            config.msd_paddings, config.msd_groups
        ):
            self.blocks.append(
                norm_func(torch.nn.Conv1d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    groups=groups
                ))
            )
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        hiddens = []
        for layer in self.blocks[:-1]:
            x = layer(x)
            x = self.leaky_relu(x)
            hiddens.append(torch.flatten(torch.clone(x), 1))

        return torch.flatten(self.blocks[-1](x), 1), torch.cat(hiddens, 1)


class TotalDiscriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config

        self.mpd_discriminators = torch.nn.ModuleList()

        for period in config.mpd_periods:
            self.mpd_discriminators.append(PeriodDiscriminator(period, config))

        self.msd_pooling = torch.nn.AvgPool1d(
            kernel_size=config.pool_kernel,
            stride=config.pool_stride,
            padding=config.pool_padding
        )
        self.msd_discriminators = torch.nn.ModuleList([
            ScaleDiscriminator(config, spectral_norm),
            ScaleDiscriminator(config, weight_norm),
            ScaleDiscriminator(config, weight_norm),
        ])

    def forward(self, x):
        predictions = []
        hiddens = []

        for disc in self.mpd_discriminators:
            pred, hidden = disc(x)
            predictions.append(pred)
            hiddens.append(hidden)
        
        for disc in self.msd_discriminators:
            pred, hidden = disc(x)
            predictions.append(pred)
            hiddens.append(hidden)
            x = self.msd_pooling(x)

        return torch.cat(predictions, 1), torch.cat(hiddens, 1)
