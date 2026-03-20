from typing import Tuple, Optional, Union

from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization, Swish


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.register_buffer("normalization_steps", torch.tensor(1))
        self.register_buffer("running_mean", torch.zeros(vec_obs_size))
        self.register_buffer("running_variance", torch.ones(vec_obs_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        with torch.no_grad():
            steps_increment = vector_input.size()[0]
            total_new_steps = self.normalization_steps + steps_increment

            input_to_old_mean = vector_input - self.running_mean
            new_mean: torch.Tensor = self.running_mean + (
                input_to_old_mean / total_new_steps
            ).sum(0)

            input_to_new_mean = vector_input - new_mean
            new_variance = self.running_variance + (
                input_to_new_mean * input_to_old_mean
            ).sum(0)
            self.running_mean: torch.Tensor = new_mean
            self.running_variance: torch.Tensor = new_variance
            self.normalization_steps: torch.Tensor = total_new_steps

    def copy_from(self, other_normalizer: "Normalizer") -> None:
        self.normalization_steps.data.copy_(other_normalizer.normalization_steps.data)
        self.running_mean.data.copy_(other_normalizer.running_mean.data)
        self.running_variance.copy_(other_normalizer.running_variance.data)


def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


class VectorInput(nn.Module):
    def __init__(self, input_size: int, normalize: bool = False):
        super().__init__()
        self.normalizer: Optional[Normalizer] = None
        if normalize:
            self.normalizer = Normalizer(input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            inputs = self.normalizer(inputs)
        return inputs

    def copy_normalization(self, other_input: "VectorInput") -> None:
        if self.normalizer is not None and other_input.normalizer is not None:
            self.normalizer.copy_from(other_input.normalizer)

    def update_normalization(self, inputs: torch.Tensor) -> None:
        if self.normalizer is not None:
            self.normalizer.update(inputs)


class FullyConnectedVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.output_size = output_size
        self.input_size = height * width * initial_channels
        self.dense = nn.Sequential(
            linear_layer(
                self.input_size,
                self.output_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        if visual_obs.ndim == 4 and visual_obs.shape[-1] < visual_obs.shape[1]:
            pass
        else:
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        hidden = visual_obs.reshape(-1, self.input_size)
        return self.dense(hidden)


class SmallVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 3, 1)
        conv_2_hw = conv_output_shape(conv_1_hw, 3, 1)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 144

        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 35, [3, 3], [1, 1]),
            nn.LeakyReLU(),
            nn.Conv2d(35, 144, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape(-1, self.final_flat)
        return self.dense(hidden)


class SimpleVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 16, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, [4, 4], [2, 2]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape(-1, self.final_flat)
        return self.dense(hidden)


class NatureVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 64

        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 32, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)


class ResNetBlock(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.layers(input_tensor)


class ResNetVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        n_channels = [16, 32, 32]
        n_blocks = 2
        layers = []
        last_channel = initial_channels
        for _, channel in enumerate(n_channels):
            layers.append(nn.Conv2d(last_channel, channel, [3, 3], [1, 1], padding=1))
            layers.append(nn.MaxPool2d([3, 3], [2, 2]))
            height, width = pool_out_shape((height, width), 3)
            for _ in range(n_blocks):
                layers.append(ResNetBlock(channel))
            last_channel = channel
        layers.append(Swish())
        self.final_flat_size = n_channels[-1] * height * width
        self.dense = linear_layer(
            self.final_flat_size,
            output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.41,
        )
        self.sequential = nn.Sequential(*layers)

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        hidden = self.sequential(visual_obs)
        before_out = hidden.reshape(-1, self.final_flat_size)
        return torch.relu(self.dense(before_out))