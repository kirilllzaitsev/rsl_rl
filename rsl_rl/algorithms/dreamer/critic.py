import torch
import torch.nn as nn
from dreamer.utils.utils import build_network, create_normal_dist, horizontal_forward


class Critic(nn.Module):
    def __init__(
        self, stochastic_size, deterministic_size, hidden_size, num_layers, activation
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist
