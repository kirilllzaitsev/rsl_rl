import torch
import torch.nn as nn
from dreamer.utils.utils import build_network, create_normal_dist
from torch.distributions import TanhTransform


class Actor(nn.Module):
    def __init__(
        self,
        discrete_action_bool,
        action_size,
        stochastic_size,
        deterministic_size,
        hidden_size,
        num_layers,
        activation,
        mean_scale,
        init_std,
        min_std,
    ):
        super().__init__()
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.mean_scale = mean_scale
        self.init_std = init_std
        self.min_std = min_std

        action_size = action_size if discrete_action_bool else 2 * action_size
        self.std = None

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.hidden_size,
            self.num_layers,
            self.activation,
            action_size,
        )

    def forward(self, posterior, deterministic):
        x = torch.cat((posterior, deterministic), -1)
        x = self.network(x)
        if self.discrete_action_bool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist, std = create_normal_dist(
                x,
                mean_scale=self.mean_scale,
                init_std=self.init_std,
                min_std=self.min_std,
                activation=torch.tanh,
                return_std=True,
            )
            self.std = std
            # # why is this done?
            dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
            action = torch.distributions.Independent(dist, 1).rsample()
        return action
