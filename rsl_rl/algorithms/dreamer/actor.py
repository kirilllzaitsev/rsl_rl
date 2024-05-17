import torch
import torch.nn as nn
from torch.distributions import TanhTransform

from dreamer.utils.utils import build_network, create_normal_dist


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

        action_size = action_size if discrete_action_bool else 1 * action_size
        init_noise_std = 1.0
        # self.std = nn.Parameter(init_noise_std * torch.ones(action_size))

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
            dist = create_normal_dist(
                x,
                mean_scale=self.mean_scale,
                init_std=self.init_std,
                min_std=self.min_std,
                activation=torch.tanh,
            )
            # # why is this done?
            # dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
            # action = torch.distributions.Independent(dist, 1).rsample()
            # dist = torch.distributions.Normal(x, x * 0.0 + 1.0)
            # action = dist.sample()  # cannot be. need grads
            action = torch.distributions.Independent(dist, 1).rsample()
        return action


    # def update_distribution(self, observations):
    #     mean = self.actor(observations)
    #     self.distribution = Normal(mean, mean * 0.0 + self.std)

    # def act(self, observations, **kwargs):
    #     self.update_distribution(observations)
    #     return self.distribution.sample()