from typing import Callable, Sequence
from flax import linen as nn


class MLP(nn.Module):
    """Plain MLP used for both psi and the scalar potential f."""

    dim: Sequence[int]
    act_fn: Callable = nn.relu
    init_fn: Callable = nn.initializers.variance_scaling(scale=0.1, distribution="normal", mode="fan_avg")
    act_final: Callable = None

    def setup(self):
        self.layers = [nn.Dense(feature, use_bias=True, kernel_init=self.init_fn) for feature in self.dim]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        if self.act_final is not None:
            x = self.act_final(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        return x.squeeze()
