import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from src.networks.mlp import MLP


class PCAState:
    """Container for the train states optimized by the PCA loop."""

    def __init__(
        self,
        rng,
        dim_data: int,
        neural_psi: MLP,
        optimizer_psi: optax.OptState,
        neural_f: MLP,
        optimizer_f: optax.OptState,
        nb_distrib_i: int,
        optimizer_t: optax.OptState,
        second_comp: bool = False,
    ):
        self.dim_data = dim_data
        rn_state_f, rn_state_psi, rn_state_t = jax.random.split(rng, 3)

        params_psi = neural_psi.init(rn_state_psi, jnp.zeros((10, dim_data)))["params"]
        self.state_psi = train_state.TrainState.create(apply_fn=neural_psi.apply, params=params_psi, tx=optimizer_psi)

        params_f = neural_f.init(rn_state_f, jnp.zeros((10, dim_data)))["params"]
        self.state_f = train_state.TrainState.create(apply_fn=neural_f.apply, params=params_f, tx=optimizer_f)

        if second_comp:
            # The second component learns two extra times for the intersection parameterization.
            nb_distrib_i += 2

        params_t = jax.random.uniform(rn_state_t, (nb_distrib_i,)) * 2 - 1.0
        self.state_t = train_state.TrainState.create(apply_fn=None, params=params_t, tx=optimizer_t)

