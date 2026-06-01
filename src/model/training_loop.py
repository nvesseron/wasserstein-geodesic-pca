import os

import jax
import jax.numpy as jnp
import wandb
from flax.training import checkpoints

from src.model.pca_state import PCAState


class PCATraining:
    """Imperative trainer around the jitted first- and second-component steps."""

    def __init__(
        self,
        rng,
        pca_state: PCAState,
        train_step,
        num_iter: int = 10000,
        batch_size: int = 1024,
        pca_logging=None,
        first_comp=True,
    ):
        self.rng = rng
        self.pca_state = pca_state
        self.train_step = train_step
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.pca_logging = pca_logging
        self.first_comp = first_comp

    def __call__(self, iter_rho, iter_nu_i):
        eigen_min_max_first = self.train_iter(iter_rho, iter_nu_i)
        # Return the State of NN as well as the eigen values defining the time interval on which the component is valid
        return self.pca_state.state_f, self.pca_state.state_psi, self.pca_state.state_t, eigen_min_max_first

    def train_iter(self, iter_rho, iter_nu_i, logging=True):
        """Train one principal geodesic component."""
        batch = {}
        self.rng, rng = jax.random.split(self.rng, 2)
        checkpoint_every = max(1, self.num_iter // 10)

        for step in range(self.num_iter):
            # Generate mini batches of data
            rng_input, rng_target, rng = jax.random.split(rng, 3)
            data_input = iter_rho.generate_samples(rng_input, self.batch_size)
            index, data_target = iter_nu_i.generate_samples(rng_target, self.batch_size)
            batch["input"] = jnp.array(data_input)
            batch["target"] = jnp.array(data_target)
            batch["index"] = jnp.array(index)

            # Take gradient step of parameters
            (
                self.pca_state.state_f,
                self.pca_state.state_psi,
                self.pca_state.state_t,
                log_train_stat,
            ) = self.train_step(
                self.pca_state.state_f,
                self.pca_state.state_psi,
                self.pca_state.state_t,
                batch,
            )
            # Logging
            if logging and self.pca_logging is not None:
                self.pca_logging.log(self.pca_state, log_train_stat=log_train_stat, step=step)

            if step % 10000 == 0:
                print("Step: ", step)

            # Checkpointing
            if step % checkpoint_every == 0:
                suffix = "first_comp_" if self.first_comp else "second_comp_"
                checkpoint_dir = os.path.join(wandb.run.dir, suffix + "state_f_training")
                checkpoints.save_checkpoint(
                    ckpt_dir=checkpoint_dir, target=self.pca_state.state_f, step=step, keep=4)

                checkpoint_dir = os.path.join(wandb.run.dir, suffix + "state_psi_training")
                checkpoints.save_checkpoint(
                    ckpt_dir=checkpoint_dir, target=self.pca_state.state_psi, step=step, keep=4)

                checkpoint_dir = os.path.join(wandb.run.dir, suffix + "state_t_training")
                checkpoints.save_checkpoint(
                    ckpt_dir=checkpoint_dir, target=self.pca_state.state_t, step=step, keep=4)

        return (log_train_stat["lambda_min"], log_train_stat["lambda_max"])
