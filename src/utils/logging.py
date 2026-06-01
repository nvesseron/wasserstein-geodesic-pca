import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from src.utils.misc import (
    plot_densities,
    plot_densities_mnist_colored,
    plot_densities_separated,
    plot_gaussian_levelsets,
    plot_proba_separated,
)


def _clip_times_to_diffeomorphic_interval(times, lambda_max, lambda_min):
    """Keep logged geodesic times inside the interval used during training."""
    eps = jnp.finfo(float).eps
    if lambda_max == 0.0 or lambda_min == 0.0:
        return times

    lower = -1 / lambda_max + eps
    upper = -1 / lambda_min - eps
    return [
        min(t, upper) * (lambda_max < 0)
        + max(t, lower) * (lambda_min > 0)
        + min(max(t, lower), upper) * (lambda_min < 0 < lambda_max)
        for t in times
    ]


class PCALogging:
    """Small W&B logging coordinator for the two-component PCA training loop."""

    def __init__(
        self,
        rng,
        log_freq,
        log_freq_images,
        log_loss_freq,
        log_images_fn,
        log_loss_fn,
        use_wandb=True,
    ):
        self.rng = rng
        self.log_freq = log_freq
        self.log_freq_images = log_freq_images
        self.log_loss_freq = log_loss_freq
        self.log_images_fn = log_images_fn
        self.log_loss_fn = log_loss_fn
        self.use_wandb = use_wandb

    def log(self, pca_state, log_train_stat, step):
        if step % self.log_freq == 0 and self.use_wandb:
            wandb.log(log_train_stat)

        lambda_max = log_train_stat["lambda_max"]
        lambda_min = log_train_stat["lambda_min"]

        if step % self.log_loss_freq == 0 and self.log_loss_fn is not None:
            self.log_loss_fn(pca_state, lambda_max, lambda_min)

        if not self._should_log_images(step):
            return None

        times = _clip_times_to_diffeomorphic_interval(
            pca_state.state_t.params,
            lambda_max,
            lambda_min,
        )
        self._log_time_bounds(times, lambda_max, lambda_min)

        self.rng, rng_log = jax.random.split(self.rng, num=2)
        self.log_images_fn(pca_state, times, rng_log)
        return None

    def _should_log_images(self, step):
        return (
            self.use_wandb
            and self.log_images_fn is not None
            and step > 1
            and step % self.log_freq_images == 0
        )

    def _log_time_bounds(self, times, lambda_max, lambda_min):
        if lambda_max == 0.0 or lambda_min == 0.0:
            return

        times = jnp.array(times)
        wandb.log(
            {
                "min_t_lambda_min": min(times * lambda_min),
                "max_t_lambda_min": max(times * lambda_min),
                "min_t_lambda_max": min(times * lambda_max),
                "max_t_lambda_max": max(times * lambda_max),
            }
        )


def get_evaluate_loss_fn(iter_rho, iter_nu_i, batch_size_log=4096, nb_distrib=100, epsilon=None):
    nb_distrib = iter_nu_i.nb_distributions

    @jax.jit
    def compute_loss(samples_rho, t_one_hot, samples_nu_i, state_f, state_psi, state_t, lambda_max, lambda_min):
        params_psi, predict_psi = state_psi.params, state_psi.apply_fn
        pred_psi = predict_psi({"params": params_psi}, samples_rho)

        params_f, predict_f = state_f.params, state_f.apply_fn
        grad_f_point = jax.grad(predict_f, argnums=1)
        grad_f = jax.vmap(lambda x: grad_f_point({"params": params_f}, x))
        pred_v = grad_f(pred_psi)

        # The one-hot vector selects the learned time associated with nu_i.
        params_t = state_t.params
        t = jnp.sum(params_t[:nb_distrib] * t_one_hot)
        eps = jnp.finfo(float).eps
        t = (
            jax.lax.min(t, -1 / lambda_min - eps) * (lambda_max < 0)
            + jax.lax.max(t, -1 / lambda_max + eps) * (lambda_min > 0)
            + jax.lax.min(jax.lax.max(t, -1 / lambda_max + eps), -1 / lambda_min - eps)
            * (lambda_min < 0)
            * (0 < lambda_max)
        )

        pred = pred_psi + t * pred_v
        return sinkhorn_divergence(
            pointcloud.PointCloud,
            pred,
            samples_nu_i,
            epsilon=epsilon,
        )[0]

    def evaluate_loss(pca_state, lambda_max, lambda_min):
        key = jax.random.PRNGKey(0)
        loss_samples = 0
        for _ in range(nb_distrib):
            key, key1 = jax.random.split(key, 2)
            t_one_hot, samples_nu_i = iter_nu_i.generate_samples(key1, batch_size_log)
            samples_rho = iter_rho.generate_samples(key1, batch_size_log)
            loss_samples += compute_loss(
                samples_rho,
                t_one_hot,
                samples_nu_i,
                pca_state.state_f,
                pca_state.state_psi,
                pca_state.state_t,
                lambda_max,
                lambda_min,
            )

        wandb.log({"least square with " + str(nb_distrib) + " distrib": loss_samples})
        return None

    return evaluate_loss


def get_log_images_fn(iter_rho, iter_nu_i, batch_size_log=4096, colored_mnist=False):
    _, data_target = iter_nu_i.generate_samples_all(jax.random.PRNGKey(0), batch_size_log)
    min_x = jnp.min(data_target[:, 0])
    max_x = jnp.max(data_target[:, 0])
    min_y = jnp.min(data_target[:, 1])
    max_y = jnp.max(data_target[:, 1])

    aspect_ratio = (max_y - min_y) / (max_x - min_x)
    figsize = (12, aspect_ratio * 10)

    @jax.jit
    def compute_for_logging(state_f, state_psi, batch):
        params_psi, predict_psi = state_psi.params, state_psi.apply_fn
        pred_psi = predict_psi({"params": params_psi}, batch["input"])

        params_f, predict_f = state_f.params, state_f.apply_fn
        grad_f_point = jax.grad(predict_f, argnums=1)
        grad_f = jax.vmap(lambda x: grad_f_point({"params": params_f}, x))
        pred_v = grad_f(pred_psi)
        return pred_psi, pred_v

    def log_images_fn(pca_state, times, rng_log):
        """Log target samples and samples drawn along the learned geodesic."""
        rng_log, rng_input, rng_target = jax.random.split(rng_log, 3)
        data_input = iter_rho.generate_samples(rng_input, batch_size_log)
        index, data_target = iter_nu_i.generate_samples(rng_target, batch_size_log)
        batch = {
            "input": jnp.array(data_input),
            "target": jnp.array(data_target),
            "index": jnp.array(index),
        }

        pred_psi, pred_v = compute_for_logging(pca_state.state_f, pca_state.state_psi, batch)

        min_t = min(times)
        max_t = max(times)
        sampled_t = jax.random.uniform(rng_log, (len(batch["input"]), 1), minval=min_t, maxval=max_t)
        direct_points = pred_psi + sampled_t * pred_v

        fig = plt.figure(figsize=figsize)
        plt.scatter(direct_points[:, 0], direct_points[:, 1], c=sampled_t.reshape((-1,)), s=8, cmap="plasma")
        cbar = plt.colorbar()
        cbar.set_label("t")
        plt.title("First Direction")
        plt.legend()
        wandb.log({"First Direction": [wandb.Image(fig)]})
        plt.close()

        rng_log, rng_target = jax.random.split(rng_log, 2)
        index, data_target = iter_nu_i.generate_samples_all(rng_target, batch_size_log)

        fig = plt.figure(figsize=figsize)
        plt.scatter(data_target[:, 0], data_target[:, 1], c=index.reshape((-1,)), s=8, cmap="jet")
        cbar = plt.colorbar()
        cbar.set_label("i")
        plt.title("Target samples")
        plt.legend()
        wandb.log({"Target samples": [wandb.Image(fig)]})
        plt.close()

        fig = plot_gaussian_levelsets(
            min_t,
            max_t,
            pred_psi,
            pred_v,
            figsize,
            (min_x, max_x),
            (min_y, max_y),
            nb_gauss=10,
            grid_size=100,
        )
        wandb.log({"Gaussian levelsets": [wandb.Image(fig)]})
        plt.close()

        fig = plot_densities(
            min_t,
            max_t,
            pred_psi,
            pred_v,
            figsize,
            (min_x, max_x),
            (min_y, max_y),
            nb_dens=3,
            grid_size=100,
        )
        wandb.log({"Densities 1st component": [wandb.Image(fig)]})
        plt.close()

        fig = plot_densities_separated(
            min_t,
            max_t,
            pred_psi,
            pred_v,
            (figsize[0] * 3.0, figsize[1]),
            (min_x, max_x),
            (min_y, max_y),
            nb_dens=3,
            grid_size=100,
        )
        wandb.log({"Densities 1st component separated": [wandb.Image(fig)]})
        plt.close()

        fig = plot_proba_separated(
            min_t,
            max_t,
            pred_psi,
            pred_v,
            (figsize[0] * 3.0, figsize[1]),
            nb_dens=3,
        )
        wandb.log({"Proba 1st component separated": [wandb.Image(fig)]})
        plt.close()

        if colored_mnist:
            fig = plot_densities_mnist_colored(
                min_t,
                max_t,
                pred_psi,
                pred_v,
                (figsize[0] * 3.0, figsize[1]),
                nb_dens=3,
            )
            wandb.log({"Densities real colors": [wandb.Image(fig)]})
            plt.close()

        plt.close("all")

    return log_images_fn
