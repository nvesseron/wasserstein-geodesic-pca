import os

import jax
import matplotlib.pyplot as plt
import hydra
import wandb
from flax.training import checkpoints
from omegaconf import DictConfig

import src.data.samplers as samplers
import src.model.training_steps as training_steps
import src.utils.misc as utils
from src.model.pca_state import PCAState
from src.model.training_loop import PCATraining
from src.networks.mlp import MLP
from src.utils.logging import PCALogging, get_evaluate_loss_fn, get_log_images_fn


@hydra.main(version_base=None, config_path="src/config", config_name="config_bis")
def experiment(config: DictConfig) -> None:
    config = config["config"]
    rng = jax.random.PRNGKey(seed=config["training_hyper"]["seed"])

    # Initialize samplers
    iter_rho = samplers.get_sampler(
        config["sampler_rho"]["name"],
        dim=config["sampler_rho"]["dim"],
    )
    iter_nu_i = samplers.get_sampler(
        config["sampler_nu_i"]["name"],
        dim=config["sampler_rho"]["dim"],
        index=config["sampler_nu_i"].get("index"),
    )

    # Initialize psi to parameterize the first geodesic
    config_psi = config["architecture"]["neural_psi"]
    activation_function = utils.get_act_fn(config_psi["act_fn"])
    optimizer_psi = utils.get_optimizer(config_psi["optimizer"])
    neural_psi = MLP(dim=config_psi["layers"] + [config["sampler_rho"]["dim"]], act_fn=activation_function)

    # Initialize the t_i parameters
    nb_distrib_i = iter_nu_i.nb_distributions
    optimizer_t = utils.get_optimizer(config["t_i"]["optimizer"])

    # Initialize f to parameterize the first geodesic
    config_f = config["architecture"]["neural_f"]
    activation_function = utils.get_act_fn(config_f["act_fn"])
    neural_f = MLP(dim=config_f["layers"], act_fn=activation_function)
    optimizer_f = utils.get_optimizer(config_f["optimizer"])

    # Iitialize the State that keep all parameters of the first geodesic
    rng, rng_state = jax.random.split(rng, 2)
    pca_state = PCAState(
        rng=rng_state,
        dim_data=config["sampler_rho"]["dim"],
        neural_f=neural_f,
        neural_psi=neural_psi,
        nb_distrib_i=nb_distrib_i,
        optimizer_f=optimizer_f,
        optimizer_t=optimizer_t,
        optimizer_psi=optimizer_psi,
    )

    # Initialize Training step
    train_step = training_steps.get_train_step(
        stop_gradient_eigen_values=config["training_hyper"]["stop_gradient_eigen_values"],
        epsilon_scaling=config["training_hyper"]["epsilon_scaling"],
    )

    # Initialize everything for Logging
    epsilon_log = config["logging"]["epsilon_log"]
    batch_size_log = config["logging"]["batch_size_log"]
    colored_mnist = config["sampler_nu_i"]["name"] in {"mnist_geodesics", "colored_mnist_digits"}
    log_images_fn = get_log_images_fn(
        iter_rho,
        iter_nu_i,
        batch_size_log=batch_size_log,
        colored_mnist=colored_mnist,
    )
    log_loss_fn = get_evaluate_loss_fn(iter_rho, iter_nu_i, 4096 * 2, nb_distrib=100, epsilon=epsilon_log)

    rng, rng_log = jax.random.split(rng, 2)
    pca_logging = PCALogging(
        rng_log,
        log_freq=config["logging"]["log_freq"],
        log_loss_freq=config["logging"]["log_loss_freq"],
        log_freq_images=config["logging"]["log_freq_images"],
        log_images_fn=log_images_fn,
        log_loss_fn=log_loss_fn,
    )

    # Use wandb
    if config["logging"]["offline"]:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project=config["logging"]["project"],
        name=config["sampler_nu_i"]["name"],
        dir=config["saving"]["output_dir"],
        config=utils.flatten(dict(config)),
    )

    # Initialize Training loop
    pca_training = PCATraining(
        rng,
        pca_state=pca_state,
        num_iter=config["training_hyper"]["num_iter"],
        batch_size=config["training_hyper"]["batch_size"],
        train_step=train_step,
        pca_logging=pca_logging,
    )

    # Estimate the first component
    state_f_first_comp, state_psi_first_comp, state_t_first_comp, eigen_min_max_first = pca_training(
        iter_rho=iter_rho,
        iter_nu_i=iter_nu_i,
    )

    # Checkpoint parameters of the first component
    checkpoint_dir = os.path.join(wandb.run.dir, "state_f_first_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_f_first_comp, step=config["training_hyper"]["num_iter"])

    checkpoint_dir = os.path.join(wandb.run.dir, "state_psi_first_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_psi_first_comp, step=config["training_hyper"]["num_iter"])

    checkpoint_dir = os.path.join(wandb.run.dir, "state_t_first_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_t_first_comp, step=config["training_hyper"]["num_iter"])

    # Reinitialize Optimizers for the second component
    print("Compute second component")
    optimizer_psi = utils.get_optimizer(config_psi["optimizer"])
    optimizer_t = utils.get_optimizer(config["t_i"]["optimizer"])
    optimizer_f = utils.get_optimizer(config_f["optimizer"])

    # Recreate State for the second component
    rng, rng_state = jax.random.split(rng, 2)
    pca_state = PCAState(
        rng=rng_state,
        dim_data=config["sampler_rho"]["dim"],
        neural_f=neural_f,
        neural_psi=neural_psi,
        nb_distrib_i=nb_distrib_i,
        optimizer_f=optimizer_f,
        optimizer_t=optimizer_t,
        optimizer_psi=optimizer_psi,
        second_comp=True,
    )

    # Initialize training step for second component
    train_step = training_steps.get_train_step_second_comp(
        state_f_first_comp,
        state_psi_first_comp,
        eigen_min_max_first,
        stop_gradient_eigen_values=config["training_hyper"]["stop_gradient_eigen_values"],
        epsilon_scaling=config["training_hyper"]["epsilon_scaling"],
        reg_ortho=config["training_hyper"]["reg_ortho"],
        reg_intersect=config["training_hyper"]["reg_intersect"],
    )

    # Initialize training loop for second component
    pca_training = PCATraining(
        rng,
        pca_state=pca_state,
        num_iter=config["training_hyper"]["num_iter_second_comp"],
        batch_size=config["training_hyper"]["batch_size"],
        train_step=train_step,
        pca_logging=pca_logging,
        first_comp=False,
    )

    # Estinate second component
    state_f_sec_comp, state_psi_sec_comp, state_t_sec_comp, eigen_min_max_sec = pca_training(
        iter_rho=iter_rho,
        iter_nu_i=iter_nu_i,
    )
    print("Training finished")

    # Checkpoint parameters of second component
    checkpoint_dir = os.path.join(wandb.run.dir, "state_f_sec_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_f_sec_comp, step=config["training_hyper"]["num_iter_second_comp"])

    checkpoint_dir = os.path.join(wandb.run.dir, "state_psi_sec_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_psi_sec_comp, step=config["training_hyper"]["num_iter_second_comp"])

    checkpoint_dir = os.path.join(wandb.run.dir, "state_t_sec_comp")
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=state_t_sec_comp, step=config["training_hyper"]["num_iter_second_comp"])

    wandb.run.finish()
    plt.close("all")


if __name__ == "__main__":
    experiment()
