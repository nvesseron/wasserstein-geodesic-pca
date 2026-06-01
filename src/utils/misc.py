from collections.abc import MutableMapping
from typing import Callable

import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.cm import plasma
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde, multivariate_normal


def flatten(dictionary, parent_key="", separator="-"):
    """Flatten nested Hydra config dictionaries for W&B logging."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def softplus(x: float, beta: float) -> float:
    return nn.activation.softplus(beta * x) / beta


def smooth_leaky_relu(x: float, alpha: float, beta: float) -> float:
    return alpha * x + (1 - alpha) * softplus(x, beta)


def get_act_fn(act_fn_name: str, alpha=0.01, beta=0.1, negative_slope=0.2) -> Callable:
    """Build the activation specified in Hydra configs."""
    if act_fn_name == "smooth_leaky_relu":
        return lambda x: smooth_leaky_relu(x, alpha, beta)
    if act_fn_name == "softplus":
        return lambda x: softplus(x, beta)
    if act_fn_name == "leaky_relu":
        return lambda x: nn.leaky_relu(x, negative_slope)
    return getattr(nn, act_fn_name)


def get_optimizer(config: dict) -> optax.GradientTransformation:
    """Build an Optax optimizer from the compact config schema."""
    scheduler = getattr(optax, config["scheduler"]["name"])(**config["scheduler"]["options"])
    return getattr(optax, config["name"])(
        learning_rate=scheduler,
        b1=config.get("b1", 0.9),
        b2=config.get("b2", 0.999),
    )


def _linspace_between(min_t, max_t, n_steps):
    weights = np.linspace(0.0, 1.0, n_steps + 1)
    return (1 - weights) * min_t + weights * max_t, weights


def _transparent_colormap(color):
    colors = np.array([color for _ in range(256)])
    colors[:, -1] = np.linspace(0, 1, 256)
    return ListedColormap(colors)


def _trajectory_points(pred_psi, pred_v, t):
    return pred_psi + t * pred_v


def plot_gaussian_levelsets(min_t, max_t, pred_psi, pred_v, figsize, xlim, ylim, nb_gauss=10, grid_size=100):
    """Plot Gaussian one-sigma contours fitted to samples along a learned geodesic."""
    list_t, weights = _linspace_between(min_t, max_t, nb_gauss)
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    pos = np.dstack((x_mesh, y_mesh))

    fig = plt.figure(figsize=figsize)
    colors = [plasma(weight) for weight in weights]
    for t, color in zip(list_t, colors):
        points = _trajectory_points(pred_psi, pred_v, t)[:, :2]
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = centered.T @ centered / (len(points) - 1)
        density = multivariate_normal(mean=mean, cov=cov).pdf(pos)
        level = np.exp(-0.5) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        plt.contour(x_mesh, y_mesh, density, levels=[level], colors=[color], linewidths=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian level sets")
    return fig


def plot_densities(min_t, max_t, pred_psi, pred_v, figsize, xlim, ylim, nb_dens=10, grid_size=100):
    """Plot kernel-density estimates for samples along the learned geodesic."""
    list_t, weights = _linspace_between(min_t, max_t, nb_dens)
    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    fig = plt.figure(figsize=figsize)
    colors = [np.array(plasma(weight)) for weight in weights]
    for t, color in zip(list_t, colors):
        points = _trajectory_points(pred_psi, pred_v, t)
        kde = gaussian_kde([points[:, 0], points[:, 1]])
        density = kde(np.vstack([x_mesh.flatten(), y_mesh.flatten()])).reshape(x_mesh.shape)
        transparent = _transparent_colormap(color)
        plt.pcolormesh(x_mesh, y_mesh, density, cmap=transparent, shading="auto")
        plt.contour(x_mesh, y_mesh, density, levels=5, cmap=transparent)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KDE densities")
    return fig


def plot_densities_mnist_colored(min_t, max_t, pred_psi, pred_v, figsize, nb_dens=10):
    """Plot 4D MNIST samples as spatial coordinates plus RGB-like color channels."""
    list_t, _ = _linspace_between(min_t, max_t, nb_dens)
    fig = plt.figure(figsize=figsize)
    for i, t in enumerate(list_t):
        points = _trajectory_points(pred_psi, pred_v, t)
        colors = np.clip(points[:, 2:], 0.0, 1.0)
        rgb = np.hstack((colors[:, :1], np.zeros((colors.shape[0], 1)), colors[:, 1:]))
        plt.scatter(points[:, 0] + 2 * i, points[:, 1], c=rgb)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MNIST colored geodesic")
    return fig


def plot_densities_separated(min_t, max_t, pred_psi, pred_v, figsize, xlim, ylim, nb_dens=10, grid_size=100):
    """Plot KDEs in separated panels so different times do not overlap visually."""
    list_t, weights = _linspace_between(min_t, max_t, nb_dens)
    base_x = np.linspace(xlim[0], xlim[1], grid_size)
    base_y = np.linspace(ylim[0], ylim[1], grid_size)
    base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)

    fig = plt.figure(figsize=figsize)
    colors = [np.array(plasma(weight)) for weight in weights]
    for i, (t, color) in enumerate(zip(list_t, colors)):
        points = _trajectory_points(pred_psi, pred_v, t)
        kde = gaussian_kde([points[:, 0], points[:, 1]])
        density = kde(np.vstack([base_x_mesh.flatten(), base_y_mesh.flatten()])).reshape(base_x_mesh.shape)
        shifted_x, shifted_y = np.meshgrid(base_x + 2 * i, base_y)
        transparent = _transparent_colormap(color)
        plt.pcolormesh(shifted_x, shifted_y, density, cmap=transparent, shading="auto")
        plt.contour(shifted_x, shifted_y, density, levels=5, cmap=transparent)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Separated KDE densities")
    return fig


def plot_proba_separated(min_t, max_t, pred_psi, pred_v, figsize, nb_dens=10):
    """Scatter raw generated samples along the geodesic in separated panels."""
    list_t, weights = _linspace_between(min_t, max_t, nb_dens)
    fig = plt.figure(figsize=figsize)
    for i, (t, weight) in enumerate(zip(list_t, weights)):
        points = _trajectory_points(pred_psi, pred_v, t)
        plt.scatter(points[:, 0] + 2 * i, points[:, 1], c=[plasma(weight)])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Separated samples")
    return fig
