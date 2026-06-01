import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


def _clip_time_for_diffeomorphism(t, lambda_max, lambda_min):
    """Project t to the interval where x -> x + t grad f(x) stays invertible."""
    eps = jnp.finfo(float).eps
    return (
        jax.lax.min(t, -1 / lambda_min - eps) * (lambda_max < 0)
        + jax.lax.max(t, -1 / lambda_max + eps) * (lambda_min > 0)
        + jax.lax.min(jax.lax.max(t, -1 / lambda_max + eps), -1 / lambda_min - eps)
        * (lambda_min < 0)
        * (0 < lambda_max)
    )


def _sinkhorn_epsilon(target, epsilon_scaling):
    centered = target - jnp.mean(target, axis=0, keepdims=True)
    return epsilon_scaling * 0.05 * 2 * jnp.mean(jnp.sum(centered**2, axis=-1))


def get_train_step(stop_gradient_eigen_values=False, epsilon_scaling=1.0):
    def loss_fn(params_f, predict_f, params_psi, predict_psi, params_t, batch):
        """Compute the first-component Sinkhorn loss."""
        data_input = batch["input"]

        # Define grad function and hessian function of f
        grad_f_point = jax.grad(predict_f, argnums=1)
        grad_f = jax.vmap(lambda x: grad_f_point({"params": params_f}, x))
        hess_f_point = jax.hessian(predict_f, argnums=1)
        hess_f = jax.vmap(lambda x: hess_f_point({"params": params_f}, x))

        # Apply grad f and psi to the current batch
        pred_psi = predict_psi({"params": params_psi}, data_input)
        pred_v = grad_f(pred_psi)

        # Select the right t
        t = jnp.sum(params_t * batch["index"])

        # Clip t accoring to eigen values of Hessian of f
        eigen_values_hess_f = jnp.linalg.eigvalsh(hess_f(pred_psi))
        if stop_gradient_eigen_values:
            eigen_values_hess_f = jax.lax.stop_gradient(eigen_values_hess_f)
        lambda_max, lambda_min = eigen_values_hess_f.max(), eigen_values_hess_f.min()
        t = _clip_time_for_diffeomorphism(t, lambda_max, lambda_min)

        # Compute the projection distribution on geodesic
        pred = pred_psi + t * pred_v

        # Minimize with Sinkhorn
        epsilon = _sinkhorn_epsilon(batch["target"], epsilon_scaling)
        loss = sinkhorn_divergence(pointcloud.PointCloud, pred, batch["target"], epsilon=epsilon)[0]
        return loss, (lambda_max, lambda_min)
    
    @jax.jit
    def train_step(state_f, state_psi, state_t, batch):
        """Apply one optimizer step for the first component."""
        value_and_grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 2, 4], has_aux=True)
        (loss, (lambda_max, lambda_min)), (grad_f, grad_psi, grad_t) = value_and_grad_fn(
            state_f.params,
            state_f.apply_fn,
            state_psi.params,
            state_psi.apply_fn,
            state_t.params,
            batch,
        )
        log = {"loss": loss, "lambda_min": lambda_min, "lambda_max": lambda_max}
        return (
            state_f.apply_gradients(grads=grad_f),
            state_psi.apply_gradients(grads=grad_psi),
            state_t.apply_gradients(grads=grad_t),
            log,
        )

    return train_step


def get_train_step_second_comp(
    state_f_first_comp,
    state_psi_first_comp,
    eigen_min_max_first,
    stop_gradient_eigen_values=False,
    epsilon_scaling=1.0,
    reg_ortho=1.0,
    reg_intersect=1.0,
):
    def loss_fn_second_comp(params_f, predict_f, params_psi, predict_psi, params_t, batch):
        """Compute the second-component loss with intersection and orthogonality penalties."""
        data_input = batch["input"]

        # Define grad function and hessian function of f
        grad_f_point = jax.grad(predict_f, argnums=1)
        grad_f = jax.vmap(lambda x: grad_f_point({"params": params_f}, x))
        hess_f_point = jax.hessian(predict_f, argnums=1)
        hess_f = jax.vmap(lambda x: hess_f_point({"params": params_f}, x))

        # Apply grad f and psi to the current batch
        pred_psi = predict_psi({"params": params_psi}, data_input)
        pred_v = grad_f(pred_psi)

        # Select the right t
        t = jnp.sum(params_t[:-2] * batch["index"])

        # Clip t accoring to eigen values of Hessian of f
        eigen_values_hess_f = jnp.linalg.eigvalsh(hess_f(pred_psi))
        if stop_gradient_eigen_values:
            eigen_values_hess_f = jax.lax.stop_gradient(eigen_values_hess_f)
        lambda_max, lambda_min = eigen_values_hess_f.max(), eigen_values_hess_f.min()
        t = _clip_time_for_diffeomorphism(t, lambda_max, lambda_min)

        # Compute the projection distribution on geodesic
        pred = pred_psi + t * pred_v

        # Compute Sinkhorn divergence that defines the main part of the loss
        epsilon = _sinkhorn_epsilon(batch["target"], epsilon_scaling)
        loss = sinkhorn_divergence(pointcloud.PointCloud, pred, batch["target"], epsilon=epsilon)[0]

        # Define grad function of f for first component and apply grad f and psi from the first component to the current batch
        pred_psi_first_comp = state_psi_first_comp.apply_fn({"params": state_psi_first_comp.params}, data_input)
        grad_f_first_comp_point = jax.grad(state_f_first_comp.apply_fn, argnums=1)
        grad_f_first_comp = jax.vmap(lambda x: grad_f_first_comp_point({"params": state_f_first_comp.params}, x))
        pred_v_first_comp = grad_f_first_comp(pred_psi_first_comp)

        # Compute the orthogonal term O in intersecting regularization and add it to the loss
        norm_squared_pred_v = jnp.sum(pred_v ** 2, axis=1)
        norm_squared_pred_v_first_comp = jnp.sum(pred_v_first_comp ** 2, axis=1)
        reg_orthogonality = jnp.mean(
            jnp.sum(pred_v * pred_v_first_comp, axis=1) ** 2
            / (norm_squared_pred_v * norm_squared_pred_v_first_comp)
        )
        loss += reg_ortho * reg_orthogonality

        # compute distribution of first component at intersection time
        t_first_comp_inter = params_t[-1]
        lambda_min_first, lambda_max_first = eigen_min_max_first
        t_first_comp_inter = _clip_time_for_diffeomorphism(
            t_first_comp_inter,
            lambda_max_first,
            lambda_min_first,
        )
        first_comp_int = pred_psi_first_comp + t_first_comp_inter * pred_v_first_comp

        # compute distribution of second component at intersection time
        t_second_comp_inter = params_t[-2]
        t_second_comp_inter = _clip_time_for_diffeomorphism(t_second_comp_inter, lambda_max, lambda_min)
        second_comp_int = pred_psi + t_second_comp_inter * pred_v

        # Compute the intersection term I in intersecting regularization and add it to the loss
        distance_intersection_point = jnp.mean(jnp.sum((first_comp_int - second_comp_int) ** 2, axis=1))

        loss += reg_intersect * distance_intersection_point
        return loss, (
            lambda_max,
            lambda_min,
            reg_orthogonality,
            distance_intersection_point,
            jnp.mean(norm_squared_pred_v),
            jnp.mean(norm_squared_pred_v_first_comp),
        )
    
    @jax.jit
    def train_step_second_comp(state_f, state_psi, state_t, batch):
        """Apply one optimizer step for the second component."""
        value_and_grad_fn = jax.value_and_grad(loss_fn_second_comp, argnums=[0, 2, 4], has_aux=True)
        (
            loss,
            (
                lambda_max,
                lambda_min,
                reg_orthogonality,
                distance_intersection_point,
                norm_squared_pred_v,
                norm_squared_pred_v_first_comp,
            ),
        ), (grad_f, grad_psi, grad_t) = value_and_grad_fn(
            state_f.params,
            state_f.apply_fn,
            state_psi.params,
            state_psi.apply_fn,
            state_t.params,
            batch,
        )
        log = {
            "loss second comp": loss,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "reg_orthogonality": reg_orthogonality,
            "distance_intersection_point": distance_intersection_point,
            "norm_squared_pred_v": norm_squared_pred_v,
            "norm_squared_pred_v_first_comp": norm_squared_pred_v_first_comp,
        }
        return (
            state_f.apply_gradients(grads=grad_f),
            state_psi.apply_gradients(grads=grad_psi),
            state_t.apply_gradients(grads=grad_t),
            log,
        )

    return train_step_second_comp
