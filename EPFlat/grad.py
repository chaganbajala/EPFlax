"""
EPFlat/grad.py — Equilibrium Propagation gradient estimators for flat networks.

This is the EPFlat counterpart of ``EPFlax/grad.py``.  The core EP algorithm
is identical, but the network state is a single flat 1-D array rather than a
dict of per-layer arrays.  As a result:

- ``params_derivative`` is called with the flat state vector directly.
- The batch loop indexes with ``free_equi[k, :]`` rather than a dict slice.

See ``EPFlax/grad.py`` for the full theoretical background.
"""

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import time
import pickle
import gc
from functools import partial


class EP_grad:
    """EP gradient estimator for flat-state networks.

    Parameters
    ----------
    grad_params : tuple (beta, runtime, rtol, atol)
        beta    : nudging strength β.
        runtime : ODE integration time.
        rtol    : relative ODE tolerance.
        atol    : absolute ODE tolerance.
    sample_args : tuple (sample_method, batch_size, M_init)
        Sampling strategy; see ``EPFlax.grad.EP_grad`` for details.
    """

    def __init__(self, grad_params, sample_args):
        self.beta, self.runtime, self.rtol, self.atol = grad_params
        self.sample_method, self.batch_size, self.M_init = sample_args

        if self.sample_method == 'full':
            self.grad_func = self.full_gradient
        elif self.sample_method == 'mini_batch':
            self.grad_func = self.mini_batch_gradient
        elif self.sample_method == 'random_init_mini_batch':
            self.grad_func = self.radnom_init_mini_batch_gradient

    def devided_by_beta(self, x):
        """Divide a leaf of the gradient pytree by β."""
        return x / self.beta

    def full_gradient(self, input_data, target, nn, network_params, *args):
        """EP gradient using the full dataset.

        Args:
            input_data    : all inputs, shape (N_data, ...).
            target        : all targets, shape (N_data, ...).
            nn            : flat Network instance.
            network_params: current parameter pytree.

        Returns:
            (cost, gradient)
        """
        y0 = nn.get_initial_state(input_data)
        cost, params_g = self.get_params_gradient(y0, target, nn, network_params)
        del y0
        return cost, params_g

    def mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, *args):
        """EP gradient on a random mini-batch.

        Args:
            input_data    : full inputs.
            target        : full targets.
            nn            : flat Network instance.
            network_params: current parameters.
            batch_size    : mini-batch size.

        Returns:
            (cost, gradient)
        """
        y0, running_target = nn.get_initial_state_mini_batch(input_data, target, batch_size)
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g

    def radnom_init_mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, M_init):
        """EP gradient with multiple random initialisations per sample.

        Args:
            input_data    : full inputs.
            target        : full targets.
            nn            : flat Network instance.
            network_params: current parameters.
            batch_size    : mini-batch size.
            M_init        : number of random initialisations per sample.

        Returns:
            (cost, gradient)
        """
        y0, running_target = nn.get_multiple_init_initial_state(
            input_data, target, batch_size, M_init
        )
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g

    def get_params_gradient(self, y0, target, nn, network_params):
        """Core EP gradient computation for flat-state networks.

        Thermalizes to free (β=0) and nudged (β>0) equilibria, then
        estimates the gradient as (1/β)(∂E/∂θ|nudge − ∂E/∂θ|free).

        In the flat representation, each equilibrium is a 2-D array
        (N_data, state_size); individual samples are indexed with ``[k, :]``.

        Args:
            y0            : initial state, shape (N_data, state_size).
            target        : targets, shape (N_data, ...).
            nn            : flat Network instance.
            network_params: current parameter pytree.

        Returns:
            (cost, gradient)
        """
        N_data = target.shape[0]
        N_params = sum(jax.tree_util.tree_leaves(
            jax.tree.map(jnp.size, network_params)
        ))

        # --- Free phase (β = 0) ---
        t0 = time.time()
        free_equi = nn.thermalize_network(y0, target, 0., network_params)
        t1 = time.time()

        # --- Nudged phase (β > 0) ---
        nudge_equi = nn.thermalize_network(free_equi, target, self.beta, network_params)

        # --- Accumulate ∂E/∂θ over batch ---
        # Flat state: free_equi[k, :] is the state of data point k
        dEdParams_free  = nn.params_derivative(free_equi[0, :],  network_params)
        dEdParams_nudge = nn.params_derivative(nudge_equi[0, :], network_params)

        for k in range(1, N_data):
            dEdParams_free  = jax.tree_util.tree_map(jnp.add, dEdParams_free,
                                                      nn.params_derivative(free_equi[k, :],  network_params))
            dEdParams_nudge = jax.tree_util.tree_map(jnp.add, dEdParams_nudge,
                                                      nn.params_derivative(nudge_equi[k, :], network_params))

        # Average over the batch
        mean_func = lambda x: jnp.divide(x, N_data)
        mean_dEdParams_free  = jax.tree_util.tree_map(mean_func, dEdParams_free)
        mean_dEdParams_nudge = jax.tree_util.tree_map(mean_func, dEdParams_nudge)

        # EP gradient: (nudge − free) / β
        gradient = jax.tree_util.tree_map(jnp.subtract, mean_dEdParams_nudge, mean_dEdParams_free)

        # Cost at the free equilibrium output nodes
        cost = jax.vmap(nn.distance_function, (0, 0, None))(free_equi, target, network_params)
        del free_equi, nudge_equi

        return jnp.mean(cost), jax.tree_util.tree_map(self.devided_by_beta, gradient)


class Reg_EP_grad(EP_grad):
    """EP gradient estimator with L2 regularization (flat-network version).

    Adds the regularizer gradient ∂R/∂θ to the EP gradient when the network
    exposes a ``reg`` attribute.  See ``EPFlax.grad.Reg_EP_grad`` for details.
    """

    def get_params_gradient(self, y0, target, nn, network_params):
        if not hasattr(nn, 'reg'):
            return super().get_params_gradient(y0, target, nn, network_params)
        else:
            cost, gradient = super().get_params_gradient(y0, target, nn, network_params)
            # Regularizer gradient via JAX autodiff
            g_reg = jax.grad(nn.regularizer, 0)(network_params)
            return cost, jax.tree_util.tree_map(jnp.add, gradient, g_reg)
