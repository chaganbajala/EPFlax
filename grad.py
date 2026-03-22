"""
grad.py — Equilibrium Propagation gradient estimators for EPFlax networks.

Equilibrium Propagation (Scellier & Bengio, 2017) estimates the loss gradient
without backpropagation through time.  The key idea is:

    dL/dθ ≈ (1/β) [ ∂E/∂θ |_{nudged}  −  ∂E/∂θ |_{free} ]

where
  - θ        : trainable parameters (weights, bias fields)
  - E(y, θ)  : network internal energy
  - β        : nudging strength (small positive scalar)
  - free      : equilibrium reached with β = 0 (no target influence)
  - nudged    : equilibrium reached with β > 0 (weak nudge toward target)

Both equilibria are found by thermalizing the network ODE until it converges.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import pickle
import gc


class EP_grad:
    """Equilibrium Propagation gradient estimator.

    Computes the EP gradient of the loss with respect to all network
    parameters by comparing the free and nudged equilibrium states across
    a mini-batch or the full dataset.

    Parameters
    ----------
    grad_params : tuple (beta, runtime, rtol, atol)
        beta    : nudging strength β > 0.
        runtime : maximum integration time for the ODE thermalizer.
        rtol    : relative tolerance for the adaptive ODE step-size controller.
        atol    : absolute tolerance for the adaptive ODE step-size controller.
    sample_args : tuple (sample_method, batch_size, M_init)
        sample_method : one of ``'full'``, ``'mini_batch'``,
                        ``'random_init_mini_batch'``.
        batch_size    : number of samples per mini-batch (ignored for 'full').
        M_init        : number of random initialisations per sample when using
                        ``'random_init_mini_batch'``.
    """

    def __init__(self, grad_params, sample_args):
        self.beta, self.runtime, self.rtol, self.atol = grad_params
        self.sample_method, self.batch_size, self.M_init = sample_args

        # Bind the correct gradient method based on the requested sampling strategy.
        if self.sample_method == 'full':
            self.grad_func = self.full_gradient
        elif self.sample_method == 'mini_batch':
            self.grad_func = self.mini_batch_gradient
        elif self.sample_method == 'random_init_mini_batch':
            self.grad_func = self.radnom_init_mini_batch_gradient

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def full_gradient(self, input_data, target, nn, network_params, *args):
        """Compute EP gradient using the full dataset.

        Args:
            input_data    : array of shape (N_data, ...) — all inputs.
            target        : array of shape (N_data, ...) — all targets.
            nn            : network Module instance.
            network_params: current parameter pytree.

        Returns:
            (cost, gradient) where cost is the scalar mean loss and gradient
            is a pytree matching network_params.
        """
        y0 = nn.get_initial_state(input_data)
        cost, params_g = self.get_params_gradient(y0, target, nn, network_params)
        del y0
        return cost, params_g

    def mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, *args):
        """Compute EP gradient on a randomly sampled mini-batch.

        Args:
            input_data    : full dataset inputs, shape (N_data, ...).
            target        : full dataset targets, shape (N_data, ...).
            nn            : network Module instance.
            network_params: current parameter pytree.
            batch_size    : number of samples to draw.

        Returns:
            (cost, gradient) estimated on the mini-batch.
        """
        y0, running_target = nn.get_initial_state_mini_batch(input_data, target, batch_size)
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g

    def radnom_init_mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, M_init):
        """Compute EP gradient using multiple random initialisations per sample.

        For each sample in the mini-batch, M_init independent initial states
        are thermalised.  This can reduce the variance caused by getting trapped
        in local energy minima.

        Args:
            input_data    : full dataset inputs, shape (N_data, ...).
            target        : full dataset targets, shape (N_data, ...).
            nn            : network Module instance.
            network_params: current parameter pytree.
            batch_size    : number of samples to draw.
            M_init        : number of random initialisations per sample.

        Returns:
            (cost, gradient) averaged over all initialisations and samples.
        """
        y0, running_target = nn.get_multiple_init_initial_state(
            input_data, target, batch_size, M_init
        )
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g

    # ------------------------------------------------------------------
    # Core EP gradient computation
    # ------------------------------------------------------------------

    def get_params_gradient(self, y0, target, nn, network_params):
        """Estimate the EP gradient from free and nudged equilibria.

        Algorithm
        ---------
        1. Thermalize with β = 0  →  free equilibrium   y_free.
        2. Thermalize with β > 0  →  nudged equilibrium  y_nudge.
        3. Compute ∂E/∂θ at each equilibrium for every data point,
           average over the batch.
        4. Return gradient = (1/β)(mean_nudge − mean_free).

        Args:
            y0            : initial network state dict, values shape (N_data, ...).
            target        : target array, shape (N_data, ...).
            nn            : network Module instance.
            network_params: current parameter pytree θ.

        Returns:
            cost     : scalar mean loss at the free equilibrium.
            gradient : pytree of same structure as network_params.
        """
        N_data = y0['input_data'].shape[0]

        # --- Free phase: relax the network without target influence (β = 0) ---
        free_equi = nn.thermalize_network(y0, target, 0., network_params)

        # --- Nudged phase: relax with a weak nudge toward target (β > 0) ----
        nudge_equi = nn.thermalize_network(free_equi, target, self.beta, network_params)

        # --- Accumulate ∂E/∂θ over data points at each equilibrium ----------
        # Start with the first data point, then add contributions in a loop.
        # (vmap over N_data would require storing all equilibria simultaneously;
        #  the loop is memory-friendlier for large batches.)
        dEdParams_free  = nn.params_derivative({key: value[0, :] for key, value in free_equi.items()},  network_params)
        dEdParams_nudge = nn.params_derivative({key: value[0, :] for key, value in nudge_equi.items()}, network_params)

        for k in range(1, N_data):
            free_values  = {key: value[k, :] for key, value in free_equi.items()}
            nudge_values = {key: value[k, :] for key, value in nudge_equi.items()}
            dEdParams_free  = jax.tree_map(jnp.add, dEdParams_free,  nn.params_derivative(free_values,  network_params))
            dEdParams_nudge = jax.tree_map(jnp.add, dEdParams_nudge, nn.params_derivative(nudge_values, network_params))

        # Average over the batch
        mean_func = lambda x: jnp.divide(x, N_data)
        mean_dEdParams_free  = jax.tree_map(mean_func, dEdParams_free)
        mean_dEdParams_nudge = jax.tree_map(mean_func, dEdParams_nudge)

        # EP gradient: (1/β)(∂E/∂θ|nudge − ∂E/∂θ|free)
        gradient = jax.tree_map(jnp.subtract, mean_dEdParams_nudge, mean_dEdParams_free)

        # Loss is evaluated at the free equilibrium output
        cost = jax.vmap(nn.distance_function, (0, 0, None))(
            free_equi[nn.output_name], target, network_params
        )
        del free_equi, nudge_equi

        return jnp.mean(cost), jax.tree_map(lambda x: jnp.divide(x, self.beta), gradient)


class Reg_EP_grad(EP_grad):
    """EP gradient estimator with L2 regularization.

    Extends ``EP_grad`` by adding the gradient of a regularizer defined on
    the network module (``nn.regularizer(params)``) to the EP gradient.
    If the network does not have a ``reg`` attribute, falls back to standard
    EP without regularization.
    """

    def get_params_gradient(self, y0, target, nn, network_params):
        """Compute regularized EP gradient.

        If ``nn`` exposes a ``regularizer`` method (i.e. it is an instance of
        ``reg_model.Module``), the gradient of the regularizer is added to the
        EP gradient:

            gradient_total = gradient_EP + ∂R/∂θ

        where R(θ) is the regularization term (e.g. L2 weight penalty).

        Args:
            y0, target, nn, network_params: same as ``EP_grad.get_params_gradient``.

        Returns:
            (cost, gradient_total)
        """
        if not hasattr(nn, 'reg'):
            # No regularizer defined — use plain EP gradient
            return super().get_params_gradient(y0, target, nn, network_params)
        else:
            cost, gradient = super().get_params_gradient(y0, target, nn, network_params)
            # Differentiate the regularizer R(θ) w.r.t. θ
            g_reg = jax.grad(nn.regularizer, 0)(network_params)
            return cost, jax.tree_map(jnp.add, gradient, g_reg)
