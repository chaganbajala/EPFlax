"""
EPFlat/train.py — Optimizer classes for flat EPFlat networks.

This module mirrors ``EPFlax/train.py`` but targets the flat-state network
interface defined in ``EPFlat/model.py``.  See that module for full docstrings;
only EPFlat-specific differences are noted here.

Key difference from EPFlax/train.py
-------------------------------------
The EPFlat network stores the full state as a single flat vector, so
layer-wise optimizers here split the state along known ``split_points``
rather than iterating over a dict of named layers.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import time
import pickle
from functools import partial


class optimizer:
    """Abstract base optimizer for EPFlat networks.

    Parameters
    ----------
    grad_method      : EP_grad instance.
    nn               : flat Network instance.
    network_params_0 : initial parameter pytree.
    """

    def __init__(self, grad_method, nn, network_params_0):
        self.grad_method = grad_method
        self.nn = nn
        self.network_params_0 = network_params_0

    def optimize_step(self, network_params, g_params):
        """Single parameter update step (to be overridden)."""
        pass

    def train(self, input_data, target, show_process=False, dynamical_saving=False, suffix=None):
        """Full training loop (to be overridden)."""
        pass


# ---------------------------------------------------------------------------
# Vanilla gradient descent
# ---------------------------------------------------------------------------

class gradient_descent(optimizer):
    """Vanilla gradient descent for EPFlat networks.

    θ ← θ − η · ∂L/∂θ
    """

    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate):
        """JIT-compiled GD step."""
        def add_func(x, gx, learning_rate):
            return x - gx * learning_rate
        return jax.tree_map(lambda x, gx: add_func(x, gx, learning_rate), network_params, g_params)

    def train(self, N_epoch, learning_rate, input_data, target,
              show_process=False, dynamical_save=False, suffix=None):
        """Run GD training.

        Args:
            N_epoch       : number of epochs.
            learning_rate : step size η.
            input_data    : training inputs.
            target        : training targets.
            show_process  : print progress if True.
            dynamical_save: save checkpoint each epoch if True.
            suffix        : file suffix for checkpoint files.

        Returns:
            costL, paramsL
        """
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            running_params = self.optimize_step(running_params, params_g, learning_rate)
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL


# ---------------------------------------------------------------------------
# Gradient descent with momentum
# ---------------------------------------------------------------------------

class moment_gradient_descent(gradient_descent):
    """Momentum gradient descent for EPFlat networks.

    g_eff ← g + r · g_prev;  θ ← θ − η · g_eff
    """

    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate, last_grad, r):
        """JIT-compiled momentum update step.

        Args:
            network_params: current θ.
            g_params      : current gradient.
            learning_rate : step size η.
            last_grad     : previous effective gradient.
            r             : momentum coefficient.

        Returns:
            (new_grad, new_params)
        """
        def moment_func(gx, lgx, r): return gx + r * lgx
        def add_func(x, tot_gx, lr): return x - tot_gx * lr

        new_grad = jax.tree_map(lambda gx, lgx: moment_func(gx, lgx, r), g_params, last_grad)
        new_params = jax.tree_map(lambda x, tot_gx: add_func(x, tot_gx, learning_rate), network_params, new_grad)
        return new_grad, new_params

    def train(self, N_epoch, learning_rate, input_data, target,
              r=0.9, show_process=False, dynamical_save=False, suffix=None):
        """Run momentum GD training."""
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        last_grad = jax.tree_map(lambda x: 0. * x, running_params)

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            last_grad, running_params = self.optimize_step(
                running_params, params_g, learning_rate, last_grad, r
            )
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL


# ---------------------------------------------------------------------------
# Layer-wise optimizers (flat-network specific)
# ---------------------------------------------------------------------------

class layerwise_gradient_descent(gradient_descent):
    """GD with per-layer learning rates for flat-state networks.

    Requires the network to have a layered architecture (``structure_name ==
    'layered'``) with pre-computed ``split_points`` that delimit each layer
    within the flat state/parameter vector.

    Raises
    ------
    ValueError
        If the network does not expose a layered structure.
    """

    def __init__(self, grad_method, nn, network_params_0):
        super().__init__(grad_method, nn, network_params_0)
        if nn.structure_name != 'layered':
            raise ValueError('The network does not have layer architecture!')

    def optimize_step(self, network_params, g_params, learning_rate_list):
        """Layer-wise GD step.

        The flat parameter vector is split at ``nn.split_points`` into per-layer
        segments; each segment is updated with its own learning rate.

        Args:
            network_params    : (WL, bias) where WL is a list of weight matrices
                                and bias is the concatenated bias vector.
            g_params          : (g_WL, g_bias) — corresponding gradients.
            learning_rate_list: list of per-layer learning rates.

        Returns:
            Updated (new_WL, new_bias).
        """
        def add_func(x, gx, learning_rate):
            return x - learning_rate * gx

        WL, bias = network_params
        # Split the flat bias vector back into per-layer segments
        biasL = np.split(bias, self.nn.split_points, axis=-1)

        g_WL, g_bias = g_params
        g_biasL = np.split(g_bias, self.nn.split_points, axis=-1)

        new_WL = jax.tree_map(add_func, WL, g_WL, learning_rate_list)
        # Skip the input bias (index 0) — it is not a trainable parameter
        new_biasL = jax.tree_map(add_func, biasL[1:], g_biasL[1:], learning_rate_list)
        new_bias = np.concatenate((biasL[0], *new_biasL), axis=-1)

        return new_WL, new_bias


class layerwise_moment_gradient_descent(moment_gradient_descent):
    """Momentum GD with per-layer learning rates for flat-state networks."""

    def get_tot_grad(self, g_params, old_params, r):
        """Compute momentum-augmented gradient: g_eff = g + r * g_prev."""
        def g_func(g, lg, r): return g + r * lg
        return jax.tree_map(lambda g, lg: g_func(g, lg, r), g_params, old_params)

    def layer_wise_update(self, network_params, g_params, learning_rate_list):
        """Apply per-layer update to the flat parameter representation."""
        def add_func(x, gx, learning_rate):
            return x - learning_rate * gx

        WL, bias = network_params
        biasL = np.split(bias, self.nn.split_points, axis=-1)
        g_WL, g_bias = g_params
        g_biasL = np.split(g_bias, self.nn.split_points, axis=-1)

        new_WL = jax.tree_map(add_func, WL, g_WL, learning_rate_list)
        new_biasL = jax.tree_map(add_func, biasL[1:], g_biasL[1:], learning_rate_list)
        new_bias = np.concatenate((biasL[0], *new_biasL), axis=-1)

        return new_WL, new_bias

    def optimize_step(self, running_params, params_g, learning_rate_list, last_grad, r):
        """Single momentum + layer-wise update step."""
        tot_g = self.get_tot_grad(params_g, last_grad, r)
        new_params = self.layer_wise_update(running_params, tot_g, learning_rate_list)
        return tot_g, new_params

    def train(self, N_epoch, learning_rate_list, input_data, target,
              r=0.9, show_process=False, dynamical_save=False, suffix=None):
        """Run layer-wise momentum GD training."""
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        last_grad = jax.tree_map(lambda x: 0. * x, running_params)

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            last_grad, running_params = self.optimize_step(
                running_params, params_g, learning_rate_list, last_grad, r
            )
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL


# ---------------------------------------------------------------------------
# Self-adaptive layer-wise optimizers
# ---------------------------------------------------------------------------

class self_ad_layerwise_gd(layerwise_gradient_descent):
    """Layer-wise GD with self-adjusted per-layer learning rates.

    At each epoch, the per-layer learning rate for layer l is set to:

        η_l = (‖W_l‖ / ‖∂L/∂W_l‖) · (η / η_last)

    where η_last normalises so that the deepest layer gets rate η.  This
    automatically balances the relative update magnitude across layers.
    """

    def train(self, N_epoch, learning_rate, input_data, target,
              r=0.9, show_process=False, dynamical_save=False, suffix=None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []

        def norm_ratio(x, gx):
            # Per-layer ratio ‖W‖ / ‖g‖ — used as the adaptive learning rate scale
            return jnp.linalg.norm(x) / jnp.linalg.norm(gx)

        def list_multiply(L, x):
            # Normalise so the last layer gets exactly rate x, then scale the rest
            a0 = L[-1]
            for k in range(len(L)):
                L[k] = L[k] / a0 * x
            return L

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            norm_ratio_list = jax.tree_map(norm_ratio, running_params[0], params_g[0])
            learning_rate_list = list_multiply(norm_ratio_list, learning_rate)
            running_params = self.optimize_step(running_params, params_g, learning_rate_list)
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL


class self_ad_layerwise_mgd(layerwise_moment_gradient_descent):
    """Layer-wise momentum GD with self-adjusted per-layer learning rates.

    Combines the adaptive rate scaling of ``self_ad_layerwise_gd`` with
    momentum (r > 0).
    """

    def train(self, N_epoch, learning_rate, input_data, target,
              r=0.9, show_process=False, dynamical_save=False, suffix=None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        last_grad = jax.tree_map(lambda x: 0. * x, running_params)

        def norm_ratio(x, gx):
            return jnp.linalg.norm(x) / jnp.linalg.norm(gx)

        def list_multiply(L, x):
            a0 = L[-1]
            for k in range(len(L)):
                L[k] = L[k] / a0 * x
            return L

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            norm_ratio_list = jax.tree_map(norm_ratio, running_params[0], params_g[0])
            learning_rate_list = list_multiply(norm_ratio_list, learning_rate)
            last_grad, running_params = self.optimize_step(
                running_params, params_g, learning_rate_list, last_grad, r
            )
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL


# ---------------------------------------------------------------------------
# Optax wrapper (duplicate in original; kept for compatibility)
# ---------------------------------------------------------------------------

class optax_optimize(optimizer):
    """Wrapper around any Optax optimizer for EPFlat networks.

    Parameters
    ----------
    grad_method      : EP_grad instance.
    nn               : flat Network instance.
    network_params_0 : initial parameters.
    optimizer        : Optax optimizer constructor (e.g. ``optax.adam``).
    """

    def __init__(self, grad_method, nn, network_params_0, optimizer):
        super().__init__(grad_method, nn, network_params_0)
        self.optimizer = optimizer

    def train(self, N_epoch, learning_rate, input_data, target,
              show_process=False, dynamical_save=False, suffix=None):
        """Run training with the Optax optimizer."""
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []

        solver = self.optimizer(learning_rate=learning_rate)
        opt_state = solver.init(running_params)

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()
            updates, opt_state = solver.update(params_g, opt_state, running_params)
            running_params = optax.apply_updates(running_params, updates)
            t2 = time.time()

            paramsL.append(running_params)
            costL.append(cost)

            if show_process:
                print(k, "current cost =", cost,
                      "thermalization time:", t1 - t0,
                      "update time:", t2 - t1)

            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)

        return costL, paramsL
