"""
train.py — Optimizer classes for training EPFlax networks.

Each optimizer wraps an EP gradient estimator (from ``grad.py``) and a network
module, and exposes a ``train`` method that runs the gradient-descent loop.

All optimizers share the same interface::

    optimizer = SomeOptimizer(grad_method, nn, initial_params)
    costL, paramsL = optimizer.train(N_epoch, learning_rate,
                                     input_data, target,
                                     show_process=True)

where ``costL`` is a list of per-epoch losses and ``paramsL`` is a list of
parameter pytrees (one per epoch, starting from the initial params).
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import time
import pickle
from functools import partial


class Optimizer:
    """Abstract base class for EPFlax optimizers.

    Subclasses must implement ``optimize_step`` and ``train``.

    Parameters
    ----------
    grad_method : EP_grad (or subclass)
        Gradient estimator that provides ``grad_method.grad_func``.
    nn : Module (or subclass)
        The network being trained.
    network_params_0 : pytree
        Initial network parameters.
    """

    def __init__(self, grad_method, nn, network_params_0):
        self.grad_method = grad_method
        self.nn = nn
        self.network_params_0 = network_params_0

    def optimize_step(self, network_params, g_params):
        """Perform a single parameter update. To be overridden."""
        pass

    def train(self, input_data, target, show_process=False, dynamical_saving=False, suffix=None):
        """Run the full training loop. To be overridden."""
        pass


# ---------------------------------------------------------------------------
# Vanilla gradient descent
# ---------------------------------------------------------------------------

class Gradient_descent(Optimizer):
    """Vanilla (stochastic) gradient descent.

    Parameter update rule:
        θ ← θ − η · ∂L/∂θ

    where η is the learning rate.
    """

    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate):
        """JIT-compiled single gradient-descent step.

        Args:
            network_params : current parameter pytree θ.
            g_params       : gradient pytree ∂L/∂θ.
            learning_rate  : scalar step size η.

        Returns:
            Updated parameter pytree θ − η · g.
        """
        def add_func(x, gx, learning_rate):
            return x - gx * learning_rate

        return jax.tree_map(lambda x, gx: add_func(x, gx, learning_rate), network_params, g_params)

    def train(self, N_epoch, learning_rate, input_data, target,
              show_process=False, dynamical_save=False, suffix=None):
        """Run gradient-descent training.

        Args:
            N_epoch       : number of training epochs.
            learning_rate : step size η.
            input_data    : training inputs, shape (N_data, ...).
            target        : training targets, shape (N_data, ...).
            show_process  : if True, print cost and timing each epoch.
            dynamical_save: if True, pickle ``paramsL`` and ``costL`` to disk
                            after every epoch (useful for long runs).
            suffix        : filename suffix for saved files when dynamical_save=True.

        Returns:
            costL  : list of per-epoch mean loss values.
            paramsL: list of parameter pytrees, length N_epoch + 1
                     (index 0 is the initial params).
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

class Moment_gradient_descent(Gradient_descent):
    """Gradient descent with (heavy-ball) momentum.

    Update rule:
        g_eff ← ∂L/∂θ + r · g_prev
        θ     ← θ − η · g_eff

    where r ∈ [0, 1) is the momentum coefficient and g_prev is the effective
    gradient from the previous step.
    """

    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate, last_grad, r):
        """JIT-compiled momentum update step.

        Args:
            network_params : current parameter pytree θ.
            g_params       : current EP gradient ∂L/∂θ.
            learning_rate  : step size η.
            last_grad      : effective gradient from the previous step.
            r              : momentum coefficient (0 = no momentum, →1 = heavy momentum).

        Returns:
            (new_grad, new_params) — updated effective gradient and parameters.
        """
        # Accumulate momentum: g_eff = g + r * g_prev
        def moment_func(gx, lgx, r): return gx + r * lgx
        def add_func(x, tot_gx, learning_rate): return x - tot_gx * learning_rate

        new_grad = jax.tree_map(lambda gx, lgx: moment_func(gx, lgx, r), g_params, last_grad)
        new_params = jax.tree_map(lambda x, tot_gx: add_func(x, tot_gx, learning_rate), network_params, new_grad)

        return new_grad, new_params

    def train(self, N_epoch, learning_rate, input_data, target,
              r=0.9, show_process=False, dynamical_save=False, suffix=None):
        """Run momentum gradient-descent training.

        Args:
            N_epoch       : number of training epochs.
            learning_rate : step size η.
            input_data    : training inputs.
            target        : training targets.
            r             : momentum coefficient (default 0.9).
            show_process  : print progress if True.
            dynamical_save: save to disk each epoch if True.
            suffix        : filename suffix for saved files.

        Returns:
            costL, paramsL
        """
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []

        # Initialise effective gradient to zero (same structure as params)
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
# Optax-based optimizer
# ---------------------------------------------------------------------------

class Optax_optimize(Optimizer):
    """Wrapper around any Optax optimizer (Adam, RMSProp, etc.).

    Parameters
    ----------
    grad_method    : EP_grad instance.
    nn             : network Module instance.
    network_params_0: initial parameters.
    optimizer      : an Optax optimizer *constructor* (not an instance), e.g.
                     ``optax.adam``.  It will be called as
                     ``optimizer(learning_rate=lr)`` inside ``train``.
    """

    def __init__(self, grad_method, nn, network_params_0, optimizer):
        super().__init__(grad_method, nn, network_params_0)
        # Store the constructor; the solver instance is created in train()
        # so the learning rate can be set at training time.
        self.optimizer = optimizer

    def train(self, N_epoch, learning_rate, input_data, target,
              show_process=False, dynamical_save=False, suffix=None):
        """Run training with the chosen Optax optimizer.

        Args:
            N_epoch       : number of training epochs.
            learning_rate : passed to the Optax optimizer constructor.
            input_data    : training inputs.
            target        : training targets.
            show_process  : print progress if True.
            dynamical_save: save to disk each epoch if True.
            suffix        : filename suffix for saved files.

        Returns:
            costL, paramsL
        """
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []

        # Build the Optax solver and initialise its state
        solver = self.optimizer(learning_rate=learning_rate)
        opt_state = solver.init(running_params)

        for k in range(N_epoch):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(
                input_data, target, self.nn, running_params,
                self.grad_method.batch_size, self.grad_method.M_init
            )
            t1 = time.time()

            # Optax computes the actual update from the raw gradient
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


# ---------------------------------------------------------------------------
# Layer-wise adaptive optimizers
# ---------------------------------------------------------------------------

class Layerwise_mgd(Moment_gradient_descent):
    """Momentum gradient descent with self-adjusted layer-wise learning rates.

    Instead of a single global learning rate, each layer receives its own
    effective step size proportional to ``‖W‖ / ‖∂L/∂W‖``, so that the
    relative update size is normalised across layers.  This can stabilise
    training of deep networks where gradient magnitudes vary by orders of
    magnitude between layers.
    """

    def get_tot_grad(self, g_params, old_params, r):
        """Combine current gradient with momentum term.

        Args:
            g_params   : current EP gradient pytree.
            old_params : effective gradient from the previous step (momentum buffer).
            r          : momentum coefficient.

        Returns:
            Effective gradient g_eff = g + r * g_prev (pytree).
        """
        def g_func(g, lg, r): return g + r * lg
        return jax.tree_map(lambda g, lg: g_func(g, lg, r), g_params, old_params)

    def layer_wise_update(self, params, g_params, learning_rate):
        """Apply layer-wise parameter update using per-layer learning rates.

        The per-layer learning rates are obtained from the network via
        ``nn.get_normalized_learning_rate``, which scales the base rate by
        ``‖W_l‖ / ‖∂L/∂W_l‖`` for each layer l.

        Args:
            params        : current parameter pytree.
            g_params      : effective gradient pytree.
            learning_rate : base learning rate (scalar).

        Returns:
            Updated parameter pytree.
        """
        def single_update_func(x, gx, learning_rate):
            return x - learning_rate * gx

        # Get per-layer rates; each entry scales the update for that layer
        learning_rate_list = self.nn.get_normalized_learning_rate(params, g_params, learning_rate)
        return jax.tree_map(single_update_func, params, g_params, learning_rate_list)

    def optimize_step(self, running_params, params_g, learning_rate, last_grad, r):
        """Single momentum + layer-wise update step.

        Args:
            running_params: current θ.
            params_g      : raw EP gradient.
            learning_rate : base step size.
            last_grad     : previous effective gradient (momentum buffer).
            r             : momentum coefficient.

        Returns:
            (tot_g, new_params) — updated momentum buffer and parameters.
        """
        tot_g = self.get_tot_grad(params_g, last_grad, r)
        new_params = self.layer_wise_update(running_params, tot_g, learning_rate)
        return tot_g, new_params

    def train(self, N_epoch, learning_rate, input_data, target,
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


class Layerwise_gd(Layerwise_mgd):
    """Layer-wise adaptive gradient descent (no momentum).

    Equivalent to ``Layerwise_mgd`` with r = 0.
    """

    def train(self, N_epoch, learning_rate, input_data, target,
              r=0., show_process=False, dynamical_save=False, suffix=None):
        """Run layer-wise GD training (momentum disabled by setting r=0)."""
        return super().train(N_epoch, learning_rate, input_data, target,
                             r, show_process, dynamical_save, suffix)
