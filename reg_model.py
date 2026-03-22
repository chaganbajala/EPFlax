"""
reg_model.py — Regularized layer and network classes for EPFlax.

This module extends the base ``model.py`` classes by adding L2 weight
regularization.  The regularization term is:

    R(θ) = λ · Σ_l ‖W_l‖²

where λ is the regularization strength and the sum runs over all layers.

The regularizer gradient ∂R/∂θ is automatically computed by JAX autodiff
and added to the EP gradient inside ``Reg_EP_grad`` (see ``grad.py``).

Usage
-----
Use these classes instead of their ``model.py`` counterparts when L2
regularization is desired::

    import EPFlax.reg_model as rm

    nn = rm.Module(cost_func, reg=0.01)   # λ = 0.01
    nn.setup(...)
"""

import jax
import jax.numpy as jnp
import numpy as np
import EPFlax.model as lm


# ---------------------------------------------------------------------------
# Regularized layer classes
# ---------------------------------------------------------------------------

class Layer(lm.Layer):
    """Base regularized layer.

    Adds ``params_norm`` to the base ``Layer`` interface.  The default
    implementation returns 0 (no regularizable parameters).
    """

    def params_norm(self, params):
        """Return the squared L2 norm of this layer's regularizable parameters.

        Subclasses override this to sum over the relevant weight arrays.

        Args:
            params: parameter dict for this layer (e.g. ``{'weights': W, ...}``).

        Returns:
            Scalar — the L2-squared norm used in the regularization term.
        """
        return jnp.sum(jnp.square(params['weights']))


class Denselayer(lm.Denselayer, Layer):
    """Dense (fully connected) layer with L2 weight regularization.

    Regularization is applied to the weight matrix W only (not the bias field).
    The bias field {h, ψ} encodes local fields and phase offsets, which are
    generally not penalized.
    """

    def params_norm(self, params):
        """L2-squared norm of the weight matrix.

        Args:
            params: dict with key ``'weights'``, shape (input_size, output_size).

        Returns:
            Scalar ‖W‖².
        """
        return jnp.sum(jnp.square(params['weights']))


class Conv1D(lm.Conv1D, Denselayer):
    """1-D convolutional layer with L2 kernel regularization."""

    def params_norm(self, params):
        """L2-squared norm of the convolutional kernel.

        Args:
            params: dict with key ``'kernel'``,
                    shape (out_channels, in_channels, filter_size).

        Returns:
            Scalar ‖F‖².
        """
        return jnp.sum(jnp.square(params['kernel']))


class Conv2D(lm.Conv1D, Denselayer):
    """2-D convolutional layer with L2 kernel regularization."""

    def params_norm(self, params):
        """L2-squared norm of the 2-D convolutional kernel.

        Args:
            params: dict with key ``'kernel'``,
                    shape (out_channels, in_channels, kH, kW).

        Returns:
            Scalar ‖F‖².
        """
        return jnp.sum(jnp.square(params['kernel']))


# ---------------------------------------------------------------------------
# Regularized network module
# ---------------------------------------------------------------------------

class Module(lm.Module):
    """Network module with L2 weight regularization.

    Extends ``model.Module`` with a ``regularizer`` method that computes
    R(θ) = λ · Σ_l ‖W_l‖².  The ``Reg_EP_grad`` gradient estimator checks
    for the presence of ``self.reg`` and, if found, adds ∂R/∂θ to the EP
    gradient.

    Parameters
    ----------
    cost_func   : callable — per-sample loss function.
    run_params  : (runtime, rtol, atol) for the ODE thermalizer.
    opt_params  : (tol, maxtime) for optional optimizer-based thermalization.
    optimizer   : optional optimizer for thermalization (None → use ODE).
    network_type: string tag for the network type (default 'general XY').
    structure_name: string tag for the layer structure (default 'dnn').
    reg         : λ, the L2 regularization strength (default 0.1).
    """

    def __init__(self, cost_func, run_params=(100, 1e-3, 1e-6),
                 opt_params=(1e-3, 1000), optimizer=None,
                 network_type='general XY', structure_name='dnn', reg=0.1):
        super().__init__(cost_func, run_params, opt_params, optimizer,
                         network_type, structure_name)
        # λ: regularization strength; stored so Reg_EP_grad can detect it
        self.reg = reg

    def regularizer(self, params):
        """Compute the L2 regularization term R(θ) = λ · Σ_l ‖W_l‖².

        This method is called by ``Reg_EP_grad.get_params_gradient``; JAX
        differentiates it w.r.t. ``params`` to obtain the regularization
        gradient.

        Args:
            params: full network parameter pytree, keyed by layer name.

        Returns:
            Scalar regularization energy R(θ).
        """
        E = 0.
        for name in self.layer_order:
            layer = self.layers[name]
            # Add λ · ‖W_l‖² for each layer (params_norm defined per layer class)
            E += self.reg * layer.params_norm(params[name])
        return E
