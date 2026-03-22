"""
EPFlat/reg_model.py — Regularized network classes for flat EPFlat networks.

Extends the flat network classes from ``EPFlat/model.py`` with an L2
regularization term:

    R(θ) = λ · ‖W‖²

The regularizer gradient is used by ``EPFlat.grad.Reg_EP_grad`` to add a
weight-decay correction to the EP gradient.

The flat representation stores weights differently from EPFlax:
- ``General_XY_Network``       : params = (W, bias) where W is a single matrix.
- ``Layered_General_XY_Network``: params = (WL, bias) where WL is a list of
                                   per-layer matrices.
- ``Square_Lattice``            : params = (Ws, bias) where Ws is a list of
                                   neighbourhood coupling matrices.
"""

import numpy as np
import jax
import jax.numpy as jnp
import EPFlax.EPFlat.model as fm


class Network(fm.Network):
    """Base regularized flat network.

    Adds ``set_reg``, ``params_norm``, and ``regularizer`` methods to the
    flat ``Network`` interface.  The defaults return 0 (no regularization).
    """

    def set_reg(self, reg=0.1):
        """Set the regularization strength λ.

        Calling this method marks the network as regularized, so that
        ``Reg_EP_grad`` will include the regularizer gradient during training.

        Args:
            reg: L2 regularization strength λ (default 0.1).
        """
        self.reg = reg

    def params_norm(self, params):
        """Squared L2 norm of regularizable parameters (default: 0).

        Subclasses override this to compute ‖W‖² for their specific parameter
        layout.

        Args:
            params: network parameter pytree.

        Returns:
            Scalar — the squared norm contributing to R(θ).
        """
        return 0.

    def regularizer(self, params):
        """Compute the regularization energy R(θ) (default: 0).

        Subclasses override this to return λ · ‖W‖² for their layout.

        Args:
            params: network parameter pytree.

        Returns:
            Scalar regularization energy.
        """
        return 0.


class General_XY_Network(fm.General_XY_Network, Network):
    """Regularized general XY network (flat, single weight matrix).

    The parameter layout is ``params = (W, bias)`` where W has shape
    (N_hidden, N_total).  The regularizer penalizes W only.
    """

    def params_norm(self, params):
        """‖W‖² for the global weight matrix.

        Args:
            params: (W, bias); W is params[0].

        Returns:
            Scalar ‖W‖² = Σ_{ij} W_{ij}².
        """
        # params[0] is the weight matrix W
        return jnp.sum(params[0] * params[0])

    def regularizer(self, params):
        """R(θ) = λ · ‖W‖² for the single weight matrix.

        Args:
            params: (W, bias).

        Returns:
            Scalar λ · ‖W‖².
        """
        return self.reg * self.params_norm(params)


class Layered_General_XY_Network(fm.Layered_General_XY_Network, Network):
    """Regularized layered XY network (flat, list of per-layer weight matrices).

    The parameter layout is ``params = (WL, bias)`` where WL is a list of
    matrices [W_1, W_2, ...], one per layer.
    """

    def params_norm(self, W):
        """‖W‖² for a single layer's weight matrix.

        Args:
            W: 2-D weight matrix for one layer.

        Returns:
            Scalar ‖W‖².
        """
        return jnp.sum(W * W)

    def regularizer(self, params):
        """R(θ) = λ · Σ_l ‖W_l‖² summed over all layers.

        Args:
            params: (WL, bias) where WL = params[0] is a list of weight matrices.

        Returns:
            Scalar total regularization energy.
        """
        # Sum ‖W_l‖² across all layers
        return self.reg * sum([self.params_norm(W) for W in params[0]])


class Square_Lattice(fm.Square_Lattice, Network):
    """Regularized square-lattice XY network.

    In the square-lattice variant, the weights are stored as a list of
    neighbourhood coupling arrays Ws = [W_h, W_v, ...] (horizontal, vertical,
    etc.).  The regularizer sums ‖W‖² over all coupling arrays.
    """

    def params_norm(self, Ws):
        """Sum of ‖W‖² over all neighbourhood coupling matrices.

        Args:
            Ws: list of coupling arrays (one per neighbour direction).

        Returns:
            Scalar Σ_d ‖W_d‖².
        """
        return sum([jnp.sum(W * W) for W in Ws])

    def regularizer(self, params):
        """R(θ) = Σ_d ‖W_d‖² for the square-lattice couplings.

        Note: the regularization strength λ is not applied here (unlike the
        other classes).  If needed, multiply the result by ``self.reg``
        externally or override this method.

        Args:
            params: (Ws, bias) where Ws = params[0] is the list of coupling arrays.

        Returns:
            Scalar total coupling norm.
        """
        return self.params_norm(params[0])
