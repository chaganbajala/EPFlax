"""
EPFlat
======

A simplified, flat-network variant of EPFlax.

Rather than organising state as a dict keyed by layer name (as in EPFlax),
EPFlat represents the full network state as a single concatenated 1-D vector.
This makes it easier to interface with generic ODE solvers and avoids the
overhead of JAX pytree handling, at the cost of less flexibility in network
architecture.

Submodules
----------
model
    Network class with a flat state vector and ODE-based thermalization.
grad
    EP gradient estimator operating on the flat state representation.
train
    Optimizer classes compatible with the flat network interface.
"""

from EPFlat import model, grad, train
