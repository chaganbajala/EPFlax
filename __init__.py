"""
EPFlax
======

Equilibrium Propagation (EP) for energy-based neural networks with XY-model
interactions, implemented in JAX.

Submodules
----------
model
    Layer and network architecture classes. Neuron states are continuous phase
    angles θ ∈ (−π, π]; inter-layer interactions follow the XY spin model.
grad
    EP gradient estimators. The gradient is approximated by comparing the
    network energy derivatives at the free equilibrium (β = 0) and the nudged
    equilibrium (β > 0).
train
    Optimizer classes that wrap gradient estimators and update network
    parameters each epoch.
reg_model
    Regularized extensions of the model classes, adding an L2 weight penalty.

Subpackages
-----------
EPFlat
    A simpler, flat-network variant with a single consolidated state vector,
    useful as a lightweight alternative for small experiments.
"""

from EPFlax import model, grad, train, reg_model
