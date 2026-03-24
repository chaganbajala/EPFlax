# EPFlax

A JAX-based implementation of **Equilibrium Propagation (EP)** for energy-based neural networks with XY-model-style interactions.

## Overview

EPFlax implements biologically-motivated, energy-based neural networks where neuron states are continuous phase angles (as in the XY spin model). Training is performed via **Equilibrium Propagation**, a local learning algorithm that estimates gradients by comparing the network's free and nudged equilibrium states — without backpropagation through time.

The library is built on [JAX](https://github.com/google/jax) for hardware-accelerated, JIT-compiled computation, with optional integration with [Optax](https://github.com/google-deepmind/optax) and [Diffrax](https://github.com/patrick-kidger/diffrax).

## Package Structure

```
EPFlax/
├── model.py        # Layer and network architecture definitions
├── grad.py         # Equilibrium Propagation gradient computation
├── train.py        # Optimizer classes for training
├── reg_model.py    # Regularized model extensions
├── EPFlat/         # Simplified flat-structure network variant
│   ├── model.py
│   ├── grad.py
│   └── train.py
└── test.ipynb      # Usage examples
```

### `model.py` — Network Architecture

Layer classes (XY-model interactions using complex exponentials):

| Class | Description |
|---|---|
| `Layer` | Abstract base layer |
| `Denselayer` | Fully connected layer with general coupling and bias functions |
| `Conv1D` / `Conv2D` | 1D/2D convolutional layers |
| `Pool1D` / `Pool2D` | Average pooling layers |
| `TransConv1D` / `TransConv2D` | Transposed convolution (decoder) layers |
| `Unsample1D` / `Unsample2D` | Upsampling layers |
| `Intra_connected` | Dense layer with optional intra-layer connections |

Network classes:

| Class | Description |
|---|---|
| `Network` | Abstract base network (ODE/optimizer-based thermalization) |
| `Module` | Modular network built by stacking layers |
| `Autoencoder` | Encoder-decoder network for generative tasks |
| `Generate_Module` | Used for extract subnetworks viewed as encoder and decoder from a network with the form of an autoencoder |

### `grad.py` — Equilibrium Propagation Gradients

| Class | Description |
|---|---|
| `EP_grad` | Core EP gradient estimator (free phase vs. nudged phase) |
| `Reg_EP_grad` | EP gradient with L2 regularization |

Supports three sampling strategies: `full`, `mini_batch`, and `random_init_mini_batch`.

### `train.py` — Optimizers

| Class | Description |
|---|---|
| `Optimizer` | Abstract base optimizer |
| `Gradient_descent` | Vanilla gradient descent (JIT-compiled update step) |
| `Moment_gradient_descent` | Gradient descent with momentum |
| `Optax_optimize` | Wrapper for any Optax optimizer (Adam, etc.) |
| `Layerwise_mgd` | Momentum GD with layer-wise adaptive learning rates |
| `Layerwise_gd` | Layer-wise adaptive gradient descent |

### `reg_model.py` — Regularized Models

Extends `model.py` layer and module classes with L2 weight regularization via a `regularizer()` method.

## Dependencies

- `jax` / `jaxlib`
- `optax`
- `diffrax`
- `numpy`

## Quick Start

```python
import jax
import jax.numpy as jnp
import EPFlax.model as lm
import EPFlax.grad as eg
import EPFlax.train as et

# 1. Define coupling and bias functions
def coup_func(x, y): return jnp.cos(x - y)
def bias_func(x, b): return b[0] * jnp.cos(x - b[1])

# 2. Build a network module
nn = lm.Module(cost_func=my_cost_func)
nn.add_layer('hidden', lm.Denselayer(64, coup_func, bias_func))
nn.add_layer('output', lm.Denselayer(10, coup_func, bias_func))
nn.compile(input_data_sample)
params0 = nn.get_initial_params(jax.random.PRNGKey(0))

# 3. Set up EP gradient method
grad_method = eg.EP_grad(
    grad_params=(beta, runtime, rtol, atol),
    sample_args=('mini_batch', batch_size, M_init)
)

# 4. Train
optimizer = et.Optax_optimize(grad_method, nn, params0, optax.adam)
costL, paramsL = optimizer.train(N_epoch=500, learning_rate=1e-3,
                                  input_data=X_train, target=y_train,
                                  show_process=True)
```

## Background

**Equilibrium Propagation** (Scellier & Bengio, 2017) is a learning algorithm for energy-based models. The network is relaxed to a free-phase equilibrium, then weakly nudged toward the target, and the gradient is estimated from the difference in parameter derivatives between the two phases:

$$
\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{1}{\beta} \left[ \left.\frac{\partial E}{\partial \theta}\right|_{\text{nudged}} - \left.\frac{\partial E}{\partial \theta}\right|_{\text{free}} \right]
$$

where $\beta$ is the nudging strength, $E$ is the network energy, and $\theta$ are the trainable parameters.

This formulation is compatible with XY-model networks where the inter-layer energy is defined via phase differences,

$$
E(\theta_i, \theta_j) = W_{ij} \cos(\theta_i - \theta_j),
$$

naturally supporting complex-valued and oscillatory representations through terms of the form $e^{i\theta}$.

## References

- Scellier, B. & Bengio, Y. (2017). [Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full). *Frontiers in Computational Neuroscience*.
