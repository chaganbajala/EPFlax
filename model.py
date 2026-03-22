"""
model.py — XY-model Equilibrium Propagation (EP) framework implemented in JAX/diffrax.

Overview
--------
This module implements a family of neural-network layers and network containers whose
learning rule is Equilibrium Propagation (EP) applied to the XY (rotor) model.  Every
neuron stores a *phase angle* θ ∈ (-π, π].  The energy between adjacent layers l-1 and l
is the generalised XY interaction

    E  =  Σ_{i,j}  W_{ij} · coup_func(θ^{l-1}_i, θ^l_j)

For the default coupling  coup_func(a, b) = -cos(a - b)  this reduces to the standard
XY ferromagnetic energy.  In the complex-exponential representation used throughout the
convolutional layers the same quantity is

    E  =  -Re[ Σ_{ij} W_{ij} e^{i(θ^l_j - θ^{l-1}_i)} ]
        =  -Re[ e^{iθ^l} · conv(e^{-iθ^{l-1}}, F) ]

where F is the (complex) weight kernel.  This identity lets convolutions be evaluated
with a single call to jax.lax.conv on complex arrays.

Bias field
----------
Each neuron in a trainable layer carries a local bias field parameterised as
[h, ψ].  The energy contribution of this field is

    E_bias = -h · cos(θ - ψ)

i.e. a field of magnitude h that prefers angle ψ.

Forces and EP
-------------
The *force* on a neuron is the negative gradient of the total energy with respect to
its angle:

    F_θ = -∂E/∂θ

During the *free phase* (β = 0) the network state y evolves under

    ẏ = F_int(y, θ)

until equilibrium.  During the *nudged phase* (β > 0) an additional external term is
added:

    ẏ = F_int(y, θ) + β · F_ext(y, t, θ)

Both ODEs are integrated with diffrax (Tsit5 adaptive-step solver).

Module layout
-------------
Layer (base)
├── Denselayer          — fully-connected dense layer
│   ├── Conv1D          — 1-D convolutional layer
│   │   ├── Pool1D      — 1-D average-pooling layer (no trainable params)
│   │   └── TransConv1D — 1-D transposed convolution
│   │       └── Unsample1D  — 1-D learned upsampling
│   ├── Conv2D          — 2-D convolutional layer
│   │   ├── Pool2D      — 2-D average-pooling layer (no trainable params)
│   │   └── TransConv2D — 2-D transposed convolution
│   │       └── Unsample2D  — 2-D learned upsampling
│   └── Intra_connected — dense layer with optional within-layer coupling
Network (base)
└── Module              — general EP network container
    ├── Autoencoder     — encoder + decoder network with split internal energy
    ├── Generate_Module — helper to carve out sub-networks (e.g. encoder alone)
    ├── My_nn           — example inference network template
    └── My_AE           — concrete autoencoder example
"""

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import optax
import time
import pickle
import gc
# functools.partial is used to JIT-compile instance methods by fixing 'self' as static
from functools import partial


# =============================================================================
#  Base Layer
# =============================================================================

class Layer:
    """Abstract base class for all XY-model EP layers.

    Every concrete layer must implement:
        get_layer_size  — inspect incoming data shape, record input dimensions.
        get_init_params — return a dict of initial trainable parameters.
        setup           — wire up the correct energy/force methods after
                          get_layer_size has set self.former_layer_type.
        energy          — scalar XY interaction energy between two adjacent states.
        force           — (FF, BF) forward and backward forces derived from energy.

    Optional override:
        get_layer_ratio — used for per-layer learning-rate normalisation.
        params_norm     — regularisation helper (default: 0).
    """

    def __init__(self):
        pass

    def get_layer_size(self, input_data, former_layer_type):
        """Inspect the previous layer's output and determine this layer's shape.

        Args:
            input_data (jnp.ndarray): A sample output tensor from the preceding layer.
            former_layer_type (str): Layer-type string of the preceding layer
                (e.g. 'dense', 'conv1D', 'conv2D').

        Returns:
            tuple: (sample_output, n_nodes, structure_string)
                sample_output  — zero array with the shape of this layer's output.
                n_nodes        — total number of neurons (scalar int).
                structure_string — human-readable shape descriptor.
        """
        pass

    def get_init_params(self):
        """Return a dict of randomly initialised trainable parameters."""
        pass

    def energy(self):
        """Compute the scalar XY interaction energy for this layer."""
        pass

    def forward_force(self):
        """Compute the forward force on this layer's neurons (stub)."""
        pass

    def backward_force(self):
        """Compute the backward force fed to the previous layer (stub)."""
        pass

    def get_layer_ratio(self, params, g_params):
        """Compute the ratio ‖W‖ / ‖∂L/∂W‖ for every parameter tensor.

        This ratio is used in get_normalized_learning_rate to scale per-layer
        learning rates so that the relative update magnitude ‖ΔW‖/‖W‖ is
        equalised across layers (see Module.get_normalized_learning_rate).

        Args:
            params   (dict): Current parameter values for this layer.
            g_params (dict): Gradient of the loss w.r.t. each parameter.

        Returns:
            dict: Same keys as params; each value is the scalar ratio.
        """
        ratio = {}
        abs_params = 0.
        abs_g = 0.

        for name in params:
            # Accumulate squared Frobenius norms across all parameter tensors
            abs_params = abs_params + jnp.linalg.norm(params[name])**2
            abs_g = abs_g + jnp.linalg.norm(g_params[name])**2

            # Take the square root to get Frobenius norms
            abs_params = jnp.sqrt(abs_params)
            abs_g = jnp.sqrt(abs_g)

        for name in params:
            # Each parameter gets the same scalar ratio ‖params‖ / ‖grad‖
            ratio.update({name: abs_params / abs_g})

        return ratio

    def params_norm(self, params):
        """Return a regularisation norm for this layer's parameters (default 0).

        Args:
            params (dict): Parameter dict for this layer.

        Returns:
            float: Regularisation penalty (0 by default).
        """
        return 0.


# =============================================================================
#  Dense (fully-connected) layer
# =============================================================================

class Denselayer(Layer):
    """Fully-connected XY-model layer.

    Each neuron pair (i in layer l-1, j in layer l) interacts via:

        E_ij = W_{ij} · coup_func(θ^{l-1}_i, θ^l_j)

    For the default choice coup_func(a, b) = -cos(a - b) this is the standard
    XY ferromagnetic coupling.  A bias field [h_j, ψ_j] acts on every output
    neuron j:

        E_bias_j = bias_func(θ^l_j, [h_j, ψ_j])  =  -h_j · cos(θ^l_j - ψ_j)

    The weight matrix W has shape (input_size, output_size) and is initialised
    with Xavier-style scaling  W ~ N(0, 1/√(n_in + n_out)).

    Attributes:
        output_size (int): Number of neurons in this layer.
        layer_type  (str): Always 'dense'.
        coup_func   (callable): Scalar coupling function f(θ_i, θ_j).
        bias_func   (callable): Scalar bias function f(θ_j, bias_params).
    """

    def __init__(self, output_size, coup_func, bias_func, layer_type='dense'):
        """Initialise the dense layer and pre-build vmapped derivative functions.

        Args:
            output_size (int): Number of output neurons.
            coup_func   (callable): Pairwise coupling function f(θ_i, θ_j).
            bias_func   (callable): Bias energy function f(θ, [h, ψ]).
            layer_type  (str): Layer-type tag (default 'dense').
        """
        self.output_size = output_size
        self.layer_type = layer_type
        self.bias_func = bias_func
        self.coup_func = coup_func

        # Scalar partial derivatives of the coupling function
        # d0_coup = ∂coup/∂θ_i  (used for backward force BF = -∂E/∂θ^{l-1})
        # d1_coup = ∂coup/∂θ_j  (used for forward  force FF = -∂E/∂θ^l)
        self.d0_coup = jax.grad(coup_func, 0)
        self.d1_coup = jax.grad(coup_func, 1)

        # vd0_coup(θ_i, y2): vectorise over θ_j (axis 0 fixed, axis 0 mapped)
        self.vd0_coup = jax.vmap(self.d0_coup, (None, 0))
        # md0_coup(y1, y2): also vectorise over θ_i to get full matrix
        self.md0_coup = jax.vmap(self.vd0_coup, (0, None))

        # Analogous vmapped derivatives w.r.t. the second argument θ_j
        self.vd1_coup = jax.vmap(self.d1_coup, (None, 0))
        self.md1_coup = jax.vmap(self.vd1_coup, (0, None))

        # Scalar partial derivatives of the bias energy function
        self.d0_bias = jax.grad(self.bias_func, 0)   # ∂E_bias/∂θ_j
        self.d1_bias = jax.grad(self.bias_func, 1)   # ∂E_bias/∂bias_params (not used in force)

        # Vectorised bias gradient and energy over the output neuron axis
        self.vd0_bias = jax.vmap(self.d0_bias, (0, 0))
        self.vd1_bias = jax.vmap(self.d1_bias, (0, 0))

        # Vectorised energy evaluations for computing E directly
        self.vbias = jax.vmap(self.bias_func, (0, 0))   # bias energy, one per neuron
        self.vcoup = jax.vmap(self.coup_func, (None, 0))  # coupling, fixed θ_i over all θ_j
        self.mcoup = jax.vmap(self.vcoup, (0, None))      # coupling, full matrix over (θ_i, θ_j)

    def get_layer_size(self, input_data, former_layer_type):
        """Determine input size by flattening whatever shape the previous layer produces.

        Args:
            input_data       (jnp.ndarray): Sample output from the preceding layer.
            former_layer_type (str): Type tag of the preceding layer.

        Returns:
            tuple: (zero_output, output_size, structure_string)
        """
        self.former_layer_type = former_layer_type
        # Dense layer always flattens its input regardless of spatial structure
        self.input_size = input_data.flatten().shape[0]
        return jnp.zeros(self.output_size), self.output_size, '{0}'.format(self.output_size)

    def get_init_params(self, rng):
        """Randomly initialise weight matrix and bias field.

        Weights are Xavier-initialised:  W ~ N(0, 1/√(n_in + n_out)).
        Bias magnitudes h are initialised to 0 (no bias field initially).
        Bias preferred angles ψ are drawn uniformly from (-π, π).

        Args:
            rng: JAX PRNG key.

        Returns:
            dict: {'weights': W, 'bias field': array of shape (output_size, 2)}
                  where column 0 is h and column 1 is ψ.
        """
        W = jax.random.normal(rng, [self.input_size, self.output_size]) / jnp.sqrt(self.input_size + self.output_size)
        h = jnp.zeros(self.output_size)                                            # zero initial bias magnitude
        psi = jax.random.uniform(rng, shape=[self.output_size], minval=-jnp.pi, maxval=jnp.pi)
        # Stack h and ψ as columns: shape (output_size, 2)
        return {'weights': W, 'bias field': jnp.asarray([h, psi]).transpose()}

    def setup(self):
        """Select energy/force implementation based on the predecessor layer type.

        Called by Module.get_initial_params after get_layer_size has run and
        self.former_layer_type is known.

        If the previous layer is also dense, inputs arrive as a 1-D vector and
        energy_std / force_std apply directly.  Otherwise (e.g. after a conv
        layer) the input must first be flattened, so energy_flatten /
        force_flatten are used instead.
        """
        if self.former_layer_type == 'dense':
            # Input is already a flat vector; use the standard implementations
            self.energy = self.energy_std
            self.force = self.force_std
        else:
            # Input has spatial structure (e.g. conv feature map); flatten first
            self.energy = self.energy_flatten
            self.force = self.force_flatten

    def get_init_state(self, N_data):
        """Return random initial angles for a batch of N_data samples.

        Args:
            N_data (int): Batch size.

        Returns:
            np.ndarray: Shape (N_data, output_size), angles uniform in (-π, π).
        """
        y0 = np.pi * 2 * (np.random.rand(N_data, self.output_size) - 0.5)
        return y0

    def force_std(self, y1, params, y2):
        """Compute forward and backward forces for a dense→dense connection.

        The XY interaction energy between layers is
            E = Σ_{ij} W_{ij} · coup_func(y1_i, y2_j)

        Forward force (FF) on layer-l neurons: FF_j = -∂E/∂y2_j
            = -Σ_i W_{ij} · d1_coup(y1_i, y2_j)
        Backward force (BF) on layer-(l-1) neurons: BF_i = -∂E/∂y1_i
            = -Σ_j W_{ij} · d0_coup(y1_i, y2_j)
        Output force (OF): bias contribution to FF
            OF_j = -∂E_bias/∂y2_j = -vd0_bias(y2_j, bias_j)

        Note: the sign convention in coup_func is already absorbed —
        with coup_func = -cos(a-b), grad w.r.t. first arg gives +sin(a-b).

        Args:
            y1     (jnp.ndarray): Angles from the previous layer, shape (n_in,).
            params (dict):        Layer parameters with keys 'weights', 'bias field'.
            y2     (jnp.ndarray): Angles at this layer, shape (n_out,).

        Returns:
            tuple: (FF + OF, BF)  — both arrays have the same shape as y2 and y1
                   respectively.
        """
        W, bias = params['weights'], params['bias field']
        # FF_j = Σ_i W_{ij} · ∂coup/∂y2_j  (matrix-vector product structure)
        FF = jnp.sum(W * self.md0_coup(y1, y2), axis=0)
        # BF_i = Σ_j W_{ij} · ∂coup/∂y1_i
        BF = jnp.sum(W * self.md1_coup(y1, y2), axis=1)
        # OF_j = -∂E_bias/∂y2_j
        OF = -self.vd0_bias(y2, bias)

        return FF + OF, BF

    def energy_std(self, y1, params, y2):
        """Compute the total XY energy (coupling + bias) for a dense→dense edge.

        E = Σ_{ij} W_{ij} · coup_func(y1_i, y2_j)  +  Σ_j bias_func(y2_j, bias_j)

        Args:
            y1     (jnp.ndarray): Previous-layer angles, shape (n_in,).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): This-layer angles, shape (n_out,).

        Returns:
            float: Scalar total energy.
        """
        W, bias = params['weights'], params['bias field']
        # Coupling energy: elementwise W * coup matrix, then sum all entries
        E0 = jnp.sum(W * self.mcoup(y1, y2))
        # Bias energy: sum over output neurons
        E1 = jnp.sum(self.vbias(y2, bias))
        return E0 + E1

    def force_flatten(self, y1, params, y2):
        """Wrapper around force_std that flattens spatially-structured input.

        Used when the preceding layer is convolutional and y1 has shape
        (channels, spatial…).  The BF is reshaped back to match y1's original
        shape before returning.

        Args:
            y1     (jnp.ndarray): Previous-layer output, arbitrary shape.
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): This-layer angles, shape (n_out,).

        Returns:
            tuple: (FF, BF) where BF has the same shape as y1.
        """
        ny1 = y1.flatten()                          # flatten spatial dims to 1-D
        FF, BF = self.force_std(ny1, params, y2)
        BF = BF.reshape(*y1.shape)                  # restore original spatial shape
        return FF, BF

    def energy_flatten(self, y1, params, y2):
        """Wrapper around energy_std that flattens a spatially-structured input.

        Args:
            y1     (jnp.ndarray): Previous-layer output, arbitrary shape.
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): This-layer angles, shape (n_out,).

        Returns:
            float: Scalar XY energy.
        """
        ny1 = y1.flatten()
        return self.energy_std(ny1, params, y2)

    def get_layer_ratio(self, params, g_params):
        """Compute ‖W‖ / ‖∂L/∂W‖ for the weight matrix.

        Both 'weights' and 'bias field' receive the same ratio so that the
        optimizer scales both with the same per-layer multiplier.

        Args:
            params   (dict): Current parameters {'weights': W, 'bias field': B}.
            g_params (dict): Gradients with the same keys.

        Returns:
            dict: {'weights': r, 'bias field': r} where r = ‖W‖ / ‖∇W‖.
        """
        ratio = {}
        r = jnp.linalg.norm(params['weights']) / jnp.linalg.norm(g_params['weights'])
        ratio.update({'weights': r})
        # Bias field shares the same normalisation scalar as the weights
        ratio.update({'bias field': r})

        '''
        ratio.update({
            'bias field': jnp.sqrt(jnp.square(g_params['bias field'][0])
                                   + jnp.square(params['bias field'][0]) * jnp.square(g_params['bias field'][1])).sum()
        })
        '''

        return ratio


# =============================================================================
#  1-D Convolutional layer
# =============================================================================

class Conv1D(Denselayer):
    """1-D convolutional XY-model layer.

    The coupling energy between a 1-D input feature map y1 of shape
    (in_channels, L_in) and output map y2 of shape (out_channels, L_out) is

        E_conv = -Re[ Σ_c e^{i·y2_c} · conv(e^{-i·y1}, F_c) ]

    where F is the complex-valued kernel of shape (out_channels, in_channels,
    filter_shape) and conv is jax.lax.conv with the specified stride and
    'SAME' padding.

    This is the convolutional generalisation of the XY coupling:
        E_XY = -Re[ e^{i(θ_j - θ_i)} ]  ≡  -cos(θ_j - θ_i)
    with weight sharing enforced by the kernel.

    Bias fields act per output channel (one [h, ψ] pair per channel).

    Attributes:
        output_channel (int): Number of output feature-map channels.
        filter_shape   (int): Kernel spatial extent.
        strides        (tuple): Convolution stride.
        layer_type     (str): Always 'conv1D'.
    """

    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv1D'):
        """Initialise a 1-D convolutional layer.

        Args:
            output_channel (int):      Number of output channels.
            filter_shape   (int):      Spatial extent of the convolution kernel.
            strides        (tuple/int): Convolution stride along the spatial axis.
            coup_func      (callable): Pairwise coupling function (kept for API compat).
            bias_func      (callable): Bias energy function f(θ, [h, ψ]).
            layer_type     (str):      Type tag (default 'conv1D').
        """
        self.coup_func = coup_func
        self.bias_func = bias_func

        self.layer_type = layer_type

        self.output_channel = output_channel
        self.filter_shape = filter_shape

        # Vectorised bias energy and its gradient over the spatial/channel axes
        self.vbias = jax.vmap(self.bias_func, (0, 0))

        self.d0_bias = jax.grad(self.bias_func, 0)
        self.d1_bias = jax.grad(self.bias_func, 1)

        self.vd0_bias = jax.vmap(self.d0_bias, (0, 0))
        self.vd1_bias = jax.vmap(self.d1_bias, (0, 0))

        self.strides = strides

    def get_layer_size(self, input_data, former_layer_type):
        """Determine output spatial size by running a dummy convolution.

        Handles three predecessor types:
          'dense'  — treat input as a single-channel 1-D signal.
          'conv1D' — input already has (channels, length) layout.
          'conv2D' — flatten the two spatial dims of a 2-D feature map.

        Args:
            input_data       (jnp.ndarray): Sample from the preceding layer.
            former_layer_type (str):        Type tag of the preceding layer.

        Returns:
            tuple: (sample_output, n_nodes, structure_string)
        """
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            self.input_size = input_data.shape[0]
        elif former_layer_type == 'conv1D':
            self.input_channel, self.input_size = input_data.shape
        elif former_layer_type == 'conv2D':
            # Merge the two spatial dimensions of a 2-D feature map into one
            self.input_channel, self.input_size = input_data.shape[0], input_data.shape[1] * input_data.shape[2]

        # Run a dummy convolution to find the output spatial size
        y = np.zeros([self.input_channel, self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, self.filter_shape])
        y = jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')
        self.output_size = y.shape[-1]

        return y[0], self.output_channel * self.output_size, '{0}*{1}'.format(self.output_channel, self.output_size)

    def get_init_params(self, rng):
        """Randomly initialise the convolutional kernel and bias field.

        Kernel scaling: F ~ N(0, 1/√(n_in + n_out)) where n_in and n_out are
        the total number of input and output activations respectively.

        Args:
            rng: JAX PRNG key.

        Returns:
            dict: {'kernel': F, 'bias field': array of shape (out_channels, 2)}
        """
        S = self.input_channel * self.input_size + self.output_channel * self.output_size
        F = 1 / jnp.sqrt(S) * jax.random.normal(rng, shape=(self.output_channel, self.input_channel, self.filter_shape))
        h = jnp.zeros(self.output_channel)              # per-channel bias magnitude, init 0
        psi = jax.random.uniform(rng, shape=(self.output_channel,), minval=-jnp.pi, maxval=jnp.pi)
        # bias field shape: (out_channels, 2)  — column 0 = h, column 1 = ψ
        return {'kernel': F, 'bias field': jnp.asarray([h, psi]).transpose()}

    def setup(self):
        """Select the correct energy method based on the predecessor layer type.

        Dispatches to:
          energy_std        — when predecessor is another Conv1D.
          energy_from_conv2D — when predecessor is Conv2D (needs spatial flatten).
          energy_from_dense  — when predecessor is a dense layer (needs channel dim).
        """
        if self.former_layer_type == 'conv1D':
            self.energy = self.energy_std
        elif self.former_layer_type == 'conv2D':
            self.energy = self.energy_from_conv2D
        elif self.former_layer_type == 'dense':
            self.energy = self.energy_from_dense

    def get_init_state(self, N_data):
        """Return random initial angles for all output neurons.

        Args:
            N_data (int): Batch size.

        Returns:
            np.ndarray: Shape (N_data, out_channels, out_length), uniform in (-π, π).
        """
        return (np.random.rand(N_data, self.output_channel, self.output_size) - 0.5) * np.pi * 2

    def energy_std(self, y1, params, y2):
        """Compute the conv XY energy when input is a 1-D feature map.

        The energy is computed in the complex domain:
            E = -Re[ e^{i·y2} · conv(e^{-i·y1}, F) ]  +  Σ_c bias_func(y2_c, bias_c)

        Here:
          exp(-1j*y1) maps input angles to input phasors.
          F is cast to complex64 so the convolution is complex-valued.
          exp(1j*y2) * ny2 gives the elementwise inner product in the complex plane.
          Taking -Re[...] recovers the XY cos(θ_j - θ_i) structure.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_channels, L_in).
            params (dict):        {'kernel': F, 'bias field': bias}.
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar energy.
        """
        # Cast kernel to complex so JAX can perform complex convolution
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        # Bias energy summed over all output channels and spatial positions
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            # Standard 1-D conv; the leading None adds the required batch dimension
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Convert input angles to complex phasors: e^{-iθ_1}
        # Then convolve with complex kernel F to obtain the interaction field
        ny2 = conv_func(jnp.exp(-1j * y1), F)
        # Energy: -Re[ e^{iθ_2} · (conv result) ]
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0 + E1

    def energy_from_dense(self, y1, params, y2):
        """Compute conv XY energy when the predecessor is a dense layer.

        A dense output is a 1-D vector; we insert a channel dimension of size 1
        so the standard convolution pathway can be reused.

        Args:
            y1     (jnp.ndarray): Dense layer output, shape (L_in,).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar energy.
        """
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Add a singleton channel dimension: (L_in,) → (1, L_in)
        ny1 = y1[None, :]
        ny2 = conv_func(jnp.exp(-1j * ny1), F)
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0 + E1

    def energy_from_conv2D(self, y1, params, y2):
        """Compute conv XY energy when the predecessor is a 2-D conv layer.

        The 2-D feature map y1 of shape (channels, H, W) is flattened along
        the spatial axes to (channels, H*W) before the 1-D convolution.

        Args:
            y1     (jnp.ndarray): 2-D feature map, shape (in_channels, H, W).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar energy.
        """
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Concatenate spatial rows into a single axis per channel
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        ny2 = conv_func(jnp.exp(-1j * ny1), F)
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0 + E1

    def force(self, y1, params, y2):
        """Compute forward and backward forces via automatic differentiation.

        Since the conv energy is defined in the complex domain, closed-form
        derivative expressions are non-trivial; JAX autodiff handles it cleanly.

        Forward  force: FF = -∂E/∂y2  (force on this layer's neurons).
        Backward force: BF = -∂E/∂y1  (force fed back to the previous layer).

        Args:
            y1     (jnp.ndarray): Input angles from the previous layer.
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): This layer's current angles.

        Returns:
            tuple: (FF, BF)
        """
        # Differentiate energy w.r.t. argument index 2 (y2) for FF
        FF = -jax.grad(self.energy, 2)(y1, params, y2)
        # Differentiate energy w.r.t. argument index 0 (y1) for BF
        BF = -jax.grad(self.energy, 0)(y1, params, y2)

        return FF, BF

    def get_layer_ratio(self, params, g_params):
        """Compute ‖kernel‖ / ‖∂L/∂kernel‖ for learning-rate normalisation.

        Args:
            params   (dict): {'kernel': F, 'bias field': B}.
            g_params (dict): Gradients with the same keys.

        Returns:
            dict: {'kernel': r, 'bias field': r}.
        """
        ratio = {}
        r = jnp.linalg.norm(params['kernel']) / jnp.linalg.norm(g_params['kernel'])
        ratio.update({'kernel': r})
        # Bias field shares the kernel's normalisation ratio
        ratio.update({'bias field': r})

        '''
        ratio.update({
            'bias field': jnp.sqrt(jnp.square(g_params['bias field'][:,0])
                                   + jnp.square(params['bias field'][:,0]) * jnp.square(g_params['bias field'][:,1])).sum()
        })
        '''

        return ratio


# =============================================================================
#  1-D Pooling layer (no trainable parameters)
# =============================================================================

class Pool1D(Conv1D):
    """1-D average-pooling layer for the XY model.

    Uses a fixed uniform averaging kernel (not trainable):
        kernel = [1/filter_shape, …, 1/filter_shape]  (length = filter_shape)

    The energy is the same XY complex-domain expression as Conv1D but with the
    frozen kernel, which averages adjacent input phasors:
        E = -Re[ e^{iθ_2} · pool(e^{-iθ_1}) ]

    Because there are no trainable parameters, get_layer_ratio returns a dummy
    dict so the learning-rate normaliser ignores this layer.
    """

    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv1D'):
        """Initialise a 1-D pooling layer (inherits Conv1D structure).

        Args:
            output_channel (int):      Number of output channels.
            filter_shape   (int):      Pooling window size.
            strides        (tuple/int): Pooling stride.
            coup_func      (callable): Kept for API consistency (not used).
            bias_func      (callable): Kept for API consistency (not used).
            layer_type     (str):      Type tag (default 'conv1D').
        """
        super().__init__(output_channel, filter_shape, strides, coup_func, bias_func, layer_type)

    def get_init_params(self, rng):
        """Construct the fixed averaging kernel; return dummy param dict.

        The kernel is a 1-D uniform average filter of length filter_shape,
        stored as a complex64 array so it can be used in complex convolutions.

        Args:
            rng: JAX PRNG key (unused; kernel is deterministic).

        Returns:
            dict: {'kernel': 0.} — placeholder with no trainable content.
        """
        # Fixed uniform average kernel: each weight = 1/filter_shape
        self.kernel = jnp.ones([self.filter_shape], dtype=jnp.complex64) / self.filter_shape
        return {'kernel': 0.}

    def energy_std(self, y1, params, y2):
        """Compute pool XY energy for a conv1D predecessor.

        No bias term — pooling layers carry no trainable fields.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_channels, L_in).
            params (dict):        Ignored (no trainable params).
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar coupling energy.
        """
        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Apply the fixed averaging kernel in the complex domain
        ny2 = conv_func(jnp.exp(-1j * y1), self.kernel)
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def single_pool_func(self, y):
        """Apply the averaging kernel to a single-channel 1-D signal.

        Args:
            y (jnp.ndarray): 1-D signal, shape (L,).

        Returns:
            jnp.ndarray: Pooled signal, shape (1, L_out).
        """
        # Add batch and channel dims (both size 1) for jax.lax.conv_transpose API
        return jax.lax.conv(y[None, None, ...], self.kernel[None, None, ...], window_strides=self.strides, padding='SAME')[0, ...]

    def pool_func(self, y):
        """Apply the averaging kernel independently to each channel.

        Args:
            y (jnp.ndarray): Multi-channel signal, shape (channels, L).

        Returns:
            jnp.ndarray: Pooled output, shape (channels, L_out).
        """
        # vmap over the channel axis (axis 0)
        return jax.vmap(self.single_pool_func, 0)(y)

    def energy_from_dense(self, y1, params, y2):
        """Pool XY energy when the predecessor is a dense layer.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L_in,).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar energy.
        """
        # Insert channel dimension: (L_in,) → (1, L_in)
        ny1 = y1[None, :]
        ny2 = self.pool_func(jnp.exp(-1j * ny1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def energy_from_conv2D(self, y1, params, y2):
        """Pool XY energy when the predecessor is a 2-D conv layer.

        Args:
            y1     (jnp.ndarray): 2-D feature map, shape (in_channels, H, W).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_channels, L_out).

        Returns:
            float: Scalar energy.
        """
        # Flatten spatial dimensions H, W into a single axis per channel
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        ny2 = self.pool_func(jnp.exp(-1j * ny1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def get_layer_ratio(self, params, g_params):
        """Return a dummy ratio dict (no trainable parameters to normalise).

        Returns:
            dict: {'kernel': 0.}
        """
        return {'kernel': 0.}


# =============================================================================
#  2-D Convolutional layer
# =============================================================================

class Conv2D(Conv1D):
    """2-D convolutional XY-model layer.

    Extends Conv1D to handle 2-D spatial feature maps of shape
    (channels, H, W).  The complex-domain energy is the same:

        E = -Re[ e^{i·y2} · conv2d(e^{-i·y1}, F) ]

    where F has shape (out_channels, in_channels, kH, kW) and the convolution
    uses 'SAME' padding with the specified 2-D strides.

    Attributes:
        filter_shape (list[int, int]): Kernel spatial extent [kH, kW].
        strides      (list[int, int]): Convolution strides [sH, sW].
        layer_type   (str): Always 'conv2D'.
    """

    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv2D'):
        """Initialise a 2-D convolutional layer.

        Args:
            output_channel (int):           Number of output channels.
            filter_shape   (list[int, int]): Kernel size [kH, kW].
            strides        (list[int, int]): Stride [sH, sW].
            coup_func      (callable):       Coupling function (API compat).
            bias_func      (callable):       Bias energy function.
            layer_type     (str):            Type tag (default 'conv2D').
        """
        super().__init__(output_channel, filter_shape, strides, coup_func, bias_func, layer_type)
        # Note: self.filter_shape is now a 2-element list [kH, kW]

    def get_layer_size(self, input_data, former_layer_type):
        """Determine output spatial size from a dummy 2-D convolution.

        Handles predecessor types 'dense', 'conv1D', and 'conv2D'.

        Args:
            input_data       (jnp.ndarray): Sample from the preceding layer.
            former_layer_type (str):        Type tag of the preceding layer.

        Returns:
            tuple: (sample_output, n_nodes, structure_string)
        """
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            # Treat the 1-D dense output as a (L, 1) spatial map
            self.input_size = [input_data.shape[0], 1]
        elif former_layer_type == 'conv1D':
            self.input_channel = input_data.shape[0]
            # Promote 1-D spatial to 2-D by appending a size-1 second axis
            self.input_size = [input_data.shape[1], 1]
        elif former_layer_type == 'conv2D':
            self.input_channel = input_data.shape[0]
            self.input_size = [input_data.shape[1], input_data.shape[2]]

        # Run a dummy 2-D conv to determine output spatial dimensions
        y = np.zeros([self.input_channel, *self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, *self.filter_shape])
        y = jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')
        self.output_size = [y.shape[-2], y.shape[-1]]

        return y[0], self.output_channel * self.output_size[0] * self.output_size[1], '{0}*{1}*{2}'.format(self.output_channel, *self.output_size)

    def get_init_params(self, rng):
        """Randomly initialise the 2-D convolutional kernel and bias field.

        Args:
            rng: JAX PRNG key.

        Returns:
            dict: {'kernel': F shape (out_ch, in_ch, kH, kW),
                   'bias field': shape (out_ch, 2)}
        """
        S = self.input_channel * self.input_size[0] * self.input_size[1] + self.output_channel * self.output_size[0] * self.output_size[1]
        F = 1 / jnp.sqrt(S) * jax.random.normal(rng, shape=(self.output_channel, self.input_channel, *self.filter_shape))
        h = jnp.zeros(self.output_channel)
        psi = jax.random.uniform(rng, shape=(self.output_channel,), minval=-jnp.pi, maxval=jnp.pi)
        return {'kernel': F, 'bias field': jnp.asarray([h, psi]).transpose()}

    def get_init_state(self, N_data):
        """Return random initial angles for all output neurons.

        Args:
            N_data (int): Batch size.

        Returns:
            np.ndarray: Shape (N_data, out_channels, H_out, W_out).
        """
        return (np.random.rand(N_data, self.output_channel, *self.output_size) - 0.5) * np.pi * 2

    def setup(self):
        """Select energy method based on predecessor layer type.

        Dispatches to:
          energy_std         — Conv2D → Conv2D.
          energy_from_conv1D — Conv1D → Conv2D (expand W axis).
          energy_from_dense  — dense → Conv2D (expand channel and W axes).
        """
        if self.former_layer_type == 'conv2D':
            self.energy = self.energy_std
        elif self.former_layer_type == 'conv1D':
            self.energy = self.energy_from_conv1D
        elif self.former_layer_type == 'dense':
            self.energy = self.energy_from_dense

    def energy_std(self, y1, params, y2):
        """Compute 2-D conv XY energy for a conv2D → conv2D connection.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_ch, H_in, W_in).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # e^{-iθ_1}: input phasors; complex conv gives the interaction field
        ny2 = conv_func(jnp.exp(-1j * y1), F)
        # -Re[ e^{iθ_2} · field ] recovers the XY coupling
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0 + E1

    def energy_from_dense(self, y1, params, y2):
        """Compute 2-D conv XY energy when predecessor is a dense layer.

        Adds a singleton channel and W dimension: (L,) → (1, L, 1).

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Reshape to (1, L, 1) to match (channels, H, W) convention
        ny1 = y1[None, :, None]
        return self.energy_std(ny1, params, y2)

    def energy_from_conv1D(self, y1, params, y2):
        """Compute 2-D conv XY energy when predecessor is a 1-D conv layer.

        Adds a trailing singleton W dimension: (ch, L) → (ch, L, 1).

        Args:
            y1     (jnp.ndarray): 1-D feature map, shape (in_ch, L).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Promote 1-D spatial extent to 2-D by appending a size-1 W axis
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)

    '''
    def energy_from_conv1D(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]

        ny1 = y1[..., None]
        ny2 = conv_func(jnp.exp(-1j*ny1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1

    def energy_from_dense(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        ny1 = y1[None,...,None]
        ny2 = conv_func(jnp.exp(-1j*ny1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1
    '''


# =============================================================================
#  2-D Pooling layer (no trainable parameters)
# =============================================================================

class Pool2D(Conv2D):
    """2-D average-pooling layer for the XY model.

    Analogous to Pool1D but for 2-D spatial feature maps.  The fixed kernel
    is a uniform average over the [kH, kW] pooling window:
        kernel[h, w] = 1 / (kH * kW)

    No trainable parameters; get_layer_ratio returns a dummy dict.
    """

    def get_init_params(self, rng):
        """Construct the fixed 2-D averaging kernel; return dummy param dict.

        Args:
            rng: JAX PRNG key (unused).

        Returns:
            dict: {'kernel': 0.}
        """
        # Uniform average over the kH × kW window
        self.kernel = jnp.ones([*self.filter_shape], dtype=jnp.complex64) / (self.filter_shape[0] * self.filter_shape[1])
        return {'kernel': 0.}

    def single_pool_func(self, y):
        """Apply the 2-D averaging kernel to a single-channel spatial map.

        Args:
            y (jnp.ndarray): 2-D array, shape (H, W).

        Returns:
            jnp.ndarray: Pooled output, shape (1, H_out, W_out).
        """
        return jax.lax.conv(y[None, None, ...], self.kernel[None, None, ...], window_strides=self.strides, padding='SAME')[0, ...]

    def pool_func(self, y):
        """Apply the averaging kernel independently to each channel of y.

        Args:
            y (jnp.ndarray): Multi-channel map, shape (channels, H, W).

        Returns:
            jnp.ndarray: Pooled output, shape (channels, H_out, W_out).
        """
        return jax.vmap(self.single_pool_func, 0)(y)

    def energy_std(self, y1, params, y2):
        """Pool XY energy for a conv2D predecessor.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_ch, H_in, W_in).
            params (dict):        Ignored (no trainable params).
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar coupling energy (no bias term).
        """
        ny2 = self.pool_func(jnp.exp(-1j * y1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def energy_from_dense(self, y1, params, y2):
        """Pool XY energy for a dense predecessor.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Insert channel dim: (L,) → (1, L)
        ny1 = y1[None, :]
        ny2 = self.pool_func(jnp.exp(-1j * ny1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def energy_from_conv1D(self, y1, params, y2):
        """Pool XY energy for a conv1D predecessor.

        Args:
            y1     (jnp.ndarray): 1-D feature map, shape (in_ch, L).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Flatten spatial rows into a single axis per channel
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        # NOTE: self.kernel arg passed here is redundant — pool_func uses self.kernel internally
        ny2 = self.pool_func(jnp.exp(-1j * ny1), self.kernel)
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def get_layer_ratio(self, params, g_params):
        """Return a dummy ratio (no trainable parameters).

        Returns:
            dict: {'kernel': 0.}
        """
        return {'kernel': 0.}


# =============================================================================
#  Dense layer with optional intra-layer (within-layer) coupling
# =============================================================================

class Intra_connected(Denselayer):
    """Dense layer that optionally adds coupling *within* the layer itself.

    In a standard EP network layers interact only with adjacent layers.  This
    class allows neurons in the same layer to interact with each other via a
    separate weight matrix W_in, enabling e.g. Hopfield-like dynamics.

    Attributes:
        ic_type (str): One of 'None', 'full', 'graph', 'layer', 'lattice'.
        ic_args:       Extra arguments needed for the chosen intra-connection
                       topology (e.g. an adjacency matrix for 'graph').
    """

    def __init__(self, output_size, coup_func, bias_func, layer_type='dense', ic_type='None', ic_args=None):
        """Initialise an intra-connected dense layer.

        Args:
            output_size (int):      Number of neurons.
            coup_func   (callable): Pairwise coupling function.
            bias_func   (callable): Bias energy function.
            layer_type  (str):      Type tag (default 'dense').
            ic_type     (str):      Intra-connection type: 'None', 'full',
                                    'graph', 'layer', or 'lattice'.
            ic_args:                Topology descriptor (e.g. adjacency array)
                                    required for 'graph' ic_type.
        """
        super().__init__(output_size, coup_func, bias_func, layer_type)
        '''
        ic_type refers to "intra connection type". It can be either 'None', 'full', 'graph', 'layer' or 'lattice'.
        '''
        self.ic_type = ic_type
        self.ic_args = ic_args

    def get_init_params(self, rng):
        """Initialise inter-layer weights plus optional intra-layer weights.

        Args:
            rng: JAX PRNG key.

        Returns:
            dict: Parameter dict whose keys depend on ic_type:
                  'None'  — {'weights': W, 'bias field': [h, ψ]}
                  'full'  — adds 'intra coupling': W_in (output_size × output_size)
                  'graph' — adds 'intra coupling': W_in (one weight per graph edge)
        """
        W = jax.random.normal(rng, [self.input_size, self.output_size]) / jnp.sqrt(self.input_size + self.output_size)
        h = jnp.zeros(self.output_size)
        psi = jax.random.uniform(rng, shape=[self.output_size], minval=-jnp.pi, maxval=jnp.pi)

        if self.ic_type == 'None':
            # No intra-layer coupling; standard dense parameters only
            return {'weights': W, 'bias field': [h, psi]}
        elif self.ic_type == 'full':
            # Full all-to-all intra-layer weight matrix
            W_in = jax.random.normal(rng, shape=[self.output_size, self.output_size]) / jnp.sqrt(self.output_size)
            return {'weight': W, 'bias field': [h, psi], 'intra coupling': W_in}
        elif self.ic_type == 'graph':
            # One scalar weight per edge in the graph; ic_args holds the edge list
            W_in = jax.random.normal(rng, shape=[self.ic_args.shape[0]]) / jnp.sqrt(self.output_size)
            return {'weight': W, 'bias field': [h, psi], 'intra coupling': W_in}


# =============================================================================
#  Transposed Convolution and Upsample Layers
# =============================================================================

class TransConv1D(Conv1D):
    """1-D transposed convolutional (fractionally-strided) XY-model layer.

    Used in decoder / generative parts of the network to increase spatial
    resolution.  The energy is defined analogously to Conv1D but with the
    roles of input and output swapped so that the transpose conv is the
    natural operation:

        E = -Re[ e^{i·y1} · conv(e^{-i·y2}, flip(F^T)) ]

    where F^T denotes the kernel with input/output channels transposed and
    flip reverses the spatial axis, which implements the mathematical
    transpose of the forward convolution.
    """

    @staticmethod
    def transposed_conv(y, kernel, strides, padding='SAME'):
        """Perform a 1-D transposed convolution using jax.lax.conv_transpose.

        jax.lax.conv_transpose expects the layout
          input:  (batch, spatial, channels)
          kernel: (spatial, in_channels, out_channels)
        so we transpose before and after the call.

        Args:
            y       (jnp.ndarray): Input tensor, shape (batch, in_channels, L).
            kernel  (jnp.ndarray): Kernel, shape (out_channels, in_channels, K).
            strides (tuple):       Transposed conv strides.
            padding (str):         Padding mode (default 'SAME').

        Returns:
            jnp.ndarray: Output tensor, shape (batch, out_channels, L_out).
        """
        # Rearrange to (batch, L, in_channels) for conv_transpose
        y_run = y.transpose([0, 2, 1])
        # Rearrange kernel to (K, in_channels, out_channels) for conv_transpose
        kernel_run = kernel.transpose([2, 1, 0])

        # Perform the transposed convolution; output: (batch, L_out, out_channels)
        output = jax.lax.conv_transpose(y_run, kernel_run, strides=strides, padding=padding)
        # Rearrange back to (batch, out_channels, L_out)
        return output.transpose([0, 2, 1])

    def get_layer_size(self, input_data, former_layer_type):
        """Determine output size by running a dummy transposed convolution.

        Args:
            input_data       (jnp.ndarray): Sample from the preceding layer.
            former_layer_type (str):        Type tag of the preceding layer.

        Returns:
            tuple: (sample_output, n_nodes, structure_string)
        """
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            self.input_size = input_data.shape[0]
        elif former_layer_type == 'conv1D':
            self.input_channel, self.input_size = input_data.shape
        elif former_layer_type == 'conv2D':
            self.input_channel, self.input_size = input_data.shape[0], input_data.shape[1] * input_data.shape[2]

        y = np.zeros([self.input_channel, self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, self.filter_shape])
        # Run dummy transposed conv to discover the upsampled output size
        y = self.transposed_conv(y[None, ...], kernel, strides=self.strides, padding='SAME')
        self.output_size = y.shape[-1]
        print(y.shape)

        return y[0], self.output_channel * self.output_size, '{0}*{1}'.format(self.output_channel, self.output_size)

    def energy_std(self, y1, params, y2):
        """Transposed-conv XY energy for a conv1D predecessor.

        The trick is to express the transposed convolution energy in terms of a
        *forward* convolution on y2 using the flipped/transposed kernel:

            E = -Re[ e^{i·y1} · conv(e^{-i·y2}, flip(F^T, spatial)) ]

        This identity follows from the adjoint relationship between conv and
        conv_transpose and ensures that the grad w.r.t. y1 recovers the
        standard force formula.

        Args:
            y1     (jnp.ndarray): Previous-layer angles, shape (in_ch, L_in).
            params (dict):        {'kernel': F, 'bias field': bias}.
            y2     (jnp.ndarray): This-layer angles, shape (out_ch, L_out).

        Returns:
            float: Scalar energy.
        """
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Transpose channels and flip spatial axis to get the adjoint kernel
        # F shape: (out_ch, in_ch, K) → transposed: (in_ch, out_ch, K) → flip K axis
        ny2 = conv_func(jnp.exp(-1j * y2), jnp.flip(F.transpose([1, 0, 2]), axis=(2)))
        # Energy: -Re[ e^{i·y1} · (adjoint conv result) ]
        E0 = -jnp.real(jnp.exp(1j * y1) * ny2).sum()
        return E0 + E1

    def energy_from_conv2D(self, y1, params, y2):
        """TransConv XY energy when the predecessor is a 2-D conv layer.

        Args:
            y1     (jnp.ndarray): 2-D feature map, shape (in_ch, H, W).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, L_out).

        Returns:
            float: Scalar energy.
        """
        # Add trailing W dimension: (in_ch, H, W) → used as (in_ch, H*W, 1)
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)

    def energy_from_dense(self, y1, params, y2):
        """TransConv XY energy when the predecessor is a dense layer.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, L_out).

        Returns:
            float: Scalar energy.
        """
        # Add channel and W dimensions: (L,) → (1, L, 1)
        ny1 = y1[None, ..., None]
        return self.energy_std(ny1, params, y2)


class TransConv2D(Conv2D):
    """2-D transposed convolutional XY-model layer.

    2-D version of TransConv1D.  Increases spatial resolution in both H and W.
    The energy uses the adjoint of the 2-D forward convolution:

        E = -Re[ e^{i·y1} · conv2d(e^{-i·y2}, flip(F^T, (H, W))) ]
    """

    @staticmethod
    def transposed_conv(y, kernel, strides, padding='SAME'):
        """Perform a 2-D transposed convolution using jax.lax.conv_transpose.

        Args:
            y       (jnp.ndarray): Shape (batch, in_channels, H, W).
            kernel  (jnp.ndarray): Shape (out_channels, in_channels, kH, kW).
            strides (tuple):       Strides [sH, sW].
            padding (str):         Padding mode.

        Returns:
            jnp.ndarray: Shape (batch, out_channels, H_out, W_out).
        """
        # Rearrange to (batch, H, W, in_channels) for conv_transpose
        y_run = y.transpose([0, 2, 3, 1])
        # Rearrange kernel to (kH, kW, in_channels, out_channels)
        kernel_run = kernel.transpose([2, 3, 1, 0])

        # Perform the 2-D transposed convolution; output: (batch, H_out, W_out, out_channels)
        output = jax.lax.conv_transpose(y_run, kernel_run, strides=strides, padding=padding)
        # Rearrange back to (batch, out_channels, H_out, W_out)
        return output.transpose([0, 3, 1, 2])

    def get_layer_size(self, input_data, former_layer_type):
        """Determine output shape via a dummy 2-D transposed convolution.

        Args:
            input_data       (jnp.ndarray): Sample from the preceding layer.
            former_layer_type (str):        Type tag of the preceding layer.

        Returns:
            tuple: (sample_output, n_nodes, structure_string)
        """
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            self.input_size = [input_data.shape[0], 1]
        elif former_layer_type == 'conv1D':
            self.input_channel = input_data.shape[0]
            self.input_size = [input_data.shape[1], 1]
        elif former_layer_type == 'conv2D':
            self.input_channel = input_data.shape[0]
            self.input_size = [input_data.shape[1], input_data.shape[2]]

        y = np.zeros([self.input_channel, *self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, *self.filter_shape])
        y = self.transposed_conv(y[None, ...], kernel, strides=self.strides, padding='SAME')
        self.output_size = [y.shape[-2], y.shape[-1]]

        return y[0], self.output_channel * self.output_size[0] * self.output_size[1], '{0}*{1}*{2}'.format(self.output_channel, *self.output_size)

    def energy_std(self, y1, params, y2):
        """2-D transposed-conv XY energy for a conv2D predecessor.

        Uses the same adjoint-kernel trick as TransConv1D:
            flip(F^T, (H, W)) transposes in/out channels and flips both spatial
            axes to implement the exact adjoint of the 2-D forward convolution.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_ch, H_in, W_in).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))

        def conv_func(y, kernel):
            return jax.lax.conv(y[None, ...], kernel, window_strides=self.strides, padding='SAME')[0, ...]

        # Transpose (out_ch, in_ch, kH, kW) → (in_ch, out_ch, kH, kW)
        # then flip both spatial axes to form the adjoint kernel
        ny2 = conv_func(jnp.exp(-1j * y2), jnp.flip(F.transpose([1, 0, 2, 3]), axis=(2, 3)))
        E0 = -jnp.real(jnp.exp(1j * y1) * ny2).sum()
        return E0 + E1

    def energy_from_conv1D(self, y1, params, y2):
        """2-D transposed-conv XY energy for a conv1D predecessor.

        Args:
            y1     (jnp.ndarray): 1-D feature map, shape (in_ch, L).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Append trailing W=1 dimension: (in_ch, L) → (in_ch, L, 1)
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)

    def energy_from_dense(self, y1, params, y2):
        """2-D transposed-conv XY energy for a dense predecessor.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Layer parameters.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # Reshape to (1, L, 1): add channel and W dims
        ny1 = y1[None, ..., None]
        return self.energy_std(ny1, params, y2)


# =============================================================================
#  1-D Upsample layer (no trainable parameters)
# =============================================================================

class Unsample1D(TransConv1D):
    """1-D upsampling layer using a fixed all-ones transposed-conv kernel.

    Analogous to Pool1D (no trainable parameters) but for the decoder path.
    The fixed kernel of all ones spreads each input activation across
    filter_shape output positions, effectively repeating values.

    No bias field; get_layer_ratio returns a dummy dict.
    """

    def get_init_params(self, rng):
        """Construct the fixed all-ones upsample kernel.

        Args:
            rng: JAX PRNG key (unused).

        Returns:
            dict: {'kernel': 0.}
        """
        # All-ones kernel: each input phasor contributes equally to filter_shape outputs
        self.kernel = jnp.ones([self.filter_shape], dtype=jnp.complex64)
        return {'kernel': 0.}

    def single_unsample_func(self, y):
        """Upsample a single-channel 1-D signal via transposed convolution.

        Args:
            y (jnp.ndarray): 1-D signal, shape (L,).

        Returns:
            jnp.ndarray: Upsampled signal, shape (1, L_out).
        """
        def transpose_conv(y, kernel, padding='SAME'):
            """Inner transposed-conv helper operating on (batch, channels, L)."""
            # Rearrange to (batch, L, channels) for conv_transpose
            y_run = y.transpose([0, 2, 1])
            # Rearrange kernel to (K, in_ch, out_ch)
            kernel_run = kernel.transpose([2, 1, 0])
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=self.strides, padding=padding, rhs_dilation=None)
            return output.transpose([0, 2, 1])

        # Add batch and channel dims: (L,) → (1, 1, L)
        y = y[None, None, ...]
        # Apply transposed conv with fixed kernel; remove batch/channel dims
        return transpose_conv(y, self.kernel[None, None, ...])[0, 0, ...]

    def unsample_func(self, y):
        """Apply upsampling independently to each channel.

        Args:
            y (jnp.ndarray): Multi-channel 1-D signal, shape (channels, L).

        Returns:
            jnp.ndarray: Upsampled output, shape (channels, L_out).
        """
        return jax.vmap(self.single_unsample_func, (0))(y)

    def energy_std(self, y1, params, y2):
        """Upsample XY energy for a conv1D predecessor.

        No bias term (no trainable parameters).

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_ch, L_in).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, L_out).

        Returns:
            float: Scalar coupling energy.
        """
        # Upsample the input phasors with the fixed transposed-conv kernel
        ny2 = self.unsample_func(jnp.exp(-1j * y1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def energy_from_dense(self, y1, params, y2):
        """Upsample XY energy when the predecessor is a dense layer.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, L_out).

        Returns:
            float: Scalar energy.
        """
        # Add channel and W dims: (L,) → (1, L, 1)
        ny1 = y1[None, :, None]
        return self.energy_std(ny1, params, y2)

    def energy_from_conv1D(self, y1, params, y2):
        """Upsample XY energy when predecessor is a 1-D conv layer.

        Args:
            y1     (jnp.ndarray): Feature map, shape (in_ch, L).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, L_out).

        Returns:
            float: Scalar energy.
        """
        # Add trailing W=1: (in_ch, L) → (in_ch, L, 1)
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)

    def get_layer_ratio(self, params, g_params):
        """Return a dummy ratio (no trainable parameters).

        Returns:
            dict: {'kernel': 0.}
        """
        return {'kernel': 0.}


# =============================================================================
#  2-D Upsample layer (no trainable parameters)
# =============================================================================

class Unsample2D(TransConv2D):
    """2-D upsampling layer using a fixed all-ones transposed-conv kernel.

    2-D analogue of Unsample1D.  Spreads each input phasor across a
    filter_shape[0] × filter_shape[1] output patch.

    No bias field; get_layer_ratio returns a dummy dict.
    """

    def get_init_params(self, rng):
        """Construct the fixed 2-D all-ones upsample kernel.

        Args:
            rng: JAX PRNG key (unused).

        Returns:
            dict: {'kernel': 0.}
        """
        self.kernel = jnp.ones([*self.filter_shape], dtype=jnp.complex64)
        return {'kernel': 0.}

    def single_unsample_func(self, y):
        """Upsample a single-channel 2-D spatial map via transposed convolution.

        Args:
            y (jnp.ndarray): 2-D map, shape (H, W).

        Returns:
            jnp.ndarray: Upsampled output, shape (1, H_out, W_out).
        """
        def transpose_conv(y, kernel, padding='SAME'):
            """Inner 2-D transposed-conv helper operating on (batch, ch, H, W)."""
            # Rearrange to (batch, H, W, channels) for conv_transpose
            y_run = y.transpose([0, 2, 3, 1])
            # Rearrange kernel to (kH, kW, in_ch, out_ch)
            kernel_run = kernel.transpose([2, 3, 1, 0])
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=self.strides, padding=padding, rhs_dilation=None)
            return output.transpose([0, 3, 1, 2])

        # Add batch and channel dims: (H, W) → (1, 1, H, W)
        y = y[None, None, ...]
        return transpose_conv(y, self.kernel[None, None, ...])[0, 0, ...]

    def unsample_func(self, y):
        """Apply 2-D upsampling independently to each channel.

        Args:
            y (jnp.ndarray): Multi-channel map, shape (channels, H, W).

        Returns:
            jnp.ndarray: Upsampled output, shape (channels, H_out, W_out).
        """
        return jax.vmap(self.single_unsample_func, (0))(y)

    def energy_std(self, y1, params, y2):
        """Upsample XY energy for a conv2D predecessor.

        Args:
            y1     (jnp.ndarray): Input angles, shape (in_ch, H_in, W_in).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar coupling energy.
        """
        ny2 = self.unsample_func(jnp.exp(-1j * y1))
        E0 = -jnp.real(jnp.exp(1j * y2) * ny2).sum()
        return E0

    def energy_from_dense(self, y1, params, y2):
        """Upsample XY energy when predecessor is a dense layer.

        Args:
            y1     (jnp.ndarray): Dense output, shape (L,).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # (L,) → (1, L, 1)
        ny1 = y1[None, :, None]
        return self.energy_std(ny1, params, y2)

    def energy_from_conv1D(self, y1, params, y2):
        """Upsample XY energy when predecessor is a 1-D conv layer.

        Args:
            y1     (jnp.ndarray): Feature map, shape (in_ch, L).
            params (dict):        Ignored.
            y2     (jnp.ndarray): Output angles, shape (out_ch, H_out, W_out).

        Returns:
            float: Scalar energy.
        """
        # (in_ch, L) → (in_ch, L, 1)
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)

    def get_layer_ratio(self, params, g_params):
        """Return a dummy ratio (no trainable parameters).

        Returns:
            dict: {'kernel': 0.}
        """
        return {'kernel': 0.}


# =============================================================================
#  Network base class
# =============================================================================

class Network:
    """Abstract base class for all XY-model EP networks.

    Defines the interface that every concrete network must implement:
        get_initial_params  — randomly initialise all trainable parameters.
        get_initial_state   — create random initial phase angles for all layers.
        internal_energy     — compute the total XY coupling energy of the network.
        distance_function   — measure distance between network output and target.
        external_energy     — EP nudging energy (defaults to distance_function).
        internal_force      — gradient of internal energy w.r.t. all state angles.
        external_force      — gradient of external energy w.r.t. output angles.
        params_derivative   — gradient of internal energy w.r.t. all parameters
                              (used to compute EP parameter updates).

    In EP training:
      - Free phase: integrate ẏ = F_int(y, θ) → equilibrium y*.
      - Nudged phase: integrate ẏ = F_int(y, θ) + β·F_ext(y, t, θ) → nudged eq. ŷ.
      - Parameter update: ΔW ∝ ∂E/∂W|_{y=ŷ} − ∂E/∂W|_{y=y*}  (contrastive Hebbian).
    """

    def __init__(self) -> None:
        pass

    def get_initial_params(self):
        """Randomly initialise all trainable parameters in the network."""
        pass

    def get_initial_state(input_data):
        """Create random initial phase angles given a batch of input data."""
        pass

    def internal_energy(self, y, network_params):
        """Compute the total XY internal energy summed over all layers.

        Args:
            y              (dict): Network state, keyed by layer name.
            network_params (dict): Trainable parameters, keyed by layer name.

        Returns:
            float: Scalar total internal energy.
        """
        pass

    def distance_function(self, y, target, network_params):
        """Measure the distance between output angles and target.

        Args:
            y              (jnp.ndarray): Output-layer angles.
            target         (jnp.ndarray): Target angles.
            network_params (dict):        Trainable parameters (may be unused).

        Returns:
            float: Scalar distance (e.g. 1 - cos(y - target)).
        """
        pass

    def external_energy(self, y, target, network_params):
        """Compute the external (nudging) energy.

        Defaults to the distance function; subclasses may override.

        Args:
            y              (jnp.ndarray): Output-layer angles.
            target         (jnp.ndarray): Target angles.
            network_params (dict):        Trainable parameters.

        Returns:
            float: Scalar external energy.
        """
        return self.distance_function(y, target, network_params)

    def internal_force(self, y, network_params):
        """Compute force on every neuron from the internal energy.

        F_int = -∂E_int/∂y

        Args:
            y              (dict): Network state.
            network_params (dict): Trainable parameters.

        Returns:
            dict: Same structure as y; each entry is the force on that layer.
        """
        return -jax.grad(self.internal_energy, argnums=0)(y, network_params)

    def external_force(self, y, target, network_params):
        """Compute force on the output layer from the external (nudging) energy.

        F_ext = -∂E_ext/∂y_out

        Args:
            y              (jnp.ndarray): Output-layer angles.
            target         (jnp.ndarray): Target angles.
            network_params (dict):        Trainable parameters.

        Returns:
            jnp.ndarray: Force on the output layer.
        """
        return -jax.grad(self.external_energy, argnums=0)(y, target, network_params)

    def params_derivative(self, y, network_params):
        """Compute ∂E_int/∂θ for all parameters (used in the EP update rule).

        Args:
            y              (dict): Network state (e.g. free-phase equilibrium).
            network_params (dict): Current trainable parameters.

        Returns:
            dict: Same structure as network_params; each entry is the gradient.
        """
        d_energy = jax.grad(self.internal_energy, argnums=1)
        return d_energy(y, network_params)

    def get_random_index(self, N_data, batch_size):
        """Sample a random mini-batch index array.

        Args:
            N_data     (int): Total dataset size.
            batch_size (int): Mini-batch size.

        Returns:
            np.ndarray: 1-D integer array of sampled indices.
        """
        if batch_size == N_data:
            # Return all indices when the batch covers the full dataset
            return np.arange(0, N_data, dtype=np.int32)
        return np.random.randint(0, N_data, batch_size)


# =============================================================================
#  Module — general EP network container
# =============================================================================

class Module(Network):
    """General-purpose XY-model EP network built from a list of Layer objects.

    Users subclass Module and override setup() to define the layer stack.
    All EP training machinery (ODE thermalisation, pmap multi-GPU support,
    learning-rate normalisation) is provided here.

    Key attributes set after get_initial_params() is called:
        layers      (dict): Maps layer names → Layer instances.
        params      (dict): Maps layer names → parameter dicts.
        layer_order (list): Ordered list of layer names (defines forward pass).
        output_name (str):  Name of the output layer.
        F0          (dict): Zero-force template (same structure as state dict).
        N           (int):  Total number of neurons across all layers.
        N_list      (list): Number of neurons per layer (including input).
        structure   (list): Human-readable shape strings per layer.

    Thermalisation strategy (selected automatically in get_initial_params):
        ODE single-GPU : thermalize_network_ode   (Tsit5, diffrax)
        ODE multi-GPU  : pmap_thermalize_ode
        OPT single-GPU : thermalize_network_opt   (optax optimizer)
        OPT multi-GPU  : pmap_thermalize_opt
    """

    def __init__(self, cost_func, run_params=(100, 1e-3, 1e-6), opt_params=(1e-5, 10000), optimizer=None, network_type='general XY', structure_name='dnn'):
        """Initialise a Module.

        Args:
            cost_func      (callable): External cost function f(y_out, target)
                                       used to compute the nudging energy.
            run_params     (tuple):    (runtime, rtol, atol) — ODE integration
                                       time-span and tolerances for diffrax.
            opt_params     (tuple):    (tol, maxtime) — convergence tolerance
                                       and max iterations for optax thermalisation.
            optimizer:                 Optional optax optimizer for finding
                                       equilibria by gradient descent instead of
                                       ODE integration.  None → use ODE.
            network_type   (str):      Descriptive tag (default 'general XY').
            structure_name (str):      Descriptive tag (default 'dnn').
        """
        # ODE solver parameters
        self.runtime, self.rtol, self.atol = run_params
        # Optax-based equilibrium finder parameters
        self.tol, self.maxtime = opt_params
        self.network_type = network_type
        self.structure_name = structure_name

        # These are populated by setup() and get_initial_params()
        self.layer_order = []
        self.layers = {}
        self.params = {}

        self.cost_func = cost_func
        self.optimizer = optimizer

    def get_variable_name(self, variable, scope=locals()):
        """Retrieve the attribute name of a given variable by identity.

        Searches both the local scope and the instance __dict__.

        Args:
            variable: The object whose name is sought.
            scope (dict): Local variable scope to search first.

        Returns:
            str or None: The attribute name, or None if not found.
        """
        for name, value in scope.items():
            if value is variable:
                return name

        for name, value in self.__dict__.items():
            if value is variable:
                return name

    def setup(self):
        """Define the network architecture (override in subclasses).

        The default stub sets a single dummy layer.  Concrete subclasses
        should set:
          self.input_type  — type string of the first layer's input.
          self.<name>      — Layer instances.
          self.layer_order — list of attribute names in forward order.
          self.output_name — attribute name of the output layer.
        """
        # Here the user needs to clarify input_type, layers, name of output layer and the order of the layers
        self.input_type = None
        self.l1 = None
        self.layer_order = ['l1']
        self.output_name = self.layer_order[-1]

    def show_hyperparams(self):
        """Print a summary of network structure and hyperparameters."""
        print("Structure: ", self.structure)
        print("Number of Nodes: ", self.N_list)
        print("Total Number of Nodes: ", self.N)
        if self.optimizer is None:
            print("Optimizer: ODE")
        else:
            print("Optimizer: OPT")

    def get_initial_params(self, rng, input_data):
        """Initialise all layer parameters and wire up the thermalisation function.

        Steps:
          1. Call setup() to instantiate the layer objects.
          2. Select the thermalisation strategy based on optimizer and device count.
          3. Discover all Layer instances from instance attributes.
          4. Iterate through layer_order, calling get_layer_size → get_init_params
             → setup() for each layer in sequence.
          5. Build the zero-force template F0, node count N, and structure list.

        Args:
            rng        (jax.random.PRNGKey): Random seed.
            input_data (jnp.ndarray):        One sample input (shape only matters).

        Returns:
            dict: Nested parameter dict keyed by layer name.
        """
        self.setup()
        N_devices = len(jax.devices())
        # Select thermalisation method: ODE vs optax, single-GPU vs pmap
        if self.optimizer is None:
            if N_devices > 1:
                self.thermalize_network = self.pmap_thermalize_ode
            else:
                self.thermalize_network = self.thermalize_network_ode
        else:
            if N_devices > 1:
                self.thermalize_network = self.pmap_thermalize_opt
            else:
                self.thermalize_network = self.thermalize_network_opt

        # Discover all Layer instances from instance attributes
        for name, value in self.__dict__.items():
            if hasattr(value, 'layer_type'):
                self.layers.update({name: value})

        def prod(xl):
            """Compute the product of a list of integers."""
            p = 1
            for k in range(0, len(xl)):
                p = p * xl[k]
            return p

        def to_str(xl):
            """Convert a shape tuple to a human-readable '*'-separated string."""
            y = '{0}'.format(xl[0])
            for k in range(1, len(xl)):
                y = y + '*{0}'.format(xl[k])
            return y

        input_shape = input_data.shape
        input_size = prod(input_shape)
        # F0 is a zero-valued copy of the state dict used as a force accumulator template
        self.F0 = {'input_data': 0 * input_data}
        self.N = input_size
        self.N_list = [input_size]
        self.structure = [to_str(input_shape)]
        former_layer_type = self.input_type
        for name in self.layer_order:
            # get_layer_size records input shape and returns the output shape
            input_data, N_nodes, layer_structure = self.layers[name].get_layer_size(input_data, former_layer_type)
            # get_init_params creates the trainable parameter tensors
            self.params.update({name: self.layers[name].get_init_params(rng)})
            # setup() wires the correct energy/force function pointers
            self.layers[name].setup()
            self.N = self.N + N_nodes
            self.N_list.append(N_nodes)
            self.structure.append(layer_structure)
            # Pre-allocate a zero entry in F0 matching this layer's output shape
            self.F0.update({name: 0 * input_data})
            former_layer_type = self.layers[name].layer_type
        return self.params

    def get_initial_state(self, input_data):
        """Create a random initial phase state for a batch of inputs.

        The input angles are clamped to the given input_data values; all hidden
        and output layer angles are drawn uniformly from (-π, π).

        Args:
            input_data (jnp.ndarray): Batch of input data, shape (N_data, …).

        Returns:
            dict: State dict keyed by layer name (including 'input_data').
        """
        initial_state = {'input_data': input_data}
        N_data = input_data.shape[0]
        for name in self.layer_order:
            current_layer = self.layers[name]
            state = current_layer.get_init_state(N_data)
            initial_state.update({name: state})
        return initial_state

    @partial(jax.jit, static_argnames=['self'])
    def internal_force(self, y, params):
        """Compute the internal force on every layer neuron.

        Iterates through layers in order, accumulating:
          - FF (forward force) on the current layer from the layer below.
          - BF (backward force) fed back to the previous layer.

        The input layer ('input_data') never receives a backward force
        (the `(last_layer_name != 'input_data')` mask handles this).

        Args:
            y      (dict): Network state keyed by layer name.
            params (dict): Trainable parameters keyed by layer name.

        Returns:
            dict: Force dict with the same structure as y.
        """
        y1 = y['input_data']
        # Start with all-zero forces (copy of the pre-allocated template)
        F = self.F0.copy()
        last_layer_name = 'input_data'
        for layer_name in self.layer_order:
            current_layer = self.layers[layer_name]
            y2 = y[layer_name]
            # FF: force on y2 from y1;  BF: force fed back to y1
            FF, BF = current_layer.force(y1, params[layer_name], y2)
            # Accumulate BF on the previous layer; suppress for 'input_data'
            F[last_layer_name] += BF * (last_layer_name != 'input_data')
            # Accumulate FF on the current layer
            F[layer_name] += FF

            last_layer_name = layer_name
            y1 = y2
        '''
        F = {'input_data': 0.*y['input_data']}
        last_layer_name = 'input_data'
        for layer_name in self.layer_order:
            current_layer = self.layers[layer_name]
            y2 = y[layer_name]
            FF, BF = current_layer.force(y1, params[layer_name], y2)
            F[last_layer_name] += BF * (last_layer_name!='input_data')
            F.update({layer_name: FF})

            last_layer_name = layer_name
            y1 = y2
        '''
        return F

    @partial(jax.jit, static_argnames=['self'])
    def internal_energy(self, y, params):
        """Compute the total internal XY energy summed over all layer pairs.

        E_int = Σ_l  E_l(y^{l-1}, y^l, W^l)

        Args:
            y      (dict): Network state.
            params (dict): Trainable parameters.

        Returns:
            float: Scalar total internal energy.
        """
        y1 = y['input_data']
        E = 0.

        for layer_name in self.layer_order:
            layer = self.layers[layer_name]
            y2 = y[layer_name]
            # Accumulate the XY coupling energy for each consecutive layer pair
            E += layer.energy(y1, params[layer_name], y2)
            y1 = y2
        '''
        for layer_name, layer in self.__dict__.items():
            if hasattr(layer, 'layer_type'):
                y2 = y[layer_name]
                E += layer.energy(y1, params[layer_name], y2)
                y1 = y2
        '''
        return E

    @partial(jax.jit, static_argnames=['self'])
    def distance_function(self, y, target, params):
        """Compute the XY angular distance between y and target.

        Uses the standard XY metric: d = (1 - cos(y - target)) / 2
        which equals 0 when y = target and 1 when y = target + π.

        Args:
            y      (jnp.ndarray): Output-layer angles.
            target (jnp.ndarray): Target angles (same shape as y).
            params (dict):        Unused (kept for API consistency).

        Returns:
            float: Scalar distance summed over all output neurons.
        """
        # y is only the output
        return jnp.sum(1 - jnp.cos(y - target)) / 2.

    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        """Compute the external nudging energy using the user-supplied cost function.

        Applies cost_func(y_i, target_i) independently to each data point in
        the batch, then sums the results.

        Args:
            y      (jnp.ndarray): Output-layer angles, shape (N_data, …).
            target (jnp.ndarray): Target angles, same shape as y.

        Returns:
            float: Scalar total external energy over the batch.
        """
        # y is only the output
        return jnp.sum(jax.vmap(self.cost_func, (0, 0))(y, target))

    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        """Compute the force on the output layer from the external energy.

        F_ext = -∂E_ext/∂y_out

        Args:
            y      (jnp.ndarray): Output-layer angles.
            target (jnp.ndarray): Target angles.
            params (dict):        Unused (kept for API consistency).

        Returns:
            jnp.ndarray: Force on the output layer, same shape as y.
        """
        return -jax.grad(self.external_energy, argnums=0)(y, target)

    def convert_to_lnn_params(self, input_data, params):
        """Convert structured params to a flat list suitable for LXY-NN.

        Args:
            input_data (jnp.ndarray): Sample input (used to build bias offset).
            params     (dict):        Nested parameter dict.

        Returns:
            tuple: (WL, biasL)
                WL     — list of weight matrices per layer.
                biasL  — concatenated bias array over all layers (including input).
        """
        WL = []
        biasL = [np.zeros([2, input_data.shape[1]])]
        for layer_name in self.layer_order:
            WL.append(params[layer_name]['weights'])
            biasL.append(np.asarray(params[layer_name]['bias field']))

        return WL, jnp.concatenate(biasL, axis=1)

    def merge_y(self, y):
        """Concatenate all layer states into a single array along the neuron axis.

        Args:
            y (dict): State dict keyed by layer name.

        Returns:
            jnp.ndarray: Concatenated angles, shape (N_data, N_total).
        """
        yl = []
        for layer_name, layer in y.items():
            yl.append(y[layer_name])

        return jnp.concatenate(yl, axis=1)

    # =========================================================================
    #  ODE-based thermalisation (single GPU)
    # =========================================================================

    @partial(jax.jit, static_argnames=['self'])
    def free_force(self, t, y0, params):
        """Return the internal force for the free-phase ODE (β = 0).

        The free-phase ODE is:  ẏ = F_int(y, θ)

        Args:
            t      (float): Current ODE time (unused; autonomous system).
            y0     (dict):  Current network state.
            params (dict):  Trainable parameters.

        Returns:
            dict: Force dict matching the structure of y0.
        """
        return self.internal_force(y0, params)

    def single_free_run(self, y0, params):
        """Thermalise a single data point under the free-phase ODE.

        Integrates ẏ = F_int(y, θ) from t=0 to t=runtime using diffrax Tsit5
        (an adaptive-step Runge-Kutta 4/5 scheme).

        Args:
            y0     (dict): Initial state for this data point.
            params (dict): Trainable parameters.

        Returns:
            dict: Final equilibrium state (the last time-step of the solution).
        """
        t_span = [0, self.runtime]

        # Wrap the force function in the diffrax ODETerm interface
        odefunc = lambda t, y, args: self.free_force(t, y, params)
        eqs = diffrax.ODETerm(odefunc)

        # Tsit5: explicit adaptive Runge-Kutta of order 4(5), good for smooth ODEs
        solver = diffrax.Tsit5()

        # PIDController adapts the step size to meet rtol/atol tolerances
        stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0=None, y0=y0,
                                       stepsize_controller=stepsize_controller, max_steps=10000000)

        # solution.ys has a leading time axis of length 1; concatenate removes it
        return jax.tree_map(jnp.concatenate, solution.ys)

    @partial(jax.jit, static_argnames=['self'])
    def apply_to(self, input_data, params):
        """Run the free-phase thermalisation on a full batch (inference).

        Args:
            input_data (jnp.ndarray): Batch of input data.
            params     (dict):        Trainable parameters.

        Returns:
            dict: Equilibrium state for each data point in the batch.
        """
        y0 = self.get_initial_state(input_data)
        # vmap over the batch axis (axis 0 of y0 and axis 0 of each leaf)
        return jax.vmap(self.single_free_run, (0, None))(y0, params)

    # =========================================================================
    #  ODE-based thermalisation for EP training
    # =========================================================================

    @partial(jax.jit, static_argnames=['self'])
    def total_force(self, t, y, target, beta, params):
        """Compute the total (internal + nudged external) force for a single sample.

        During the nudged phase of EP, the ODE is:
            ẏ = F_int(y, θ) + β · F_ext(y_out, target, θ)

        Only the output layer receives the external force; all other layers
        are driven only by the internal XY interactions.

        Args:
            t      (float):         Current ODE time (unused; autonomous).
            y      (dict):          Current network state.
            target (jnp.ndarray):   Target for the output layer.
            beta   (float):         Nudging strength β.
            params (dict):          Trainable parameters.

        Returns:
            dict: Total force on every layer.
        """
        # Internal force from XY interactions across all layers
        F = self.internal_force(y, params)
        # Add β * external force only on the output-layer neurons
        F[self.output_name] += beta * self.external_force(y[self.output_name], target, params)
        return F

    def single_run_func(self, y0, target, beta, params):
        """Thermalise a single data point under the nudged-phase ODE.

        Integrates ẏ = F_int + β·F_ext until t = runtime.

        Args:
            y0     (dict):         Initial state for this sample.
            target (jnp.ndarray):  Target angles.
            beta   (float):        Nudging strength β.
            params (dict):         Trainable parameters.

        Returns:
            dict: Nudged equilibrium state.
        """
        t_span = [0, self.runtime]

        odefunc = lambda t, y0, args: self.total_force(t, y0, target, beta, params)
        eqs = diffrax.ODETerm(odefunc)

        # Tsit5 adaptive solver
        solver = diffrax.Tsit5()

        stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)

        # Solve the nudged ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0=None, y0=y0,
                                       stepsize_controller=stepsize_controller, max_steps=10000000)

        return jax.tree_map(jnp.concatenate, solution.ys)

    @partial(jax.jit, static_argnames=['self'])
    def thermalize_network_ode(self, y0, target, beta, params):
        """Thermalise a full batch using the ODE solver (single-GPU).

        Applies single_run_func to each (y0_i, target_i) pair via vmap.

        Args:
            y0     (dict):         Batch of initial states.
            target (jnp.ndarray):  Batch of targets, shape (N_data, …).
            beta   (float):        Nudging strength β.
            params (dict):         Trainable parameters.

        Returns:
            dict: Batch of equilibrium states.
        """
        run_func = lambda y0, target: self.single_run_func(y0, target, beta, params)
        # vmap vectorises over the data axis (axis 0) of y0 and target
        y = jax.vmap(run_func, (0, 0))(y0, target)
        return y

    @staticmethod
    def tree_expand(tree, n):
        """Replicate every leaf of a pytree n times along a new leading axis.

        Used to broadcast scalar params/beta to all devices in pmap.

        Args:
            tree: Any JAX pytree.
            n    (int): Number of replicas.

        Returns:
            Pytree with the same structure; each leaf has shape (n, …).
        """
        def leaf_expand(leaf):
            # tensordot with ones(n) along 0 contracted dimensions = outer product → shape (n, *leaf.shape)
            return jnp.tensordot(jnp.ones(n), leaf, 0)
        return jax.tree_map(leaf_expand, tree)

    @staticmethod
    def pad_data(y, target, N_devices):
        """Pad and reshape data so it can be split evenly across N_devices.

        If N_data is not divisible by N_devices, zero-pad to the next multiple
        and then reshape to (N_devices, N_per_device, …).

        Args:
            y         (dict):         Batch state dict, each leaf shape (N_data, …).
            target    (jnp.ndarray):  Target array, shape (N_data, …).
            N_devices (int):          Number of devices (GPUs).

        Returns:
            tuple: (y_reshaped, target_reshaped) with leading axis = N_devices.
        """
        def pad_func(y):
            """Zero-pad y along axis 0 to the next multiple of N_devices."""
            N_data = y.shape[0]
            # Padding needed on the first axis only
            pad_width = [(0, N_devices - N_data % N_devices)]
            for k in range(1, len(y.shape)):
                pad_width.append((0, 0))
            y_pad = jnp.pad(y, pad_width, mode='constant')
            return y_pad

        def reshape_func(y):
            """Reshape (N_data, …) to (N_devices, N_per_device, …)."""
            N_data = y.shape[0]
            N_rem = N_data % N_devices
            N_per = N_data // N_devices
            return y.reshape(N_devices, N_per + int(N_rem == 1), *y.shape[1:])

        N_data = target.shape[0]
        if N_data % N_devices == 0:
            # No padding needed; just reshape
            return jax.tree_map(reshape_func, y), reshape_func(target)
        else:
            # Pad first, then reshape
            y_pad = jax.tree_map(pad_func, y)
            target_pad = pad_func(target)
            return jax.tree_map(reshape_func, y_pad), reshape_func(target_pad)

    from functools import partial

    @partial(jax.jit, static_argnames=['self'])
    def pmap_thermalize_ode(self, y0, target, beta, params):
        """Thermalise a batch using ODE integration distributed across multiple GPUs.

        Data is split into N_devices shards; each GPU runs thermalize_network_ode
        on its shard in parallel via jax.pmap.  Padding ensures all shards have
        equal size; padding rows are discarded from the output.

        Args:
            y0     (dict):         Batch of initial states.
            target (jnp.ndarray):  Batch of targets.
            beta   (float):        Nudging strength β.
            params (dict):         Trainable parameters.

        Returns:
            dict: Equilibrium states for the original (un-padded) batch.
        """
        devices = jax.devices()
        N_devices = len(devices)

        # Split data across devices, padding if necessary
        y0_run, target_run = self.pad_data(y0, target, N_devices)
        # Replicate params and beta for each device
        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        # Run thermalisation in parallel across GPUs
        y_run = jax.pmap(self.thermalize_network_ode)(y0_run, target_run, beta_run, params_run)

        def form_data(y, N_data):
            """Discard padding rows and flatten the device axis."""
            return jnp.concatenate(y[0:N_data, ...])

        # Strip padding: keep only the first target.shape[0] rows
        return jax.tree_map(lambda y: form_data(y, target.shape[0]), y_run)

    # =========================================================================
    #  Optax-based thermalisation (gradient-descent equilibrium finder)
    # =========================================================================

    @partial(jax.jit, static_argnames=['self'])
    def total_force_opt(self, y, target, beta, params):
        """Return the *negative* total force for optax minimisation.

        Optax minimises a function, so we feed it −F (the gradient of the
        total energy) rather than F (the negative gradient that drives ẏ).

        Args:
            y      (dict):        Current network state.
            target (jnp.ndarray): Target angles.
            beta   (float):       Nudging strength.
            params (dict):        Trainable parameters.

        Returns:
            dict: Negative total force (= gradient of total energy w.r.t. y).
        """
        F = self.total_force(0., y, target, beta, params)
        # Negate: optax minimises by moving in the direction of -gradient = +F
        return jax.tree_map(jnp.negative, F)

    def single_run_func_opt(self, y0, target, beta, params):
        """Find equilibrium for a single sample using an optax optimizer.

        Iterates the optax update rule until the total force norm falls below
        self.tol or self.maxtime steps are reached.  The loop is implemented
        with jax.lax.while_loop for JIT-compatibility.

        Args:
            y0     (dict):         Initial state.
            target (jnp.ndarray):  Target angles.
            beta   (float):        Nudging strength.
            params (dict):         Trainable parameters.

        Returns:
            dict: Approximate equilibrium state.
        """
        opt_state = self.optimizer.init(y0)
        tol = 1e-2
        tries = 0
        absF = 1.
        y = y0

        def update_func(y, opt_state, target):
            """Perform one optax gradient step."""
            # Compute negative force = gradient of total energy
            F = self.total_force_opt(y, target, beta, params)
            updates, opt_state = self.optimizer.update(F, opt_state)
            y = optax.apply_updates(y, updates)
            return y, opt_state, F

        def cond_func(vals):
            """Continue while force norm > tol and iterations < maxtime."""
            absF, tries, y, opt_state = vals
            return jnp.logical_and(absF > self.tol, tries < self.maxtime)

        def body_func(vals):
            """One iteration: update state and recompute force norm."""
            absF, tries, y, opt_state = vals
            y, opt_state, F = update_func(y, opt_state, target)
            # Sum Frobenius norms across all layers to get a scalar convergence metric
            absF = jnp.sum(jnp.asarray((jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, F)))))
            tries += 1
            return absF, tries, y, opt_state

        init_vals = absF, tries, y, opt_state
        # while_loop is XLA-compilable and avoids Python-level iteration overhead
        absF, tries, y, opt_state = jax.lax.while_loop(cond_func, body_func, init_vals)

        return y

    @partial(jax.jit, static_argnames=['self'])
    def thermalize_network_opt(self, y0, target, beta, params):
        """Thermalise a full batch using optax (single-GPU).

        Args:
            y0     (dict):         Batch of initial states.
            target (jnp.ndarray):  Batch of targets.
            beta   (float):        Nudging strength.
            params (dict):         Trainable parameters.

        Returns:
            dict: Batch of equilibrium states.
        """
        run_func = lambda y0, target: self.single_run_func_opt(y0, target, beta, params)
        y = jax.vmap(run_func, (0, 0))(y0, target)
        return y

    @partial(jax.jit, static_argnames=['self'])
    def pmap_thermalize_opt(self, y0, target, beta, params):
        """Thermalise a batch using optax distributed across multiple GPUs.

        Analogous to pmap_thermalize_ode but uses optax for finding equilibria.

        Args:
            y0     (dict):         Batch of initial states.
            target (jnp.ndarray):  Batch of targets.
            beta   (float):        Nudging strength.
            params (dict):         Trainable parameters.

        Returns:
            dict: Equilibrium states for the original batch.
        """
        devices = jax.devices()
        N_devices = len(devices)

        y0_run, target_run = self.pad_data(y0, target, N_devices)
        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        y_run = jax.pmap(self.thermalize_network_opt)(y0_run, target_run, beta_run, params_run)

        def form_data(y, N_data):
            return jnp.concatenate(y[0:N_data, ...])

        return jax.tree_map(lambda y: form_data(y, target.shape[0]), y_run)

    # =========================================================================
    #  Per-layer learning-rate normalisation
    # =========================================================================

    def get_layer_ratio(self, params, g_params):
        """Compute ‖W‖ / ‖∂L/∂W‖ for every layer.

        Aggregates per-layer ratios from each Layer's own get_layer_ratio method
        into a nested dict with the same structure as params.

        Args:
            params   (dict): Nested parameter dict.
            g_params (dict): Nested gradient dict (same structure).

        Returns:
            dict: Nested dict of scalar ratios, keyed by layer name.
        """
        ratio_dict = {}
        for layer_name in self.layer_order:
            ratio_dict.update({layer_name: self.layers[layer_name].get_layer_ratio(params[layer_name], g_params[layer_name])})
        return ratio_dict

    def get_normalized_learning_rate(self, params, g_params, learning_rate):
        """Compute per-layer learning rates normalised by parameter-to-gradient ratio.

        The idea is to scale each layer's learning rate so that the relative
        parameter update ‖ΔW‖/‖W‖ is equal across all layers.  Specifically:

            lr_l = learning_rate * (‖W_l‖ / ‖∇W_l‖) / max_l(‖W_l‖ / ‖∇W_l‖)

        The max-normalisation keeps the maximum per-layer rate equal to
        learning_rate so the global scale is preserved.

        Args:
            params        (dict):  Current parameters.
            g_params      (dict):  Current gradients.
            learning_rate (float): Global base learning rate.

        Returns:
            dict: Per-parameter learning rates with the same structure as params.
        """
        ratio_dict = self.get_layer_ratio(params, g_params)
        # Normalise so the largest ratio = 1 (preserves the global learning_rate scale)
        normalizer = max(jax.tree_util.tree_leaves(ratio_dict))
        augment_ratio = jax.tree_map(lambda x: jnp.divide(x, normalizer), ratio_dict)
        return jax.tree_map(lambda x: jnp.multiply(x, learning_rate), augment_ratio)


# =============================================================================
#  Autoencoder (first definition — simple version without inference layer)
# =============================================================================

class Autoencoder(Module):
    """XY-model EP autoencoder (simple version).

    Stacks a convolutional encoder (Conv2D + Pool2D) followed by a decoder
    (Unsample2D + Unsample2D) so that the output has the same spatial shape
    as the input.

    The internal energy is naturally split into two halves:
      encoder half:  E_enc = E(input → c1) + E(c1 → pool1)
      decoder half:  E_dec = E(pool1 → u1) + E(u1 → tc1)

    Both halves contribute equally to the total internal energy used in EP
    (this symmetry is baked into the layer_order list).

    The external energy compares only the output (tc1) against the target,
    using slice indexing to handle potential size mismatches between the
    output feature map and the target.
    """

    def setup(self):
        """Define the encoder–decoder architecture.

        Override this method in subclasses to change the layer configuration.
        """
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x - y)
        def bias_func(x, bias_params): return -bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'

        output_channel, filter_shape, strides = 5, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)

        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)

        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)

        output_channel, filter_shape, strides = 1, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)

        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]

    @partial(jax.jit, static_argnames=['self'])
    def distance_function(self, y, target, params):
        """XY angular distance between output and target, with slice to match shapes.

        The output may be slightly larger than the target due to 'SAME' padding
        in the transposed convolution layers; slicing aligns them.

        Args:
            y      (jnp.ndarray): Output angles (may be larger than target).
            target (jnp.ndarray): Target angles.
            params (dict):        Unused.

        Returns:
            float: Scalar angular distance.
        """
        # y is only the output
        # Build a slice that selects only the target-shaped region from y
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(1 - jnp.cos(y[slices] - target)) / 2.

    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        """Compute nudging energy, slicing the output to match target shape.

        Args:
            y      (jnp.ndarray): Output angles.
            target (jnp.ndarray): Target angles.

        Returns:
            float: Scalar external energy.
        """
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(jax.vmap(self.cost_func, (0, 0))(y[slices], target))

    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        """Compute the external force on the output layer.

        Args:
            y      (jnp.ndarray): Output angles.
            target (jnp.ndarray): Target angles.
            params (dict):        Unused.

        Returns:
            jnp.ndarray: Nudging force on the output layer.
        """
        return -jax.grad(self.external_energy, argnums=0)(y, target)


# =============================================================================
#  Generate_Module — carve out a sub-network from an existing module
# =============================================================================

class Generate_Module(Module):
    """Helper that constructs a sub-network from an existing Module's layers.

    Useful for extracting the encoder or inference sub-graph of an autoencoder
    after training, so it can be run independently.

    Usage::

        encoder, enc_params = autoencoder.generate_encoder(rng, params)

    The extracted sub-network shares the *same Layer objects* as the original
    network (no copying), so the energy/force functions remain identical.
    """

    def setup(self, layers, layer_order, input_type):
        """Wire up a pre-built set of layers in the specified order.

        Args:
            layers      (dict): Layer objects keyed by name.
            layer_order (list): Ordered list of layer names.
            input_type  (str):  Type tag for the first layer's input.
        """
        self.input_type = input_type
        self.layers = layers
        self.layer_order = layer_order
        self.output_name = layer_order[-1]

    def select_equi_func(self):
        """Select the thermalisation function based on optimizer and device count."""
        N_devices = len(jax.devices())
        if self.optimizer is None:
            if N_devices > 1:
                self.thermalize_network = self.pmap_thermalize_ode
            else:
                self.thermalize_network = self.thermalize_network_ode
        else:
            if N_devices > 1:
                self.thermalize_network = self.pmap_thermalize_opt
            else:
                self.thermalize_network = self.thermalize_network_opt

    def get_init_params(self, rng, input_type, input_data, layers, layer_order, optimizer, params):
        """Initialise the sub-network, reusing existing params where available.

        For each layer in layer_order:
          - If the layer name is already in `params`, reuse those parameters.
          - Otherwise, call get_init_params to generate new random ones.

        Args:
            rng         (jax.random.PRNGKey): Random seed.
            input_type  (str):  Type tag for the first input.
            input_data  (jnp.ndarray): Sample input (shape matters).
            layers      (dict): Layer objects.
            layer_order (list): Ordered layer names.
            optimizer:          Optax optimizer or None.
            params      (dict): Existing parameter dict (may be partial).

        Returns:
            dict: Parameter dict for the sub-network.
        """
        self.setup(layers, layer_order, input_type)
        self.optimizer = optimizer
        self.select_equi_func()

        def prod(xl):
            p = 1
            for k in range(0, len(xl)):
                p = p * xl[k]
            return p

        def to_str(xl):
            y = '{0}'.format(xl[0])
            for k in range(1, len(xl)):
                y = y + '*{0}'.format(xl[k])
            return y

        input_shape = input_data.shape
        input_size = prod(input_shape)
        self.F0 = {'input_data': 0 * input_data}
        self.N = input_size
        self.N_list = [input_size]
        self.structure = [to_str(input_shape)]
        former_layer_type = self.input_type
        k = 0
        for name in self.layer_order:
            input_data, N_nodes, layer_structure = self.layers[name].get_layer_size(input_data, former_layer_type)
            if name in params.keys():
                # Reuse pre-trained parameters for this layer
                self.params.update({name: params[name]})
            else:
                # Initialise new random parameters
                self.params.update({name: self.layers[name].get_init_params(rng)})
            self.layers[name].setup()
            self.N = self.N + N_nodes
            self.N_list.append(N_nodes)
            self.structure.append(layer_structure)
            self.F0.update({name: 0 * input_data})
            former_layer_type = self.layers[name].layer_type
        return self.params


# =============================================================================
#  Autoencoder (second definition — full version with inference layer)
# =============================================================================

class Autoencoder(Module):
    """XY-model EP autoencoder with an additional inference (classification) head.

    Architecture:
      Encoder:   Conv2D (c1) → Pool2D (pool1)
      Decoder:   Unsample2D (u1) → Unsample2D (tc1)
      Inference: c1 → pool1 → Denselayer (output_layer)

    The full layer_order for reconstruction training is ['c1', 'pool1', 'u1', 'tc1'].
    The encoder_order ['c1', 'pool1'] and inference_order ['c1', 'pool1', 'output_layer']
    let generate_encoder() and generate_inference_nn() carve out sub-networks.

    External energy and distance function slice the output to handle padding
    size mismatches from transposed convolutions (same as the simple Autoencoder).
    """

    def setup(self):
        """Define the encoder, decoder, and inference head architecture."""
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x - y)
        def bias_func(x, bias_params): return -bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'

        # --- Encoder ---
        output_channel, filter_shape, strides = 5, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)

        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)

        # --- Decoder ---
        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)

        output_channel, filter_shape, strides = 1, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)

        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]

        # --- Inference head (classification output) ---
        layer_params = 10, coup_func, bias_func
        self.output_layer = Denselayer(*layer_params)

        # Sub-network orders for carving out sub-graphs
        self.encoder_order = ['c1', 'pool1']
        self.inference_order = ['c1', 'pool1', 'output_layer']

    @partial(jax.jit, static_argnames=['self'])
    def distance_function(self, y, target, params):
        """XY distance with slice alignment (same as simple Autoencoder).

        Args:
            y      (jnp.ndarray): Output angles.
            target (jnp.ndarray): Target angles.
            params (dict):        Unused.

        Returns:
            float: Scalar angular distance.
        """
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(1 - jnp.cos(y[slices] - target)) / 2.

    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        """Compute nudging energy with slice alignment.

        Args:
            y      (jnp.ndarray): Output angles.
            target (jnp.ndarray): Target angles.

        Returns:
            float: Scalar external energy.
        """
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(jax.vmap(self.cost_func, (0, 0))(y[slices], target))

    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        """Compute the external force on the output layer.

        Args:
            y      (jnp.ndarray): Output angles.
            target (jnp.ndarray): Target angles.
            params (dict):        Unused.

        Returns:
            jnp.ndarray: Nudging force.
        """
        return -jax.grad(self.external_energy, argnums=0)(y, target)

    def generate_encoder(self, rng, params):
        """Extract the encoder sub-network (c1 + pool1) from this autoencoder.

        Args:
            rng    (jax.random.PRNGKey): Random key (used if new params needed).
            params (dict):               Current parameter dict.

        Returns:
            tuple: (encoder_module, encoder_params)
        """
        nn = Generate_Module(self.cost_func)
        encoder_layers = {}
        for name in self.encoder_order:
            encoder_layers.update({name: self.layers[name]})
        # self.F0['input_data'] holds a zero array with the right input shape
        new_params = nn.get_init_params(rng, self.input_type, self.F0['input_data'], encoder_layers, self.encoder_order, self.optimizer, params)
        return nn, new_params

    def generate_inference_nn(self, rng, params):
        """Extract the inference sub-network (c1 + pool1 + output_layer).

        Args:
            rng    (jax.random.PRNGKey): Random key.
            params (dict):               Current parameter dict.

        Returns:
            tuple: (inference_module, inference_params)
        """
        nn = Generate_Module(self.cost_func)
        inf_layers = {}
        for name in self.inference_order:
            inf_layers.update({name: self.layers[name]})
        new_params = nn.get_init_params(rng, self.input_type, self.F0['input_data'], inf_layers, self.inference_order, self.optimizer, params)
        return nn, new_params


# =============================================================================
#  My_nn — example inference network template
# =============================================================================

class My_nn(Module):
    """Concrete example inference network: Conv2D → Pool2D → Dense.

    This class serves as a template showing how to subclass Module to define
    a custom architecture for image classification.

    Architecture:
        Conv2D   (3 channels, 4×4 kernel, stride 2×2)
        Pool2D   (2 channels, 3×3 kernel, stride 3×3)
        Dense    (10 output neurons)
    """

    def __init__(self, cost_func, network_type='general XY', structure_name='dnn'):
        """Initialise My_nn without an optax optimizer (ODE thermalisation).

        Args:
            cost_func      (callable): External cost function.
            network_type   (str):      Descriptive tag.
            structure_name (str):      Descriptive tag.
        """
        # Here network_structure = N, input_index, output_index, layer_sizes

        self.network_type = network_type
        self.structure_name = structure_name

        self.layer_order = []
        self.layers = {}
        self.params = {}

        self.cost_func = cost_func

    def setup(self):
        """Define the Conv2D → Pool2D → Dense architecture."""
        def coup_func(x, y): return -jnp.cos(x - y)
        def bias_func(x, bias_params): return -bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'

        # First convolutional layer: 3 channels, 4×4 kernel, stride 2
        output_channel, filter_shape, strides = 3, [4, 4], [2, 2]
        self.c1 = Conv2D(output_channel, filter_shape, strides, coup_func, bias_func)

        # Pooling layer: 2 channels, 3×3 window, stride 3
        output_channel, filter_shape, strides = 2, [3, 3], [3, 3]
        self.pool1 = Pool2D(output_channel, filter_shape, strides, coup_func, bias_func)

        # Dense classification head: 10 output classes
        layer_params = 10, coup_func, bias_func
        self.l1 = Denselayer(*layer_params)

        self.layer_order = ['c1', 'pool1', 'l1']
        self.output_name = self.layer_order[-1]


# =============================================================================
#  My_AE — concrete autoencoder example
# =============================================================================

class My_AE(Autoencoder):
    """Concrete autoencoder example (reconstruction only, no inference head).

    Architecture:
        Encoder: Conv2D (c1, 5 ch, 2×2, s=2) → Pool2D (pool1, 5 ch, 3×3, s=3)
        Decoder: Unsample2D (u1, 5 ch, 3×3, s=3) → Unsample2D (tc1, 1 ch, 2×2, s=2)

    Inherits external_energy, distance_function, and external_force from
    the parent Autoencoder class.  Does not include an inference head.
    """

    def setup(self):
        """Define the encoder–decoder architecture (no inference layer)."""
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x - y)
        def bias_func(x, bias_params): return -bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'

        output_channel, filter_shape, strides = 5, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)

        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)

        output_channel, filter_shape, strides = 5, [3, 3], [3, 3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)

        output_channel, filter_shape, strides = 1, [2, 2], [2, 2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)

        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]
