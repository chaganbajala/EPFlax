import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import optax
import time
import pickle
import gc
# Code for training
from functools import partial

class Layer:
    def __init__(self):
        pass
    
    def get_layer_size(self, input_data, former_layer_type):
        '''
        Get information of privious layer, determine the shape of the current layer.
        Return a smaple data, the size and the shape of current layer. 
        '''
        pass
    
    def get_init_params(self):
        pass
    
    def energy(self):
        pass
    
    def forward_force(self):
        pass
    
    def backward_force(self):
        pass
    
    def get_layer_ratio(self, params, g_params):
        ratio = {}
        abs_params = 0.
        abs_g = 0.
        
        for name in params:
            abs_params = abs_params + jnp.linalg.norm(params[name])**2
            abs_g = abs_g + jnp.linalg.norm(g_params[name])**2
        
            abs_params = jnp.sqrt(abs_params)
            abs_g = jnp.sqrt(abs_g)
        
        for name in params:
            ratio.update({name:abs_params/abs_g})
            
        return ratio
    
    def params_norm(self, params):
        return 0.
    

class Denselayer(Layer):
    '''
    This define a layer with general connectivity
    '''
    def __init__(self, output_size, coup_func, bias_func, layer_type='dense'):
        self.output_size = output_size
        self.layer_type = layer_type
        self.bias_func = bias_func
        self.coup_func = coup_func
        
        self.d0_coup = jax.grad(coup_func, 0)
        self.d1_coup = jax.grad(coup_func, 1)
        
        self.vd0_coup = jax.vmap(self.d0_coup, (None, 0))
        self.md0_coup = jax.vmap(self.vd0_coup, (0, None))
        
        self.vd1_coup = jax.vmap(self.d1_coup, (None, 0))
        self.md1_coup = jax.vmap(self.vd1_coup, (0, None))
        
        self.d0_bias = jax.grad(self.bias_func, 0)
        self.d1_bias = jax.grad(self.bias_func, 1)
        
        self.vd0_bias = jax.vmap(self.d0_bias, (0, 0))
        self.vd1_bias = jax.vmap(self.d1_bias, (0, 0))
        
        self.vbias = jax.vmap(self.bias_func, (0,0))
        self.vcoup = jax.vmap(self.coup_func, (None, 0))
        self.mcoup = jax.vmap(self.vcoup, (0, None))
    
    def get_layer_size(self, input_data, former_layer_type):
        self.former_layer_type = former_layer_type
        self.input_size = input_data.flatten().shape[0]
        return jnp.zeros(self.output_size), self.output_size, '{0}'.format(self.output_size)
    
    def get_init_params(self, rng):
        W = jax.random.normal(rng, [self.input_size, self.output_size]) / jnp.sqrt(self.input_size + self.output_size)
        h = jnp.zeros(self.output_size)
        psi = jax.random.uniform(rng, shape=[self.output_size], minval=-jnp.pi, maxval=jnp.pi)
        return {'weights':W, 'bias field': jnp.asarray([h , psi]).transpose()}
    
    def setup(self):
        if self.former_layer_type == 'dense':
            self.energy = self.energy_std
            self.force = self.force_std
        else:
            self.energy = self.energy_flatten
            self.force = self.force_flatten
    
    def get_init_state(self, N_data):
        y0 = np.pi * 2 * (np.random.rand(N_data, self.output_size) - 0.5)
        return y0
    
    def force_std(self, y1, params, y2):
        # y1: value from last layer
        # y2: value at this layer
        W, bias = params['weights'], params['bias field']
        FF = jnp.sum(W*self.md0_coup(y1, y2), axis=0)
        BF = jnp.sum(W*self.md1_coup(y1, y2), axis=1)
        OF = - self.vd0_bias(y2, bias)
        
        return FF+OF, BF
    
    def energy_std(self, y1, params, y2):
        W, bias = params['weights'], params['bias field']
        E0 = jnp.sum(W*self.mcoup(y1, y2))
        E1 = jnp.sum(self.vbias(y2, bias))
        return E0+E1
    
    def force_flatten(self, y1, params, y2):
        ny1 = y1.flatten()
        FF, BF = self.force_std(ny1, params, y2)
        BF = BF.reshape(*y1.shape)
        return FF, BF
    
    def energy_flatten(self, y1, params, y2):
        ny1 = y1.flatten()
        return self.energy_std(ny1, params, y2)
    
    def get_layer_ratio(self, params, g_params):
        ratio = {}
        r = jnp.linalg.norm(params['weights'])/jnp.linalg.norm(g_params['weights'])
        ratio.update({'weights': r})
        ratio.update({'bias field': r})
        
        '''
        ratio.update({
            'bias field': jnp.sqrt(jnp.square(g_params['bias field'][0]) 
                                   + jnp.square(params['bias field'][0]) * jnp.square(g_params['bias field'][1])).sum()
        })
        '''
            
        return ratio

class Conv1D(Denselayer):
    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv1D'):
        
        self.coup_func = coup_func
        self.bias_func = bias_func
        
        self.layer_type = layer_type
        
        self.output_channel = output_channel
        self.filter_shape = filter_shape
        
        self.vbias = jax.vmap(self.bias_func, (0,0))
        
        self.d0_bias = jax.grad(self.bias_func, 0)
        self.d1_bias = jax.grad(self.bias_func, 1)
        
        self.vd0_bias = jax.vmap(self.d0_bias, (0, 0))
        self.vd1_bias = jax.vmap(self.d1_bias, (0, 0))
        
        self.strides = strides
        
    def get_layer_size(self, input_data, former_layer_type):
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
        y = jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')
        self.output_size = y.shape[-1]
        
        return y[0], self.output_channel * self.output_size, '{0}*{1}'.format(self.output_channel, self.output_size)
        
    
    def get_init_params(self, rng):
        S = self.input_channel * self.input_size + self.output_channel * self.output_size
        F = 1/jnp.sqrt(S) * jax.random.normal(rng, shape=(self.output_channel, self.input_channel, self.filter_shape))
        h = jnp.zeros(self.output_channel)
        psi = jax.random.uniform(rng, shape=(self.output_channel,), minval=-jnp.pi, maxval=jnp.pi)
        return {'kernel':F, 'bias field': jnp.asarray([h, psi]).transpose()}
    
    def setup(self):
        if self.former_layer_type == 'conv1D':
            self.energy = self.energy_std
        elif self.former_layer_type == 'conv2D':
            self.energy = self.energy_from_conv2D
        elif self.former_layer_type == 'dense':
            self.energy = self.energy_from_dense
    
    def get_init_state(self, N_data):
        return (np.random.rand(N_data, self.output_channel, self.output_size) - 0.5) * np.pi *2
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_std(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny2 = conv_func(jnp.exp(-1j*y1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_from_dense(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        ny1 = y1[None,:]
        ny2 = conv_func(jnp.exp(-1j*ny1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1
    
    def energy_from_conv2D(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        ny2 = conv_func(jnp.exp(-1j*ny1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1
    
    #@partial(jax.jit, static_argnames=['self'])
    def force(self, y1, params, y2):
        #print(y1.shape, y2.shape)
        FF = -jax.grad(self.energy, 2)(y1, params, y2)
        BF = -jax.grad(self.energy, 0)(y1, params, y2)
        
        return FF, BF
    
    def get_layer_ratio(self, params, g_params):
        ratio = {}
        r = jnp.linalg.norm(params['kernel'])/jnp.linalg.norm(g_params['kernel'])
        ratio.update({'kernel': r})
        ratio.update({'bias field': r})
        
        '''
        ratio.update({
            'bias field': jnp.sqrt(jnp.square(g_params['bias field'][:,0]) 
                                   + jnp.square(params['bias field'][:,0]) * jnp.square(g_params['bias field'][:,1])).sum()
        })
        '''
            
        return ratio
    
class Pool1D(Conv1D):
    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv1D'):
        super().__init__(output_channel, filter_shape, strides, coup_func, bias_func, layer_type)
        
    def get_init_params(self, rng):
        self.kernel = jnp.ones([self.filter_shape], dtype=jnp.complex64)/self.filter_shape
        return {'kernel':0.}
    
    def energy_std(self, y1, params, y2):
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny2 = conv_func(jnp.exp(-1j*y1), self.kernel)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    def single_pool_func(self, y):
        return jax.lax.conv(y[None,None,...], self.kernel[None,None,...], window_strides=self.strides ,padding='SAME')[0,...]
    
    def pool_func(self, y):
        return jax.vmap(self.single_pool_func, 0)(y)
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,:]
        ny2 = self.pool_func(jnp.exp(-1j*ny1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    def energy_from_conv2D(self, y1, params, y2):
        
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        ny2 = self.pool_func(jnp.exp(-1j*ny1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    def get_layer_ratio(self, params, g_params):
        return {'kernel':0.}
    
class Conv2D(Conv1D):
    def __init__(self, output_channel, filter_shape, strides, coup_func, bias_func, layer_type='conv2D'):
        super().__init__(output_channel, filter_shape, strides, coup_func, bias_func, layer_type)
        # Note that now self.filter_shape is a list [a, b]
        
    def get_layer_size(self, input_data, former_layer_type):
            
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            self.input_size = [input_data.shape[0], 1]
        elif former_layer_type == 'conv1D':
            self.input_channel, self.input_size = input_data.shape[0], [input_data.shape[1], 1]
        elif former_layer_type == 'conv2D':
            self.input_channel, self.input_size = input_data.shape[0], [input_data.shape[1], input_data.shape[2]]
                
        y = np.zeros([self.input_channel, *self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, *self.filter_shape])
        #print(y.shape, kernel.shape)
        y = jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')
        self.output_size = [ y.shape[-2], y.shape[-1] ]
            
        return y[0], self.output_channel * self.output_size[0] * self.output_size[1], '{0}*{1}*{2}'.format(self.output_channel, *self.output_size)
        
    def get_init_params(self, rng):
        S = self.input_channel * self.input_size[0] * self.input_size[1] + self.output_channel * self.output_size[0] * self.output_size[1]
        F = 1/jnp.sqrt(S) * jax.random.normal(rng, shape=(self.output_channel, self.input_channel, *self.filter_shape))
        h = jnp.zeros(self.output_channel)
        psi = jax.random.uniform(rng, shape=(self.output_channel,), minval=-jnp.pi, maxval=jnp.pi)
        return {'kernel':F, 'bias field': jnp.asarray([h, psi]).transpose()}
    
    def get_init_state(self, N_data):
        return (np.random.rand(N_data, self.output_channel, *self.output_size) - 0.5) * np.pi *2
    
    def setup(self):
        if self.former_layer_type == 'conv2D':
            self.energy = self.energy_std
        elif self.former_layer_type == 'conv1D':
            self.energy = self.energy_from_conv1D
        elif self.former_layer_type == 'dense':
            self.energy = self.energy_from_dense
            
    def energy_std(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny2 = conv_func(jnp.exp(-1j*y1), F)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0 + E1
    
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,:,None]
        return self.energy_std(ny1, params, y2)
    
    def energy_from_conv1D(self, y1, params, y2):
        ny1 = y1[...,None]
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
    
class Pool2D(Conv2D):
    def get_init_params(self, rng):
        self.kernel = jnp.ones([*self.filter_shape], dtype=jnp.complex64)/(self.filter_shape[0] * self.filter_shape[1])
        return {'kernel':0.}
    
    def single_pool_func(self, y):
        return jax.lax.conv(y[None,None,...], self.kernel[None,None,...], window_strides=self.strides ,padding='SAME')[0,...]
    
    def pool_func(self, y):
        return jax.vmap(self.single_pool_func, 0)(y)
    
    def energy_std(self, y1, params, y2):
        ny2 = self.pool_func(jnp.exp(-1j*y1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,:]
        ny2 = self.pool_func(jnp.exp(-1j*ny1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    def energy_from_conv1D(self, y1, params, y2):
        ny1 = jax.vmap(jnp.concatenate, (0))(y1)
        ny2 = self.pool_func(jnp.exp(-1j*ny1), self.kernel)
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    def get_layer_ratio(self, params, g_params):
        return {'kernel':0.}
    
class Intra_connected(Denselayer):
    def __init__(self, output_size, coup_func, bias_func, layer_type='dense', ic_type='None', ic_args = None):
        super().__init__(output_size, coup_func, bias_func, layer_type)
        '''
        ic_type refers to "intra connection type". It can be either 'None', 'full', 'graph', 'layer' or 'lattice'. 
        '''
        self.ic_type = ic_type
        self.ic_args = ic_args
    
    def get_init_params(self, rng):
        W = jax.random.normal(rng, [self.input_size, self.output_size]) / jnp.sqrt(self.input_size + self.output_size)
        h = jnp.zeros(self.output_size)
        psi = jax.random.uniform(rng, shape=[self.output_size], minval=-jnp.pi, maxval=jnp.pi)
        
        if self.ic_type == 'None':
            return {'weights':W, 'bias field': [h , psi]}
        elif self.ic_type == 'full':
            W_in = jax.random.normal(rng, shape = [self.output_size, self.output_size])/jnp.sqrt(self.output_size)
            return {'weight':W, 'bias field': [h , psi], 'intra coupling': W_in}
        elif self.ic_type == 'graph':
            W_in = jax.random.normal(rng, shape = [self.ic_args.shape[0]])/jnp.sqrt(self.output_size)
            return {'weight':W, 'bias field': [h , psi], 'intra coupling': W_in}

        
#----------------------------- Transposed Convolution and Unsample Layers ---------------------

class TransConv1D(Conv1D):
    
    @staticmethod
    def transposed_conv(y, kernel, strides, padding='SAME'):
            # Input tensor: batch size, input channel, dim1, dim2
            # Input kernel: output channel, input channel, dim1, dim2
            #print(y.shape)
            y_run = y.transpose([0,2,1]) # For jax.lax.conv_transpose: batch size, dim1, dim2, input channel
            kernel_run = kernel.transpose([2,1,0]) # For jax.lax.conv_transpose:  dim1, dim2, input channels, output channels
            

            # Perform the transposed convolution, output: batch size, dim1, dim2, channel
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=strides, padding=padding)
            return output.transpose([0,2,1])

    def get_layer_size(self, input_data, former_layer_type):
        
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
        #print(y.shape, kernel.shape)
        y = self.transposed_conv(y[None,...], kernel, strides=self.strides ,padding='SAME')
        self.output_size = y.shape[-1]
        print(y.shape)
            
        return y[0], self.output_channel * self.output_size, '{0}*{1}'.format(self.output_channel, self.output_size)
    
    def energy_std(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny2 = conv_func(jnp.exp(-1j*y2), jnp.flip(F.transpose([1,0,2]), axis=(2)))
        E0 = - jnp.real(jnp.exp(1j*y1) * ny2).sum()
        return E0 + E1
    
    def energy_from_conv2D(self, y1, params, y2):
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)
    
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,...,None]
        return self.energy_std(ny1, params, y2)


class TransConv2D(Conv2D):
    @staticmethod
    def transposed_conv(y, kernel, strides, padding='SAME'):
            # Input tensor: batch size, input channel, dim1, dim2
            # Input kernel: output channel, input channel, dim1, dim2
            
            y_run = y.transpose([0,2,3,1]) # For jax.lax.conv_transpose: batch size, dim1, dim2, input channel
            kernel_run = kernel.transpose([2,3,1,0]) # For jax.lax.conv_transpose:  dim1, dim2, input channels, output channels
            

            # Perform the transposed convolution, output: batch size, dim1, dim2, channel
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=strides, padding=padding)
            return output.transpose([0,3,1,2])
    
    def get_layer_size(self, input_data, former_layer_type):
            
        self.former_layer_type = former_layer_type
        if former_layer_type == 'dense':
            self.input_channel = 1
            self.input_size = [input_data.shape[0], 1]
        elif former_layer_type == 'conv1D':
            self.input_channel, self.input_size = input_data.shape[0], [input_data.shape[1], 1]
        elif former_layer_type == 'conv2D':
            self.input_channel, self.input_size = input_data.shape[0], [input_data.shape[1], input_data.shape[2]]
                
        y = np.zeros([self.input_channel, *self.input_size])
        kernel = np.zeros([self.output_channel, self.input_channel, *self.filter_shape])
        #print(y.shape, kernel.shape)
        y = self.transposed_conv(y[None,...], kernel, strides=self.strides ,padding='SAME')
        self.output_size = [ y.shape[-2], y.shape[-1] ]
            
        return y[0], self.output_channel * self.output_size[0] * self.output_size[1], '{0}*{1}*{2}'.format(self.output_channel, *self.output_size)
    
    def energy_std(self, y1, params, y2):
        F, bias = jnp.asarray(params['kernel'], dtype=jnp.complex64), params['bias field']
        E1 = jnp.sum(self.vbias(y2, bias))
        
        def conv_func(y, kernel):
            return jax.lax.conv(y[None,...], kernel, window_strides=self.strides ,padding='SAME')[0,...]
        
        ny2 = conv_func(jnp.exp(-1j*y2), jnp.flip(F.transpose([1,0,2,3]), axis=(2,3)))
        E0 = - jnp.real(jnp.exp(1j*y1) * ny2).sum()
        return E0 + E1
    
    def energy_from_conv1D(self, y1, params, y2):
        ny1 = y1[..., None]
        return self.energy_std(ny1, params, y2)
    
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,...,None]
        return self.energy_std(ny1, params, y2)

    
class Unsample1D(TransConv1D):
    def get_init_params(self, rng):
        self.kernel = jnp.ones([self.filter_shape], dtype=jnp.complex64)
        return {'kernel':0.}
    
    def single_unsample_func(self, y):
        def transpose_conv(y, kernel, padding='SAME'):
            # Input tensor: batch size, input channel, dim1, dim2
            # Input kernel: output channel, input channel, dim1, dim2

            y_run = y.transpose([0,2,1]) # For jax.lax.conv_transpose: batch size, dim1, dim2, input channel
            kernel_run = kernel.transpose([2,1,0]) # For jax.lax.conv_transpose:  dim1, dim2, input channels, output channels

            # Perform the transposed convolution, output: batch size, dim1, dim2, channel
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=self.strides, padding=padding, rhs_dilation=None)
            return output.transpose([0,2,1])
        
        y = y[None, None, ...]
        #kernel = jnp.ones([1,1,*self.strides])
        return transpose_conv(y, self.kernel[None,None,...])[0,0,...]

    def unsample_func(self, y):
        return jax.vmap(self.single_unsample_func, (0))(y)
    
    def energy_std(self, y1, params, y2):
        ny2 = self.unsample_func(jnp.exp(-1j*y1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,:,None]
        return self.energy_std(ny1, params, y2)
    
    def energy_from_conv1D(self, y1, params, y2):
        ny1 = y1[...,None]
        return self.energy_std(ny1, params, y2)
    
    def get_layer_ratio(self, params, g_params):
        return {'kernel':0.}
    
    
class Unsample2D(TransConv2D):
    def get_init_params(self, rng):
        self.kernel = jnp.ones([*self.filter_shape], dtype=jnp.complex64)
        return {'kernel':0.}
    
    def single_unsample_func(self, y):
        def transpose_conv(y, kernel, padding='SAME'):
            # Input tensor: batch size, input channel, dim1, dim2
            # Input kernel: output channel, input channel, dim1, dim2

            y_run = y.transpose([0,2,3,1]) # For jax.lax.conv_transpose: batch size, dim1, dim2, input channel
            kernel_run = kernel.transpose([2,3,1,0]) # For jax.lax.conv_transpose:  dim1, dim2, input channels, output channels

            # Perform the transposed convolution, output: batch size, dim1, dim2, channel
            output = jax.lax.conv_transpose(y_run, kernel_run, strides=self.strides, padding=padding, rhs_dilation=None)
            return output.transpose([0,3,1,2])
        
        y = y[None, None, ...]
        #kernel = jnp.ones([1,1,*self.strides])
        return transpose_conv(y, self.kernel[None,None,...])[0,0,...]

    def unsample_func(self, y):
        return jax.vmap(self.single_unsample_func, (0))(y)
    
    def energy_std(self, y1, params, y2):
        ny2 = self.unsample_func(jnp.exp(-1j*y1))
        E0 = - jnp.real(jnp.exp(1j*y2) * ny2).sum()
        return E0
    
    #@partial(jax.jit, static_argnames=['self'])
    def energy_from_dense(self, y1, params, y2):
        ny1 = y1[None,:,None]
        return self.energy_std(ny1, params, y2)
    
    def energy_from_conv1D(self, y1, params, y2):
        ny1 = y1[...,None]
        return self.energy_std(ny1, params, y2)
    
    def get_layer_ratio(self, params, g_params):
        return {'kernel':0.}
        
#===============Module for a Neural Network==============
    
class Network:
    '''
    This define a neural network. It should include:
    network_structure: network_type, structure_name, structure_parameter, activation
    __init__: initialization
    get_initial_params(): randomly generate a set of trainable parameter for the neural network
    get_initial_state(input_data): generate a random initial state with specific input_data
    internal_energy(y, network_params): calculate internal energy function
    external_energy(y, target, network_params): calculate external energy
    internal_force(y, network_params): 
    external_force(y, network_params):
    params_derivative(self, y, network_params): 
    '''
    
    def __init__(self) -> None:
        pass
    
    def get_initial_params(self):
        pass
    
    def get_initial_state(input_data):
        pass
    
    def internal_energy(self, y, network_params): 
        pass
    
    def distance_function(self, y, target, network_params):
        pass
    
    def external_energy(self, y, target, network_params): 
        return self.distance_function(y, target, network_params)
    
    def internal_force(self, y, network_params): 
        return -jax.grad(self.internal_energy, argnums=0)(y, network_params)
    
    def external_force(self, y, target, network_params):
        return -jax.grad(self.external_energy, argnums=0)(y, target, network_params)
    
    def params_derivative(self, y, network_params):
        d_energy = jax.grad(self.internal_energy, argnums=1)
        return d_energy(y, network_params)    
    
    def get_random_index(self, N_data, batch_size):
        if batch_size == N_data:
            return np.arange(0, N_data, dtype=np.int32)
        return np.random.randint(0, N_data, batch_size)

    
class Module(Network):
    '''
    This is a DNN module. One need to redefine setup function. 
    '''
    def __init__(self, cost_func, run_params=(100, 1e-3, 1e-6), opt_params=(1e-3, 1000), optimizer=None, network_type='general XY', structure_name='dnn'):
        #Here network_structure = N, input_index, output_index, layer_sizes
        self.runtime, self.rtol, self.atol = run_params
        self.tol, self.maxtime = opt_params
        self.network_type = network_type
        self.structure_name = structure_name
        
        self.layer_order = []
        self.layers = {}
        self.params = {}

        self.cost_func = cost_func
        self.optimizer = optimizer
    
    def get_variable_name(self, variable, scope=locals()):
        for name, value in scope.items():
            if value is variable:
                return name
            
        for name, value in self.__dict__.items():
            if value is variable:
                return name
        
    
    def setup(self):
        # Here the user need to clarify input_type, layers, name of output layer and the order of the layers
        self.input_type = None
        self.l1 = None
        self.layer_order = ['l1']
        self.output_name = self.layer_order[-1]
    
    def show_hyperparams(self):
        print("Structure: ", self.structure)
        print("Number of Nodes: ", self.N_list)
        print("Total Number of Nodes: ", self.N)
        if self.optimizer==None:
            print("Optimizer: ODE")
        else:
            print("Optimizer: OPT")

    def get_initial_params(self, rng, input_data):
        self.setup()
        N_devices = len(jax.devices())
        if self.optimizer==None:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_ode
            else:
                self.thermalize_network = self.thermalize_network_ode
        else:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_opt
            else:
                self.thermalize_network = self.thermalize_network_opt
        
        for name, value in self.__dict__.items():
            if hasattr(value, 'layer_type'):
                self.layers.update({name:value})
        
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
        self.F0 = {'input_data': 0*input_data}
        self.N = input_size
        self.N_list = [input_size]
        self.structure = [to_str(input_shape)]
        former_layer_type = self.input_type
        for name in self.layer_order:
            input_data, N_nodes, layer_structure = self.layers[name].get_layer_size(input_data, former_layer_type)
            self.params.update({name: self.layers[name].get_init_params(rng)})
            self.layers[name].setup()
            self.N = self.N + N_nodes
            self.N_list.append(N_nodes)
            self.structure.append(layer_structure)
            self.F0.update({name:0*input_data})
            former_layer_type = self.layers[name].layer_type
        return self.params
    
    def get_initial_state(self, input_data):
        initial_state = {'input_data':input_data}
        N_data = input_data.shape[0]
        for name in self.layer_order:
            current_layer = self.layers[name]
            state = current_layer.get_init_state(N_data)
            initial_state.update({name:state})
        return initial_state


    
    @partial(jax.jit, static_argnames=['self'])
    def internal_force(self, y, params):
        y1 = y['input_data']
        '''
        def multiply_zero(x):
            return x*0
        '''
        #F = jax.tree_map(multiply_zero, y)
        #F = {key: np.zeros_like(value) for key, value in y.items()}
        
        F = self.F0.copy()
        last_layer_name = 'input_data'
        for layer_name in self.layer_order:
            current_layer = self.layers[layer_name]
            y2 = y[layer_name]
            FF, BF= current_layer.force(y1, params[layer_name], y2)
            F[last_layer_name] += BF * (last_layer_name!='input_data')
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
        y1 = y['input_data']
        E = 0.
        
        for layer_name in self.layer_order:
            layer = self.layers[layer_name]
            y2 = y[layer_name]
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
        # y is only the output
        return jnp.sum(1 - jnp.cos(y-target))/2.
    
    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        # y is only the output
        return jnp.sum(jax.vmap(self.cost_func, (0,0))(y, target))
    
    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        return -jax.grad(self.external_energy, argnums=0)(y, target)
    
    def convert_to_lnn_params(self, input_data, params):
        # This transform a structured params into params for a lxynn
        WL = []
        biasL = [np.zeros([2, input_data.shape[1]])]
        for layer_name in self.layer_order:
            WL.append(params[layer_name]['weights'])
            biasL.append(np.asarray(params[layer_name]['bias field']))
        
        return WL, jnp.concatenate(biasL, axis=1)
    
    def merge_y(self, y):
        yl = []
        for layer_name, layer in y.items():
            yl.append(y[layer_name])
        
        return jnp.concatenate(yl, axis=1)
    
    
    #===================Function for Running the Network by Solving ODE===============
    @partial(jax.jit, static_argnames=['self'])
    def free_force(self, t, y0, params):
        return self.internal_force(y0, params)
    
    def single_free_run(self, y0, params):
        t_span = [0, self.runtime]
        
        odefunc = lambda t, y, args: self.free_force(t, y, params)
        eqs = diffrax.ODETerm(odefunc)
        
        # Use 4th Runge Kutta
        solver = diffrax.Tsit5()
        
        #Use 5th Kvaerno for stiff case
        #solver = diffrax.Kvaerno4()
        
        stepsize_controller = diffrax.PIDController(rtol = self.rtol, atol = self.atol)
        #t = diffrax.SaveAt(ts=saveat)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0 = None, y0=y0,
                                stepsize_controller=stepsize_controller, max_steps=10000000)
        
        return jax.tree_map(jnp.concatenate, solution.ys)

    
    @partial(jax.jit, static_argnames=['self'])
    def apply_to(self, input_data, params):
        y0 = self.get_initial_state(input_data)
        return jax.vmap(self.single_free_run, (0,None))(y0, params)
    
    #===================Function for getting equilibriums for training by solving ODE=============
        
    @partial(jax.jit, static_argnames=['self'])
    def total_force(self, t, y, target, beta, params):
        # Calculate total force for single piece of data
        F = self.internal_force(y, params)
        #F1 = beta * self.external_force(y[self.output_name], target, params)
        F[self.output_name] += beta * self.external_force(y[self.output_name], target, params)
        return F
    
    #@partial(jax.jit, static_argnames=['self', 'nn'])
    def single_run_func(self, y0, target, beta, params):
        # Find equilibrium for single piece of data
        
        t_span = [0, self.runtime]
        
        odefunc = lambda t, y0, args: self.total_force(t, y0, target, beta, params)
        eqs = diffrax.ODETerm(odefunc)
        
        # Use 4th Runge Kutta
        solver = diffrax.Tsit5()
        
        #Use 5th Kvaerno for stiff case
        #solver = diffrax.Kvaerno4()
        
        stepsize_controller = diffrax.PIDController(rtol = self.rtol, atol = self.atol)
        #t = diffrax.SaveAt(ts=saveat)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0 = None, y0=y0,
                                stepsize_controller=stepsize_controller, max_steps=10000000)
        
        return jax.tree_map(jnp.concatenate, solution.ys)
    
    @partial(jax.jit, static_argnames=['self'])
    def thermalize_network_ode(self, y0, target, beta, params):
        # Tree_map single_run_func with multiple data and target
        run_func = lambda y0, target: self.single_run_func(y0, target, beta, params)
        y = jax.vmap(run_func, (0,0))(y0, target)
        return y
    
    @staticmethod
    def tree_expand(tree, n):
        def leaf_expand(leaf):
            return jnp.tensordot(jnp.ones(n), leaf, 0)
        return jax.tree_map(leaf_expand, tree)
    
    @staticmethod
    def pad_data(y, target, N_devices):

        def pad_func(y):
            N_data = y.shape[0]
            N_per = N_data//N_devices
            pad_width = [(0, N_devices-N_data%N_devices)]
            for k in range(1, len(y.shape)):
                pad_width.append((0,0))
            #print(pad_width)
            y_pad = jnp.pad(y, pad_width, mode='constant')
            return y_pad

        def reshape_func(y):
            N_data = y.shape[0]
            N_rem  = N_data%N_devices
            N_per = N_data//N_devices
            return y.reshape(N_devices, N_per+int(N_rem==1), *y.shape[1:])

        N_data = target.shape[0]
        if N_data%N_devices==0:
            return jax.tree_map(reshape_func, y), reshape_func(target)
        else:
            y_pad = jax.tree_map(pad_func, y)
            target_pad = pad_func(target)
            return jax.tree_map(reshape_func, y_pad), reshape_func(target_pad)


    from functools import partial

    @partial(jax.jit, static_argnames=['self'])
    def pmap_thermalize_ode(self, y0, target, beta, params):
        devices = jax.devices()
        N_devices = len(devices)
        
        y0_run, target_run = self.pad_data(y0, target, N_devices)
        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        y_run = jax.pmap(self.thermalize_network_ode)(y0_run, target_run, beta_run, params_run)
        def form_data(y, N_data):
            return jnp.concatenate(y[0:N_data,...])
        return jax.tree_map(lambda y: form_data(y, target.shape[0]), y_run)


    #========Function for getting equilibriums for training by searching with optax========
    @partial(jax.jit, static_argnames=['self'])
    def total_force_opt(self, y, target, beta, params):
        F = self.total_force(0., y, target, beta, params)
        return jax.tree_map(jnp.negative, F)
    
    #@partial(jax.jit, static_argnames=['self'])
    def single_run_func_opt(self, y0, target, beta, params):
        # This use Optax to find the equilibrium
        opt_state = self.optimizer.init(y0)
        tol = 1e-2
        tries = 0
        absF = 1.
        y = y0
        
        #@jax.jit
        def update_func(y, opt_state, target):
            F = self.total_force_opt(y, target, beta, params)
            updates, opt_state = self.optimizer.update(F, opt_state)
            y = optax.apply_updates(y, updates)
            return y, opt_state, F
        
        def cond_func(vals):
            absF, tries, y, opt_state = vals
            return jnp.logical_and(absF>self.tol, tries<self.maxtime)
        
        def body_func(vals):
            absF, tries, y, opt_state = vals
            y, opt_state, F = update_func(y, opt_state, target)
            absF = jnp.sum(jnp.asarray((jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, F)))))
            tries += 1
            return absF, tries, y, opt_state
        
        init_vals = absF, tries, y, opt_state
        absF, tries, y, opt_state = jax.lax.while_loop(cond_func, body_func, init_vals)
            
        return y
    
    @partial(jax.jit, static_argnames=['self'])
    def thermalize_network_opt(self, y0, target, beta, params):
        # Tree_map single_run_func with multiple data and target
        run_func = lambda y0, target: self.single_run_func_opt(y0, target, beta, params)
        y = jax.vmap(run_func, (0,0))(y0, target)
        return y
    
    @partial(jax.jit, static_argnames=['self'])
    def pmap_thermalize_opt(self, y0, target, beta, params):
        devices = jax.devices()
        N_devices = len(devices)

        y0_run, target_run = self.pad_data(y0, target, N_devices)
        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        y_run = jax.pmap(self.thermalize_network_opt)(y0_run, target_run, beta_run, params_run)
        def form_data(y, N_data):
            return jnp.concatenate(y[0:N_data,...])
        return jax.tree_map(lambda y: form_data(y, target.shape[0]), y_run)

    #=====================For automatic layerwise updating=================
    def get_layer_ratio(self, params, g_params):
        ratio_dict = {}
        for layer_name in self.layer_order:
            #print(params[layer_name])
            #print(g_params[layer_name])
            #print(self.layers[layer_name].get_layer_ratio(params[layer_name], g_params[layer_name]))
            ratio_dict.update({layer_name: self.layers[layer_name].get_layer_ratio(params[layer_name], g_params[layer_name])})
        return ratio_dict
    
    def get_normalized_learning_rate(self, params, g_params, learning_rate):
        ratio_dict = self.get_layer_ratio(params, g_params)
        normalizer = max(jax.tree_util.tree_leaves(ratio_dict))
        augment_ratio =  jax.tree_map(lambda x: jnp.divide(x, normalizer), ratio_dict)
        return jax.tree_map(lambda x: jnp.multiply(x, learning_rate), augment_ratio)

    
class Autoencoder(Module):
    '''
    This defines an autoencoder
    '''
    def setup(self):
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x-y)
        def bias_func(x, bias_params): return - bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'
        
        output_channel, filter_shape, strides = 5, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)
        
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)
        
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)
        
        output_channel, filter_shape, strides = 1, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)
        
        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]
        
    @partial(jax.jit, static_argnames=['self'])
    def distance_function(self, y, target, params):
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(1 - jnp.cos(y[slices]-target))/2.
    
    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(jax.vmap(self.cost_func, (0,0))(y[slices], target))
    
    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        return -jax.grad(self.external_energy, argnums=0)(y, target)
    
    
class Generate_Module(Module):
    # This help one to define the encoder from an autoencoder.
    def setup(self, layers, layer_order, input_type):
        self.input_type = input_type
        self.layers = layers
        self.layer_order = layer_order
        self.output_name = layer_order[-1]
        
    def select_equi_func(self):
        N_devices = len(jax.devices())
        if self.optimizer==None:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_ode
            else:
                self.thermalize_network = self.thermalize_network_ode
        else:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_opt
            else:
                self.thermalize_network = self.thermalize_network_opt
        
    def get_init_params(self, rng, input_type, input_data, layers, layer_order, optimizer, params):
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
        self.F0 = {'input_data': 0*input_data}
        self.N = input_size
        self.N_list = [input_size]
        self.structure = [to_str(input_shape)]
        former_layer_type = self.input_type
        k = 0
        for name in self.layer_order:
            input_data, N_nodes, layer_structure = self.layers[name].get_layer_size(input_data, former_layer_type)
            if name in params.keys():
                self.params.update({name: params[name]})
            else:
                self.params.update({name: self.layers[name].get_init_params(rng)})
            self.layers[name].setup()
            self.N = self.N + N_nodes
            self.N_list.append(N_nodes)
            self.structure.append(layer_structure)
            self.F0.update({name:0*input_data})
            former_layer_type = self.layers[name].layer_type
        return self.params

class Autoencoder(Module):
    '''
    This defines an autoencoder
    '''
    def setup(self):
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x-y)
        def bias_func(x, bias_params): return - bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'
        
        # Define the encoder
        output_channel, filter_shape, strides = 5, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)
        
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)
        
        # Define the decoder
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)
        
        output_channel, filter_shape, strides = 1, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)
        
        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]
        
        # Define the output
        layer_params = 10, coup_func, bias_func
        self.output_layer = Denselayer(*layer_params)
        
        self.encoder_order = ['c1', 'pool1']
        self.inference_order = ['c1', 'pool1', 'output_layer']
        
        
    @partial(jax.jit, static_argnames=['self'])
    def distance_function(self, y, target, params):
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(1 - jnp.cos(y[slices]-target))/2.
    
    @partial(jax.jit, static_argnames=['self'])
    def external_energy(self, y, target):
        # y is only the output
        slices = tuple(slice(0, dim) for dim in target.shape)
        return jnp.sum(jax.vmap(self.cost_func, (0,0))(y[slices], target))
    
    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, params):
        return -jax.grad(self.external_energy, argnums=0)(y, target)
    
    def generate_encoder(self, rng, params):
        nn = Generate_Module(self.cost_func)
        encoder_layers = {}
        for name in self.encoder_order:
            encoder_layers.update({name:self.layers[name]})
        new_params = nn.get_init_params(rng, self.input_type, self.F0['input_data'], encoder_layers, self.encoder_order, self.optimizer, params)
        return nn, new_params
    
    def generate_inference_nn(self, rng, params):
        nn = Generate_Module(self.cost_func)
        inf_layers = {}
        for name in self.inference_order:
            inf_layers.update({name:self.layers[name]})
        new_params = nn.get_init_params(rng, self.input_type, self.F0['input_data'], inf_layers, self.inference_order, self.optimizer, params)
        return nn, new_params
        
        
    
class My_nn(Module):
    # This is a template to define a network for inference
    def __init__(self, cost_func, network_type='general XY', structure_name='dnn'):
        #Here network_structure = N, input_index, output_index, layer_sizes
        
        self.network_type = network_type
        self.structure_name = structure_name
        
        self.layer_order = []
        self.layers = {}
        self.params = {}

        self.cost_func = cost_func
        
    def setup(self):
        def coup_func(x, y): return -jnp.cos(x-y)
        def bias_func(x, bias_params): return - bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'
        
        #layer_params = 10, coup_func, bias_func
        #self.l1 = denselayer(*layer_params)
        
        output_channel, filter_shape, strides = 3, [4,4], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(output_channel, filter_shape, strides, coup_func, bias_func)
        
        
        output_channel, filter_shape, strides = 2, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(output_channel, filter_shape, strides, coup_func, bias_func)
        
        
        layer_params = 10, coup_func, bias_func
        self.l1 = Denselayer(*layer_params)
        
        self.layer_order = ['c1','pool1', 'l1']
        self.output_name = self.layer_order[-1]
        
        
class My_AE(Autoencoder):
    '''
    This defines an autoencoder
    '''
    def setup(self):
        # One can redefine this function
        def coup_func(x, y): return -jnp.cos(x-y)
        def bias_func(x, bias_params): return - bias_params[0] * jnp.cos(x - bias_params[1])
        self.input_type = 'conv2D'
        
        output_channel, filter_shape, strides = 5, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.c1 = Conv2D(*layer_params)
        
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.pool1 = Pool2D(*layer_params)
        
        output_channel, filter_shape, strides = 5, [3,3], [3,3]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.u1 = Unsample2D(*layer_params)
        
        output_channel, filter_shape, strides = 1, [2,2], [2,2]
        layer_params = output_channel, filter_shape, strides, coup_func, bias_func
        self.tc1 = Unsample2D(*layer_params)
        
        self.layer_order = ['c1', 'pool1', 'u1', 'tc1']
        self.output_name = self.layer_order[-1]
