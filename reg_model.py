import jax
import jax.numpy as jnp

import numpy as np
import EPFlax.model as lm

class Layer(lm.Layer):
    def params_norm(self, params):
        return jnp.sum(jnp.square(params['weights']))
    
class Denselayer(lm.Denselayer, Layer):
    def params_norm(self, params):
        return jnp.sum(jnp.square(params['weights']))
    
class Conv1D(lm.Conv1D, Denselayer):
    def params_norm(self, params):
        return jnp.sum(jnp.square(params['kernel']))

class Conv2D(lm.Conv1D, Denselayer):
    def params_norm(self, params):
        return jnp.sum(jnp.square(params['kernel']))

#================================= Network Modules ============================
class Module(lm.Module):
    
    def __init__(self, cost_func, run_params=(100, 1e-3, 1e-6), opt_params=(1e-3, 1000), optimizer=None, network_type='general XY', structure_name='dnn', reg=0.1):
        super().__init__(cost_func, run_params, opt_params, optimizer, network_type, structure_name)
        self.reg = reg
    
    def regularizer(self, params):
        E = 0.
        for name in self.layer_order:
            layer = self.layers[name]
            E += self.reg * layer.params_norm(params[name])
        return E