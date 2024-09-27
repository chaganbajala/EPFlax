import numpy as np
import jax
import jax.numpy as jnp

import EPFlax.EPFlat.model as fm

class Network(fm.Network):
    def set_reg(self, reg=0.1):
        self.reg = reg
        
    def params_norm(self, params):
        return 0.
    
    def regularizer(self, params):
        return 0.
    
class General_XY_Network(fm.General_XY_Network, Network):
    def params_norm(self, params):
        return jnp.sum(params[0] * params[0])
    
    def regularizer(self, params):
        return self.reg * self.params_norm(params)
    

class Layered_General_XY_Network(fm.Layered_General_XY_Network, Network):
    def params_norm(self, W):
        return jnp.sum(W * W)
    
    def regularizer(self, params):
        return self.reg * sum([self.params_norm(W) for W in params[0]])