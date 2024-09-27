import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import time
import pickle
import gc
# Code for training
from functools import partial

class EP_grad:
    '''
    This implement equilibrium propagation training task which deals with data with vmap. 
    '''
    def __init__(self, grad_params, sample_args):
        self.beta, self.runtime, self.rtol, self.atol = grad_params
        self.sample_method, self.batch_size, self.M_init = sample_args
        if self.sample_method == 'full':
            self.grad_func = self.full_gradient
        elif self.sample_method == 'mini_batch': 
            self.grad_func = self.mini_batch_gradient
        elif self.sample_method == 'random_init_mini_batch':
            self.grad_func = self.radnom_init_mini_batch_gradient
    
    def devided_by_beta(self, x):
        return x/self.beta
    
    def full_gradient(self, input_data, target, nn, network_params, *args):

        y0 = nn.get_initial_state(input_data)
        cost, params_g = self.get_params_gradient(y0, target, nn, network_params)
        del y0
        return cost, params_g
    
    def mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, *args):

        y0, running_target = nn.get_initial_state_mini_batch(input_data, target, batch_size)
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g
    
    
    def radnom_init_mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, M_init):

        y0, running_target = nn.get_multiple_init_initial_state(input_data, target, batch_size, M_init)            
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g
    
    def get_params_gradient(self, y0, target, nn, network_params):
        # Get free equilibrium
        #run_func = jax.jit(lambda y0, target, beta: self.run_network(y0, target, nn, network_params, beta))
        #free_equi = run_func(y0, target, 0)
        #nudge_equi = run_func(free_equi, target, beta)
        N_data = y0.shape[0]
        N_params = sum(jax.tree_util.tree_leaves(jax.tree_map(jnp.size, network_params)))
        
        mean_fun = lambda x: jnp.mean(x, axis=0)
        sum_func = lambda x: jnp.sum(x, axis=0)
        mean_divide_func = lambda x: jnp.divide(x, N_data)
        zero_func = lambda x: jnp.multiply(x, 0)
        
        t0 = time.time()
        free_equi = nn.thermalize_network(y0, target, 0., network_params)
        t1 = time.time()
        
        nudge_equi = nn.thermalize_network(free_equi, target, self.beta, network_params)
        
        
        dEdParams_free = nn.params_derivative(free_equi[0,:], network_params)
        dEdParams_nudge = nn.params_derivative(nudge_equi[0,:], network_params)

        for k in range(1, N_data):
            dEdParams_free = jax.tree_map(jnp.add, dEdParams_free, nn.params_derivative(free_equi[k,:], network_params))
            dEdParams_nudge = jax.tree_map(jnp.add, dEdParams_nudge, nn.params_derivative(nudge_equi[k,:], network_params))
        
        mean_func = lambda x: jnp.divide(x, N_data)
        mean_dEdParams_free = jax.tree_map(mean_func, dEdParams_free)
        mean_dEdParams_nudge = jax.tree_map(mean_func, dEdParams_nudge)
        
        gradient = jax.tree_map(jnp.subtract, mean_dEdParams_nudge, mean_dEdParams_free)
        
        cost = jax.vmap(nn.distance_function, (0, 0, None))(free_equi, target, network_params)
        del free_equi, nudge_equi
        
        return jnp.mean(cost), jax.tree_map(self.devided_by_beta, gradient)
    
class Reg_EP_grad(EP_grad):
    def get_params_gradient(self, y0, target, nn, network_params):
        if not hasattr(nn, 'reg'):
            return super().get_params_gradient(y0, target, nn, network_params)
        else:
            cost, gradient = super().get_params_gradient(y0, target, nn, network_params)
            g_reg = jax.grad(nn.regularizer, 0)(network_params)
            return cost, jax.tree_map(jnp.add, gradient, g_reg)
        
    
        