import numpy as np
import jax
import jax.numpy as jnp
import time
import pickle
import gc
# Code for training

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
        # Get gradient params for multiple inouts:
        N_data = y0['input_data'].shape[0]
        
        free_equi = nn.thermalize_network(y0, target, 0., network_params)
        nudge_equi = nn.thermalize_network(free_equi, target, self.beta, network_params)
        
        free_values = {key: value[0,:] for key, value in free_equi.items()}
        nudge_values = {key: value[0,:] for key, value in nudge_equi.items()}
        
        dEdParams_free = nn.params_derivative(free_values, network_params)
        dEdParams_nudge = nn.params_derivative(nudge_values, network_params)

        for k in range(1, N_data):
            free_values = {key: value[k,:] for key, value in free_equi.items()}
            nudge_values = {key: value[k,:] for key, value in nudge_equi.items()}
            dEdParams_free = jax.tree_map(jnp.add, dEdParams_free, nn.params_derivative(free_values, network_params))
            dEdParams_nudge = jax.tree_map(jnp.add, dEdParams_nudge, nn.params_derivative(nudge_values, network_params))
       
        mean_func = lambda x: jnp.divide(x, N_data)
        mean_dEdParams_free = jax.tree_map(mean_func, dEdParams_free)
        mean_dEdParams_nudge = jax.tree_map(mean_func, dEdParams_nudge)
        
        gradient = jax.tree_map(jnp.subtract, mean_dEdParams_nudge, mean_dEdParams_free)
        
        cost = jax.vmap(nn.distance_function, (0, 0, None))(free_equi[nn.output_name], target, network_params)
        del free_equi, nudge_equi
        
        return jnp.mean(cost), jax.tree_map(lambda x: jnp.divide(x, self.beta), gradient)
    
    
class Reg_EP_grad(EP_grad):
    def get_params_gradient(self, y0, target, nn, network_params):
        if not hasattr(nn, 'reg'):
            return super().get_params_gradient(y0, target, nn, network_params)
        else:
            cost, gradient = super().get_params_gradient(y0, target, nn, network_params)
            g_reg = jax.grad(nn.regularizer, 0)(network_params)
            return cost, jax.tree_map(jnp.add, gradient, g_reg)