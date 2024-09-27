import numpy as np
import jax
import jax.numpy as jnp
import optax
import diffrax

from functools import partial

class Network:
    '''
    This define a neural network. It should include:
    network_structure: network_type, structure_name, structure_parameter, activation
    __init__: initialization
    get_initial_params(): randomly generate a set of trainable parameter for the neural network
    get_initial_state(input_data): generate a random initial state with specific input_data
    internal_energy(y, network_params): calculate internal energy function
    external_energy(y, target, network_params): calculate external energy
    internal_force(y, network_paarams): 
    external_force(y, network_params):
    params_derivative(self, y, network_params): 
    '''
    
    def __init__(self, opt_params=(1e-3, 1000), run_params=(100, 1e-3, 1e-6), optimizer=None) -> None:
        #print(opt_params, run_params)
        self.tol, self.maxtime = opt_params
        self.runtime, self.rtol, self.atol = run_params
        self.optimizer = optimizer
        N_devices = len(jax.devices())
        if optimizer==None:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_ode
            else:
                self.thermalize_network = self.thermalize_network_ode
        else:
            if N_devices>1:
                self.thermalize_network = self.pmap_thermalize_opt
            else:
                self.thermalize_network = self.thermalize_network_opt
        
    
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
        
        L = len(solution.ts)
        y = solution.ys[L-1,:]
        #del solution

        return y

    
    @partial(jax.jit, static_argnames=['self'])
    def apply_to(self, input_data, params):
        y0 = self.get_initial_state(input_data)
        return jax.vmap(self.single_free_run, (0,None))(y0, params)
    
    #===================Function for getting equilibriums for training by solving ODE=============
    
    @partial(jax.jit, static_argnames=['self'])
    def total_force(self, t, y, target, beta, params):
        # Calculate total force for single piece of data
        #F = self.internal_force(y, params) + beta * self.external_force(y, target, params)
        yt = y[self.network_structure[2]]
        return self.internal_force(y, params) + beta * self.external_force(y, target, params)
    '''
    @partial(jax.jit, static_argnames=['self'])
    def total_force(self, t, y, target, beta, params):
        # Calculate total force for single piece of data
        #F = self.internal_force(y, params) + beta * self.external_force(y, target, params)
        yt = y[self.network_structure[2]]
        return self.internal_force(y, params).at[self.network_structure[1]].add(beta * self.external_force(yt, target, params))
    '''
    
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
        
        L = len(solution.ts)
        y = solution.ys[L-1,:]
        del solution

        return y
    
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
            y_pad = jnp.pad(y, ((0, N_devices-N_data%N_devices),(0,0)), mode='constant')
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


    @partial(jax.jit, static_argnames=['self'])
    def pmap_thermalize_ode(self, y0, target, beta, params):
        devices = jax.devices()
        N_devices = len(devices)
        N_data = y0.shape[0]
        N_rem = N_data % N_devices

        y0_run, target_run = self.pad_data(y0, target, N_devices)

        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        #print(y0_run.shape, params[0].shape)
        y_run = jnp.concatenate(jax.pmap(self.thermalize_network_ode)(y0_run, target_run, beta_run, params_run))
        return y_run[0:N_data,...]

    
    #========Function for getting equilibriums for training by searching with optax========
    
    def total_force_opt(self, y, target, beta, params):
        return -self.total_force(0., y, target, beta, params)
    
    #@partial(jax.jit, static_argnames=['self'])
    def single_run_func_opt(self, y0, target, beta, params):
        # This use Optax to find the equilibrium
        opt_state = self.optimizer.init(y0)
        tries = 0
        absF = 1.
        y = y0
        
        @jax.jit
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
        N_data = y0.shape[0]
        N_rem = N_data % N_devices

        y0_run, target_run = self.pad_data(y0, target, N_devices)

        params_run = self.tree_expand(params, N_devices)
        beta_run = self.tree_expand(beta, N_devices)

        #print(y0_run.shape, params[0].shape)
        y_run = jnp.concatenate(jax.pmap(self.thermalize_network_opt)(y0_run, target_run, beta_run, params_run))
        return y_run[0:N_data,...]
    
class Hopfield_Network(Network):
    '''
    This defines a hopfield-like network with all-to-all connectivity
    '''
    
    
    def __init__(self, network_structure, activation, opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, 
                 network_type='Hopfield', structure_name='all to all', mask=1.):
        super().__init__(opt_params, run_params, optimizer)
        # network_structure = N, input_index, output_index
        self.activation = activation
        self.network_type = network_type
        self.structure_name = structure_name
        self.network_structure = network_structure
        self.d_activation = jax.grad(self.activation, 0)
        self.mask = mask
    
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        W = 1/np.sqrt(N) * np.random.randn(N, N)
        W = (W + np.transpose(W))/2
        
        bias = np.random.randn(N)
        
        return W, bias
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index, activation = self.network_structure
        
        y0 = 2 * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    def get_initial_state_mini_batch(self, input_data, target, batch_size):
        N_data = input_data.shape[0]
        N, input_index, output_index, activation = self.network_structure
        
        data_ind = np.random.randint(0, N_data, batch_size)
        y0 = 2 * (np.random.rand(batch_size, N) - 0.5)
        y0[:, input_index] = input_data[data_ind, :]
        
        return y0, target[data_ind, :]
    
    def internal_energy(self, y, network_params):
        W, bias = network_params
        N = bias.shape[0]
        
        E0 = jnp.dot(y, y)/2
        
        my = jnp.tensordot(self.activation(y), jnp.ones(N), 0)
        E1 = - jnp.tensordot(W, my * jnp.transpose(my))/2
        E2 = - jnp.dot(bias, self.activation(y))
        
        return E0 + E1 + E2

    def internal_force(self, y, network_params):
        W, bias = network_params
        input_index = self.network_structure[1]
        #N = bias.shape[0]
        
        sy = jax.vmap(self.d_activation,(0))(y)
        
        F = y - jnp.dot(jnp.asarray(W), self.activation(y)) * sy - bias * sy
        F = F.at[input_index].set(0)
        
        return -F
    
    
    def distance_function(self, y, target, network_params):
        W, bias = network_params
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = jnp.dot(dy, dy)/2
        return cost
    
    def external_energy(self, y, target, network_params):
        
        W, bias = network_params
        
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        #cost = jnp.sum(jnp.log(1 - jnp.power(dy, 2)))/2
        cost = jnp.sum(dy**2)
        return cost
    
    def external_force(self, y, target, network_params):
        output_index = self.network_structure[2]
        F = jnp.zeros(y.shape)
        #F = F.at[output_index].set(-(y[output_index] - target)/(1 - jnp.power(y[output_index]-target, 2)))
        F = F.at[output_index].set(- y[output_index] + target)
        
        return F
    
    def params_derivative(self, y, network_params):
        N = y.shape[0]
        sy = self.activation(y)
        my = jnp.tensordot(sy, jnp.ones(N), 0)
        return -my * jnp.transpose(my), -sy


class XY_Network(Network):
    '''
    This defince a XY model viewed as an neural network with all-to-all connectivity
    '''
    def __init__(self, network_structure, opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, network_type='XY', structure_name='all to all'):
        #print(opt_params, run_params)
        super().__init__(opt_params, run_params, optimizer)
        
        # network_structure = N, input_index, output_index
        self.network_type = network_type
        self.structure_name = structure_name
        self.network_structure = network_structure
        self.input_mask = jnp.ones(network_structure[0]).at[network_structure[1]].set(0.)
        self.output_mask = jnp.zeros(network_structure[0]).at[network_structure[1]].set(1.)
        
    
    #--------------------Initialization of the network-----------------------
    
    def get_initial_params(self, seed=None):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        if seed==None:
            W = 1/np.sqrt(N) * np.random.randn(N, N)
            W = (W + np.transpose(W))/2
            bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        else:
            rng = jax.random.key(seed)
            W = 1/np.sqrt(N) * jax.random.normal(rng, (N,N))
            W = 0.5*(W + jnp.transpose(W))
            bias = jnp.asarray([0*jax.random.normal(rng, (N,)), jax.random.uniform(rng, shape=(N,),minval=-jnp.pi, maxval=jnp.pi)])
        
        return W, bias
    
    #--------------------Initialization of states network-----------------------
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure
        
        # the initial state follows a uniform distribution over (-\pi, \pi)
        y0 = 2 * np.pi * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    def get_initial_state_mini_batch(self, input_data, target, batch_size):
        #select a random mini-batch of data from total dataset
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        data_ind = self.get_random_index(N_data, batch_size)
        y0 = 2 * np.pi * (np.random.rand(batch_size, N) - 0.5)
        y0[:, input_index] = input_data[data_ind, :]
        
        return y0, target[data_ind, :]
    
    def get_multiple_init_data(self, input_data, target, M_init, batch_size):
        # prepare folded mini-batch dataset for multiple random initialization
        N_data = input_data.shape[0]
        
        data_ind = self.get_random_index(N_data, batch_size)
        
        mini_input = input_data[data_ind, :]
        mini_target = target[data_ind, :]
        
        batch_input = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_input, 0))
        batch_target = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_target, 0))
        
        return batch_input, batch_target
    
    def get_multiple_init_initial_state(self, input_data, target, batch_size, M_init):
        
        batch_input, batch_target = self.get_multiple_init_data(input_data, target, M_init, batch_size)
        y0 = self.get_initial_state(batch_input)
        return y0, batch_target
    
    #----------------------------Dynamics of the network-------------------------------
    
    def internal_energy(self, y, network_params):
        W, bias = network_params
        N = W.shape[0]
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        E0 = - jnp.tensordot(W, jnp.cos(my - jnp.transpose(my)))/2
        
        E1 = - jnp.dot(bias[0], jnp.cos(y-bias[1]))
        
        return E0 + E1

    def internal_force(self, y, network_params):
        W, bias = network_params
        input_index = self.network_structure[1]
        N = W.shape[0]
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        F = -jnp.sum(W * jnp.sin(my - jnp.transpose(my)), axis=1) - bias[0] * jnp.sin(y - bias[1])
        #F = F.at[input_index].set(0)
        F = F * self.input_mask
        
        return F
    
    def distance_function(self, y, target, network_params):
        W, bias = network_params
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = jnp.sum(1-jnp.cos(dy))/2
        return cost
    
    
    def external_energy(self, y, target, network_params):
        
        W, bias = network_params
        
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = -jnp.sum(jnp.log(1 + jnp.cos(dy)))
        return cost
    '''
    
    def external_energy(self, yt, target, network_params):
        #output_index = self.network_structure[2]
        return -jnp.sum(jnp.log(1 + 1e-5 + jnp.cos(yt - target)))
    '''
    
    def external_force(self, y, target, network_params):
        return -jax.grad(self.external_energy, 0)(y, target, network_params)
    
    def params_derivative(self, y, network_params):
        W, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        dmy = my - jnp.transpose(my)
        
        # calculate dE/dW
        g_W = - jnp.cos(dmy)
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([-jnp.cos(y - bias[1]), -bias[0]*jnp.sin(y - bias[1])])
        return g_W, g_bias
    
class General_XY_Network(XY_Network):
    """
    This define a network implementing energy function of arbitrary two-point interaction, bias and cost function
    coup_func, bias_func: guarantee that it reaches minimum when all the other interactions are screened, e.g., for XY, coup_func = bias_func = -cos(* - *)
    """
        
    def __init__(self, network_structure, coup_func, bias_func, cost_func, opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, 
                 network_type='general XY', structure_name='all to all'):
        #print(opt_params, run_params)
        super().__init__(network_structure, opt_params, run_params, optimizer,
                         network_type, structure_name)
        
        self.coup_func, self.bias_func, self.cost_func = coup_func, bias_func, cost_func
        self.d0coup_func = jax.grad(coup_func, 0)
        self.d0bias_func = jax.grad(bias_func, 0)
        self.d1bias_func = jax.grad(bias_func, 1)
        self.dcost_func = jax.grad(cost_func, 0)
    
    #========================Internal dynamics=========================
    
    def internal_energy(self, y, network_params):
        #W, bias = network_params
        my = jnp.tile(y, (y.shape[0], 1))
        #my = jnp.tensordot(y, jnp.ones(N), 0)
        #m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))
        
        #v_bias_func = jax.vmap(self.bias_func, (0,0))
        
        E0 = jnp.tensordot(network_params[0], jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))(my, my.T))/2
        
        E1 = jnp.dot(network_params[1][0], jax.vmap(self.bias_func, (0,0))(y, network_params[1][1]))
        
        return E0 + E1
    
    def internal_force(self, y, network_params):
        return -jax.grad(self.internal_energy, 0)(y, network_params)*self.input_mask
    
    #========================External dynamics=====================================
    
    def external_energy(self, y, target, network_params):
        #cost = jnp.sum(jax.vmap(self.cost_func, (0,0))(y[self.network_structure[2]], target))
        return jnp.sum(jax.vmap(self.cost_func, (0,0))(y[self.network_structure[2]], target))
    '''
    
    def external_energy(self, yt, target, network_params):
        #cost = jnp.sum(jax.vmap(self.cost_func, (0,0))(y[self.network_structure[2]], target))
        return jnp.sum(jax.vmap(self.cost_func, (0,0))(yt, target))
    '''
    
    def external_force(self, y, target, network_params):
        return -jax.grad(self.external_energy, 0)(y, target, network_params)
    
    def params_derivative(self, y, network_params):
        W, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))
        v_dbias_func = jax.vmap(self.d1bias_func, (0, 0))
        v_bias_func = jax.vmap(self.bias_func, (0, 0))
        
        # calculate dE/dW
        g_W = m_coup_func(my, jnp.transpose(my))
        
        # calculate dE/dh
        g_bias = jnp.asarray([v_bias_func(y, bias[1]), bias[0]*v_dbias_func(y, bias[1])])
        return g_W, g_bias
    
    def get_edges(self, params):
        # In this function we generate the set of edges used for graph
        W, bias = params
        N = self.network_structure[0]
        edges = []
        graph_W = []
        for k in range(0,N):
            for l in range(k+1, N):
                edges.append([k,l])
                graph_W.append(W[k,l])
        
        graph_params = jnp.asarray(graph_W), bias
        
        return jnp.asarray(edges), graph_params
    
class Layered_General_XY_Network(General_XY_Network):
    '''
    This implement a generall XY network with layered structure
    '''
    
    def __init__(self, network_structure, coup_func, bias_func, cost_func, 
                 opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None,
                 network_type='general XY', structure_name='layered'):
        super().__init__(network_structure, coup_func, bias_func, cost_func, opt_params, run_params, optimizer, network_type, structure_name)
        
        #Here network_structure = N, input_index, output_index, layer_sizes
        
        self.split_points = [network_structure[-1][0]]
        for k in range(1, len(network_structure[-1])-1):
            self.split_points.append(self.split_points[-1]+network_structure[-1][k])
        
        index_list = [0]
        for k in range(0, len(network_structure[-1])):
            index_list.append(index_list[-1]+network_structure[-1][k])
        
        self.mask = np.zeros([network_structure[0], network_structure[0]])
        for k in range(0, len(index_list)-2):
            self.mask[index_list[k]:index_list[k+1], index_list[k+1]:index_list[k+2]] = 1
        
        self.mask = jnp.asarray(self.mask + np.transpose(self.mask))
        
        self.layer_shape = jax.tree_map(jnp.zeros, self.network_structure[-1])
        self.structure_shape = jax.tree_map(jnp.zeros, self.split_points)
        self.index_list = index_list
            
    
    #==============================================initial netowrk parameters========================
    def get_initial_params(self, seed=None):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        N_list = self.network_structure[-1]
        depth = len(N_list)
        
        WL = []
        if seed==None:
            for k in range(0, depth-1):
                WL.append( 1/np.sqrt(N_list[k] + N_list[k+1]) * np.random.randn(N_list[k], N_list[k+1]) )
            
            bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        else:
            rng = jax.random.key(seed)
            for k in range(0, depth-1):
                WL.append( 1/np.sqrt(N_list[k] + N_list[k+1]) * jax.random.normal(rng, shape=(N_list[k], N_list[k+1])) )
            
            bias = jnp.asarray([0*jax.random.normal(rng, shape=(N,)) ,  jax.random.uniform(rng, shape=(N,))])
        
        return WL, bias
    
    #========================initial states===========================
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        # the initial state follows a uniform distribution over (-\pi, \pi)
        y0 = 2 * np.pi * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return jnp.asarray(y0)
    
    #======================calculate interactions between neighbor layers=================
    
    def adjacent_energy(self, y1, W, y2):
        # This calculate \sum W_ij * f(y1_i, y2_j)
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tile(y1, (N2, 1))
        my2 = jnp.tile(y2, (N1, 1))
        #my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        #my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        return jnp.tensordot(W, jax.vmap(jax.vmap(self.coup_func, (0,0)), (0,0))(my1.T, my2))
    
    
    def adjacent_forces(self, y1, W, y2):
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tile(y1, (N2, 1))
        my2 = jnp.tile(y2, (N1, 1))
        #my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        #my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        # This force acts on y2
        forward_force = jnp.sum(W * jax.vmap(jax.vmap(self.d0coup_func, (0,0)), (0,0))(my1.T, my2), axis=0)
        
        # This force acts on y2
        backward_force = -jnp.sum(W * jax.vmap(jax.vmap(self.d0coup_func, (0,0)), (0,0))(my1.T, my2), axis=1)
        
        return forward_force, backward_force
    
    def internal_energy(self, y, network_params):
        
        WL, bias = network_params
        N = bias.shape[1]
        layer_sizes = self.network_structure[-1]
        
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))  
        v_bias_func = jax.vmap(self.bias_func, (0,0))
        
        yl = jnp.split(y, self.split_points)
        yl1 = yl.copy()
        yl2 = yl.copy()
        
        del yl1[-1]
        del yl2[0]
        
        E0 = jnp.sum(jnp.asarray(jax.tree_map(self.adjacent_energy, yl1, WL, yl2)))
        
        E1 = jnp.dot(bias[0], v_bias_func(y, bias[1]))
        
        return E0+E1
    
    @partial(jax.jit, static_argnames=['self'])
    def internal_force(self, y, network_params):
        #input_index = self.network_structure[1]
        return -jax.grad(self.internal_energy, 0)(y, network_params)*self.input_mask
        #return -jax.grad(self.internal_energy, 0)(y, network_params).at[input_index].set(0.)
    
    #=================Calculate dynamical terms from external energy============
    # This part is not interferred by the layer atchitecture and is therefore not necessary to change
    
    #=================Calculate parameter derivatives==============
    def W_derivative(self, y1, y2):
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        return jax.vmap(jax.vmap(self.coup_func, (0,0)), (0,0))(my1, my2)
    
    @partial(jax.jit, static_argnames=['self'])
    def params_derivative(self, y, network_params):
        WL, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        v_dbias_func = jax.vmap(self.d1bias_func, (0, 0))
        v_bias_func = jax.vmap(self.bias_func, (0, 0))
        
        # calculate dE/dW
        layer_sizes = self.network_structure[-1]
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        g_W = jax.tree_map(self.W_derivative, yl1, yl2)
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([v_bias_func(y, bias[1]), bias[0]*v_dbias_func(y, bias[1])])
        return g_W, g_bias
    
    #=====================Correlate layered network to a all-to-all network=================
    def merge_params(self, WL, bias):
        N = bias.shape[1]
        depth = len(self.index_list) - 1
        
        W = np.zeros([N, N])
        
        for n in range(0, depth-1):
            W[self.index_list[n]:self.index_list[n+1], self.index_list[n+1]:self.index_list[n+2]] = WL[n]
        
        W = W + np.transpose(W)
        
        return W, bias
    
    def get_edges(self, params):
        WL, bias = params
        layer_origin = self.split_points.copy()
        layer_origin.insert(0, 0)
        edges = []
        graph_params = []
        for k in range(0, len(self.index_list)-2):
            for ind1 in range(self.index_list[k], self.index_list[k+1]):
                for ind2 in range(self.index_list[k+1], self.index_list[k+2]):
                    edges.append([ind1, ind2])
                    graph_params.append(WL[k][ind1-self.index_list[k], ind2-self.index_list[k+1]])
        
        graph_params = jnp.asarray(graph_params), bias
        
        return jnp.asarray(edges), graph_params
                    
            

class Layered_Hopfield_Network(Hopfield_Network):
    '''
    This define a general network of layered structure.
    The network energy function has 2-body interaction, local bias and an external energy. 
    '''
    
    def __init__(self, network_structure, activation, cost_func=None, 
                 opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, 
                 network_type='Hopfield', structure_name='layered'):
        super().__init__(network_structure, activation, opt_params, run_params, optimizer)
        
        
        # network_structure = N, input_index, output_index, layer_sizes
        self.network_type = network_type
        self.structure_name = structure_name
        self.network_structure = network_structure
        self.activation = activation
        self.d_activation = jax.grad(self.activation, 0)
        
        self.cost_func = cost_func
        self.dcost_func = jax.grad(cost_func, 0)
        
        self.v_activation = jax.vmap(self.activation, 0)
        self.vd_activation = jax.vmap(self.d_activation, 0)
        
        self.split_points = [network_structure[-1][0]]
        for k in range(1, len(network_structure[-1])-1):
            self.split_points.append(self.split_points[-1]+network_structure[-1][k])
        
        index_list = [0]
        for k in range(0, len(network_structure[-1])):
            index_list.append(index_list[-1]+network_structure[-1][k])
        
        self.mask = np.zeros([network_structure[0], network_structure[0]])
        for k in range(0, len(index_list)-2):
            self.mask[index_list[k]:index_list[k+1], index_list[k+1]:index_list[k+2]] = 1
        
        self.mask = self.mask + np.transpose(self.mask)
        
        self.layer_shape = jax.tree_map(jnp.zeros, self.network_structure[-1])
        self.structure_shape = jax.tree_map(jnp.zeros, self.split_points)
        self.index_list = index_list
    
    #=========================initialize network=====================
    
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        N_list = self.network_structure[-1]
        depth = len(N_list)
        
        WL = []
        for k in range(0, depth-1):
            WL.append( 1/np.sqrt(N_list[k] + N_list[k+1]) * np.random.randn(N_list[k], N_list[k+1]) )
        
        bias = np.asarray(np.random.randn(N))
        
        return WL, bias
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        # the initial state follows a uniform distribution over (-\pi, \pi)
        y0 = 2 * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    def get_initial_state_mini_batch(self, input_data, target, batch_size):
        #select a random mini-batch of data from total dataset
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        data_ind = self.get_random_index(N_data, batch_size)
        y0 = 2 * np.pi * (np.random.rand(batch_size, N) - 0.5)
        y0[:, input_index] = input_data[data_ind, :]
        
        return y0, target[data_ind, :]
    
    def get_multiple_init_data(self, input_data, target, M_init, batch_size):
        # prepare folded mini-batch dataset for multiple random initialization
        N_data = input_data.shape[0]
        
        data_ind = self.get_random_index(N_data, batch_size)
        
        mini_input = input_data[data_ind, :]
        mini_target = target[data_ind, :]
        
        batch_input = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_input, 0))
        batch_target = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_target, 0))
        
        return batch_input, batch_target
    
    def get_multiple_init_initial_state(self, input_data, target, batch_size, M_init):
        
        batch_input, batch_target = self.get_multiple_init_data(input_data, target, M_init, batch_size)
        y0 = self.get_initial_state(batch_input)
        return y0, batch_target
    
    #==================calculate internal energy=======================
    def adjacent_energy(self, y1, W, y2):
        # This calculate \sum W_ij * r(y1_i)*r(y2_j)
        return -jnp.dot(self.activation(y1), jnp.dot(W, self.activation(y2)))
    
    def adjacent_forces(self, y1, W, y2):
        forward_force = self.vd_activation(y2) * jnp.dot(self.v_activation(y1), W)
        backward_force = self.vd_activation(y1) * jnp.dot(W, self.v_activation(y2))
        return forward_force, backward_force
    
    def internal_energy(self, y, network_params):
        WL, bias = network_params
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]

        E0 = 0.5 * jnp.dot(y, y)        
        E1 = jnp.sum(jnp.asarray(jax.tree_map(self.adjacent_energy, yl1, WL, yl2)))
        
        E2 = -jnp.dot(bias, self.v_activation(y))
        
        return E0 + E1 + E2
    
    def internal_force(self, y, network_params):
        WL, bias = network_params
        N = y.shape[0]
        input_index = self.network_structure[1]
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        res = jax.tree_map(self.adjacent_forces, yl1, WL, yl2)
        ff = jnp.concatenate(list(zip(*res))[0])
        bf = jnp.concatenate(list(zip(*res))[1])
        
        F1 = jnp.zeros(N)
        F2 = jnp.zeros(N)
        
        #back and forward force from 2-body interaction
        F1 = F1.at[jnp.arange(self.split_points[0], N)].set(ff)
        F2 = F2.at[jnp.arange(0, self.split_points[-1])].set(bf)
        
        #force from bias
        F3 = bias * self.vd_activation(y)
        
        F = -y + F1 + F2 + F3
        F = F.at[input_index].set(0)
        return F
    
    def W_derivative(self, y1, y2):
        return - jnp.tensordot(self.v_activation(y1), self.v_activation(y2), 0)
    
    @partial(jax.jit, static_argnames=['self'])
    def params_derivative(self, y, network_params):
        WL, bias = network_params
        
        # calculate dE/dW
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        g_W = jax.tree_map(self.W_derivative, yl1, yl2)
        
        # calculate dE/dh
        
        g_bias = - self.v_activation(y)
        return g_W, g_bias
    #=====================External energy and force=====================
    def external_energy(self, y, target, network_params):
        output_index = self.network_structure[2]
        cost = jnp.sum(jax.vmap(self.cost_func, (0,0))(y[output_index], target))
        return cost
    
    def external_force(self, y, target, network_params):
        output_index = self.network_structure[2]
        F = jnp.zeros(y.shape)
        F = F.at[output_index].set(-jax.vmap(self.dcost_func, (0,0))(y[output_index], target))
        return F
    
    #=============Generate an equivalent all-to-all network==============
    def merge_params(self, WL, bias):
        N = bias.shape[0]
        depth = len(self.index_list) - 1
        
        W = np.zeros([N, N])
        
        for n in range(0, depth-1):
            W[self.index_list[n]:self.index_list[n+1], self.index_list[n+1]:self.index_list[n+2]] = WL[n]
        
        W = W + np.transpose(W)
        
        return W, bias
    
    def generate_network(self, layer_network_params):
        network_params = self.merge_params(*layer_network_params)
        network_structure = self.network_structure[0], self.network_structure[1], self.network_structure[2]
        return Hopfield_Network(network_structure, self.activation, mask=self.mask), network_params
    
    
class Graph_Network(General_XY_Network):
    def __init__(self, network_structure, coup_func, bias_func, cost_func, 
                 opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, 
                 network_type='general XY', structure_name='graph', edges=jnp.asarray([])):
        super().__init__(network_structure, coup_func, bias_func, cost_func, 
                         opt_params, run_params, optimizer, 
                         network_type, structure_name)
        # edges is the collection of edges in the graph
        self.edges = edges
        self.v_coup = jax.vmap(self.coup_func, (0,0))
    
    def get_initial_params(self):
        N = self.network_structure[0]
        bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        couplings = np.random.randn(self.edges.shape[0]) / np.sqrt(N)
        return couplings, bias
    
    def single_coupling_energy(self, y, ind1, ind2):
        return self.coup_func(y[ind1], y[ind2])
    
    def coupling(self, y):
        ind1, ind2 = tuple(jnp.transpose(self.edges))
        return self.v_coup(y[ind1], y[ind2])
        #return jax.vmap(self.single_coupling_energy, (None, 0, 0))(y, ind1, ind2)
    
    def internal_energy(self, y, network_params):
        W, bias = network_params
        
        E0 = jnp.sum(W*self.coupling(y))
        
        E1 = - jnp.dot(bias[0], jnp.cos(y-bias[1]))
        
        return E0 + E1
    
    @partial(jax.jit, static_argnames=['self'])
    def internal_force(self, y, network_params):
        F = - jax.grad(self.internal_energy, 0)(y, network_params)
        return jnp.multiply(F, self.input_mask)
    
    def convert_to_matrix(self, network_params):
        W, bias = network_params
        edges_list = self.edges
        N = self.network_structure[0]
        matrix_W = jnp.zeros([N,N])
        matrix_W = matrix_W.at[edges_list[:,0], edges_list[:,1]].set(W)
        matrix_W = matrix_W + np.transpose(matrix_W)
        
        return matrix_W, bias 
    
    @partial(jax.jit, static_argnames=['self'])
    def params_derivative(self, y, network_params):
        return jax.grad(self.internal_energy, argnums=1)(y, network_params)
    
    #===========Functions to modify the structure after define it==========
    
    def add_egdes(self, new_edges, params = None):
        self.edges = jnp.concatenate((self.edges, new_edges), axis=0)
        if params==None:
            return self.get_initial_params()
        else: 
            couplings, bias = params
            couplings = jnp.concatenate((couplings, np.random.randn(new_edges.shape[0])), axis=0)
            
        return couplings, bias
    
class Square_Lattice(General_XY_Network):
        
    def __init__(self, network_structure, coup_func, bias_func, cost_func, 
                 opt_params=(0.001, 1000), run_params=(100, 0.001, 0.000001), optimizer=None, 
                 network_type='general XY', structure_name='suqare lattice', translation = jnp.asarray([[0,1],[1,0]])):
        super().__init__(network_structure, coup_func, bias_func, cost_func, 
                         opt_params, run_params, optimizer, 
                         network_type, structure_name)
        # edges is the collection of edges in the graph
        # network_structure = N, input_index, output_index, lattice_shape
        self.translation = tuple(translation)
        self.v_coup = jax.vmap(self.coup_func, (0, 0))
        
        self.m_coup = jax.vmap(self.v_coup, (0, 0))
        
        self.input_mask = jnp.ones(network_structure[-1])
        self.input_mask = self.input_mask.at[network_structure[1][:,0], network_structure[1][:,1]].set(0.)
        
    def get_initial_params(self):
        y_test = jnp.zeros(self.network_structure[3])
        coupling_shape = self.coupling(y_test)
        
        def get_random_params(coupling_shape):
            return np.random.randn(*coupling_shape.shape)/np.sqrt(coupling_shape.size)
        
        W = jax.tree_map(get_random_params, coupling_shape)
        bias = jnp.asarray([jnp.zeros(self.network_structure[3]), (np.random.rand(*self.network_structure[3]) - 0.5)*2*np.pi])
        
        return W, bias
    
    def get_initial_state(self, input_data):
        N_data = input_data.shape[0]
        input_index = jnp.asarray(self.network_structure[1])
        y0 = jnp.asarray((np.random.rand(N_data, *self.network_structure[3]) - 0.5)*2*np.pi)
        y0 = y0.at[:,input_index[:,0], input_index[:,1]].set(input_data)
        return y0
    
    
    def coupling(self, y):
        # This calculate the couplings of y with clarified lattice structure
        # y: configuration of the system
        # v: specific corellation
        N_row, N_col = self.network_structure[3]
        coupling_left = self.m_coup(y, jnp.roll(y,shift=[0,-1], axis=(0,1)))[:, 0:N_col-1]
        coupling_up = self.m_coup(y, jnp.roll(y,shift=[-1,0], axis=(0,1)))[0:N_row-1, :]
        return coupling_left, coupling_up
    
    @partial(jax.jit, static_argnames=['self'])
    def internal_energy(self, y, network_params):
        W, bias = network_params
        
        E0 = sum(jax.tree_map(jnp.sum, jax.tree_map(jnp.multiply, W, self.coupling(y))))
        
        E1 = - jnp.sum(bias[0] * jnp.cos(y-bias[1]))
        
        return E0 + E1
    
    @partial(jax.jit, static_argnames=['self'])
    def internal_force(self, y, network_params):
        F = - jax.grad(self.internal_energy, 0)(y, network_params)
        return F * self.input_mask
    
    def distance_function(self, y, target, network_params):
        W, bias = network_params
        output_index = self.network_structure[2]
        dy = y[output_index[:,0], output_index[:,1]] - target
        cost = jnp.sum(1-jnp.cos(dy))/2
        return cost
    
    def external_energy(self, y, target, network_params):
        
        #W, bias = network_params
        
        output_index = self.network_structure[2]
        cost = jnp.sum(jax.vmap(self.cost_func, (0,0))(y[output_index[:,0], output_index[:,1]], target))
        return cost
    
    @partial(jax.jit, static_argnames=['self'])
    def external_force(self, y, target, network_params):
        return - jax.grad(self.external_energy, 0)(y, target, network_params)
    
    @partial(jax.jit, static_argnames=['self'])
    def params_derivative(self, y, network_params):
        return jax.grad(self.internal_energy, argnums=1)(y, network_params)
    
    def get_graph_params(self, params):
        lattice_shape = self.network_structure[-1]
        W, bias = params
        edges = []
        W_graph = []
        for k in range(0, lattice_shape[0]):
            for l in range(0, lattice_shape[1]-1):
                edges.append([k*lattice_shape[1]+l, k*lattice_shape[1]+l+1])
                W_graph.append(W[0][k,l])
                
        for k in range(0, lattice_shape[0]-1):
            for l in range(0, lattice_shape[1]):
                edges.append([k*lattice_shape[1]+l, (k+1)*lattice_shape[1]+l])
                W_graph.append(W[1][k,l])
                
        bias_graph = bias.reshape(2, lattice_shape[0] * lattice_shape[1])        
        return (jnp.asarray(W_graph), bias_graph), jnp.asarray(edges)
    
    def get_matrix_params(self, params):
        params, edges = self.get_graph_params(params)
        N = self.network_structure[0]
        
        input_index = []
        for k in range(0, self.network_structure[1].shape[0]):
            input_index.append(self.network_structure[1][k,0]*self.network_structure[-1][1] + self.network_structure[1][k,1])
        
        output_index = []  
        for k in range(0, self.network_structure[2].shape[0]):
            output_index.append(self.network_structure[2][k,0]*self.network_structure[-1][1] + self.network_structure[2][k,1])
            
        input_index = jnp.asarray(input_index, dtype=jnp.int32)
        output_index = jnp.asarray(output_index, dtype=jnp.int32)
        
        network_structure = N, input_index, output_index
        graph_nn = Graph_Network(network_structure, self.coup_func, self.bias_func, self.cost_func, edges=edges)
        
        return graph_nn.convert_to_matrix(params)
    
class Triangular_Lattice(Square_Lattice):
    def coupling(self, y):
        N_row, N_col = self.network_structure[3]
        coupling_left = self.m_coup(y, jnp.roll(y,shift=[0,-1], axis=(0,1)))[:, 0:N_col-1]
        coupling_up = self.m_coup(y, jnp.roll(y,shift=[-1,0], axis=(0,1)))[0:N_row-1, :]
        coupling_diag = self.m_coup(y, jnp.roll(y,shift=[-1,-1], axis=(0,1)))[0:N_row-1, 0:N_col-1]
        return coupling_left, coupling_up, coupling_diag
    
    def get_graph_params(self, params):
        lattice_shape = self.network_structure[-1]
        W, bias = params
        edges = []
        W_graph = []
        for k in range(0, lattice_shape[0]):
            for l in range(0, lattice_shape[1]-1):
                edges.append([k*lattice_shape[1]+l, k*lattice_shape[1]+l+1])
                W_graph.append(W[0][k,l])
                
        for k in range(0, lattice_shape[0]-1):
            for l in range(0, lattice_shape[1]):
                edges.append([k*lattice_shape[1]+l, (k+1)*lattice_shape[1]+l])
                W_graph.append(W[1][k,l])
        
        for k in range(0, lattice_shape[0]-1):
            for l in range(0, lattice_shape[1]-1):
                edges.append([k*lattice_shape[1]+l, (k+1)*lattice_shape[1]+l+1])
                W_graph.append(W[2][k,l])
        
        bias_graph = bias.reshape(2, lattice_shape[0] * lattice_shape[1])        
        return (jnp.asarray(W_graph), bias_graph), jnp.asarray(edges)
    
class Hybrid_Layered(Layered_General_XY_Network):
    # This is the layered architecture network allowing assiging intra and cross-layer interaction
    def __init__(self, network_structure, coup_func, bias_func, cost_func, network_type='general XY', structure_name='layered'):
        super().__init__(network_structure, coup_func, bias_func, cost_func, network_type, structure_name)
        self.edges = []
        
    def add_egdes(self, edges):
        # Edges should be either intra-layer or cross-layer coupling
        # Each edge should have the form [layer number, node number]. 
        layer_origin = self.split_points.copy()
        layer_origin.insert(0,0)
        edge_list = []
        for ind1, ind2 in edges:
            print(ind1, ind2, layer_origin)
            edge_list.append([layer_origin[ind1[0]] + ind1[1], layer_origin[ind2[0]] + ind2[1]])
            
        self.edges = jnp.asarray(edge_list)
        
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        N_list = self.network_structure[-1]
        depth = len(N_list)
        
        WL = []
        for k in range(0, depth-1):
            WL.append( 1/np.sqrt(N_list[k] + N_list[k+1]) * np.random.randn(N_list[k], N_list[k+1]) )
        
        bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        
        if self.edges!=[]:
            self.internal_energy = self.hybrid_energy
            W_edges = np.random.randn(self.edges.shape[0]) / np.sqrt(self.network_structure[0])
            return WL, W_edges, bias
        else:
            self.internal_energy = self.layer_energy
            return WL, bias
    
    def graph_coupling(self, y):
        ind1, ind2 = tuple(jnp.transpose(self.edges))
        return jax.vmap(self.coup_func, (0, 0))(y[ind1], y[ind2])
    
    def layer_energy(self, y, network_params):
        
        WL, bias = network_params 
        v_bias_func = jax.vmap(self.bias_func, (0,0))
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        E0 = jnp.sum(jnp.asarray(jax.tree_map(self.adjacent_energy, yl1, WL, yl2)))
        
        E1 = jnp.dot(bias[0], v_bias_func(y, bias[1]))
        
        return E0+E1
    
    def hybrid_energy(self, y, network_params):
        WL, W_edges, bias = network_params
        E_layer = self.layer_energy(y, (WL, bias))
        E_edges = jnp.dot(W_edges, self.graph_coupling(y))
        return E_layer + E_edges
    
    def internal_force(self, y, network_params):
        F = - jax.grad(self.internal_energy, 0)(y, network_params)
        input_index = self.network_structure[1]
        F = F.at[input_index].set(0.)
        return F
        