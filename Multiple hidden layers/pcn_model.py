from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist

class PCN():
    def __init__(self, dkey, in_dim, out_dim, hidden_dims, T=10,
                 dt=1., tau_m=20., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        self.hidden_dims = hidden_dims
        optim_type = "adam"
        wlb, wub = -0.3, 0.3

        if loadDir is not None:
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z = []
                for i, dim in enumerate([in_dim] + hidden_dims + [out_dim]):
                    if i == 0:  # Input layer
                        self.z.append(RateCell(f"z{i}", n_units=dim, tau_m=0., 
                                               act_fx="identity"))
                    elif i == len(hidden_dims) + 1:  # Output layer
                        self.z.append(RateCell(f"z{i}", n_units=dim, tau_m=0., 
                                               act_fx="identity", 
                                               prior=("gaussian", 0.)))
                    else:  # Hidden layers
                        self.z.append(RateCell(f"z{i}", n_units=dim, tau_m=tau_m, 
                                               act_fx=act_fx,
                                               prior=("gaussian", 0.), 
                                               integration_type="euler"))
                
                self.e = [ErrorCell(f"e{i+1}", n_units=dim) 
                          for i, dim in enumerate(hidden_dims + [out_dim])]
                

                self.W = [HebbianSynapse(f"W{i+1}", shape=(prev_dim, dim), eta=eta,
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         bias_init=dist.constant(value=0.), w_bound=0.,
                                         optim_type=optim_type, sign_value=-1.,
                                         key=subkeys[i])
                          for i, (prev_dim, dim) in enumerate(zip([in_dim] + hidden_dims, 
                                                                  hidden_dims + [out_dim]))]
             
                self.E = [StaticSynapse(f"E{i+2}", shape=(next_dim, dim), 
                                        weight_init=dist.uniform(amin=wlb, amax=wub),
                                        key=subkeys[i])
                          for i, (dim, next_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:]))]
                self.E.append(StaticSynapse(f"E{len(hidden_dims)+1}", shape=(out_dim, hidden_dims[-1]), 
                                            weight_init=dist.uniform(amin=wlb, amax=wub),
                                            key=subkeys[len(hidden_dims)]))

                # Wire up the network
                for i in range(len(self.W)):
                    self.W[i].inputs << self.z[i].zF
                    self.e[i].mu << self.W[i].outputs
                    self.e[i].target << self.z[i+1].z

                for i in range(len(self.E)):
                    self.E[i].inputs << self.e[i+1].dmu
                    self.z[i+1].j << self.E[i].outputs
                    self.z[i+1].j_td << self.e[i].dtarget

                for i in range(len(self.W)):
                    self.W[i].pre << self.z[i].zF
                    self.W[i].post << self.e[i].dmu


                #  # Reset command to reset the entire network's key components
                # reset_cmd, reset_args = self.circuit.compile_by_key(
                #     self.q[0], self.q[1], self.q[2], self.q[3], self.eq[-1],  # RateCells (q) and ErrorCells (eq)
                #     self.z[0], self.z[1], self.z[2], self.z[3],  # RateCells (z)
                #     self.e[0], self.e[1], self.e[2],  # ErrorCells (e)
                #     compile_key="reset"
                # )
                

             
                # Project command to project the state after advancing
                # project_cmd, project_args = self.circuit.compile_by_key(
                #     self.q[0], self.Q[0], self.q[1], self.Q[1],  # RateCells (q) and projection (Q)
                #     self.q[2], self.Q[2], self.q[3], self.eq[-1],  # RateCells and ErrorCells (eq)
                #     compile_key="advance_state", name="project"
                # )

                

                # Inference model
                self.q = [RateCell(f"q{i}", n_units=dim, tau_m=0., act_fx="identity" if i == 0 else act_fx)
                          for i, dim in enumerate([in_dim] + hidden_dims + [out_dim])]
                self.eq = ErrorCell("eq", n_units=out_dim)
                # self.eq = [ErrorCell(f"eq{i}", n_units=out_dim) for i in range(num_error_cells)]
                self.Q = [StaticSynapse(f"Q{i+1}", shape=(prev_dim, dim),
                                        bias_init=dist.constant(value=0.),
                                        key=subkeys[i])
                          for i, (prev_dim, dim) in enumerate(zip([in_dim] + hidden_dims, 
                                                                  hidden_dims + [out_dim]))]
              

                for i in range(len(self.Q)):
                    self.Q[i].inputs << self.q[i].zF
                    self.q[i+1].j << self.Q[i].outputs

                self.eq.target << self.q[-1].z
                print("++ ",[a.name for a in [*self.q,self.eq,*self.z,*self.e]])
                
                print("++ ",[a.name for a in [self.q[0], self.q[1], self.q[2], self.q[3], self.eq,
                                                self.z[0], self.z[1], self.z[2], self.z[3],
                                                self.e[0], self.e[1], self.e[2],]])

                # Reset command to reset the entire network's key components
                reset_cmd, reset_args = self.circuit.compile_by_key( 
                    *self.q,self.eq,*self.z,*self.e,
                    # self.q[0], self.q[1], self.q[2], self.q[3], self.eq,
                    #                             self.z[0], self.z[1], self.z[2], self.z[3],
                    #                             self.e[0], self.e[1], self.e[2],
                    compile_key="reset"
                )
                
                # Advance command for the E-step, advancing network states
                advance_cmd, advance_args = self.circuit.compile_by_key(
                    *self.E,*self.z,*self.e,*self.W,
                    compile_key="advance_state"
                )
                
                # Evolve command for the M-step, evolving weights
                evolve_cmd, evolve_args = self.circuit.compile_by_key(
                    *self.W,
                    compile_key="evolve"
                )
      

                # Project command to project the state after advancing
                print("q ==================  ",[a.name for a in self.q])

                interleaved_args = [item for pair in zip(self.q, self.Q) for item in pair]+ [self.q[-1]]

                print("q ==================  ",[a.name for a in self.q])

                
                
                   # Project command to project the state after advancing
                project_cmd, project_args = self.circuit.compile_by_key(
                    *interleaved_args,self.eq,
                                compile_key="advance_state", name="project"
                )
                
                self.dynamic()


    def dynamic(self):
        vars = (self.q + self.Q + self.z + self.e + self.W + self.E)
        self.nodes = vars

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")
        self.circuit.add_command(wrap_command(jit(self.circuit.project)), name="project")

        @Context.dynamicCommand
        def clamp_input(x):
            self.z[0].j.set(x)
            self.q[0].j.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            self.z[-1].j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.eq.target.set(y)

    def save_to_disk(self, params_only=False):
        if params_only:
            model_dir = f"{self.exp_dir}/{self.model_name}/custom"
            for W in self.W:
                W.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory):
        print(f" > Loading model from {model_directory}")
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory)
            self.dynamic()

    def process(self, obs, lab, adapt_synapses=True):
        # print("Input Shape Passed to clamp_input:", obs.shape)
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        # print("\nBefore reset:", self.z[0].z.value.mean(), self.q[0].z.value.mean())
        self.circuit.reset()
        # print("After reset:", self.z[0].z.value.mean(), self.q[0].z.value.mean())


        for i in range(len(self.Q)):
            self.Q[i].weights.set(self.W[i].weights.value)
            self.Q[i].biases.set(self.W[i].biases.value)
        for i in range(len(self.E)):
            self.E[i].weights.set(jnp.transpose(self.W[i+1].weights.value))

        # print("Observed input to the model:", obs)
        # print("Observed input mean: ", jnp.mean(obs))

        self.circuit.clamp_input(obs)
        # print("input mean: ", jnp.mean(self.z[0].j.value))
        self.circuit.clamp_infer_target(_lab)

        # print("Before project:", self.q[-1].z.value.mean())
        # print("length of gausian error cells ",len(self.e))
        self.circuit.project(t=0., dt=1.)
        # print("After project:", self.q[-1].z.value.mean())

        for i in range(1, len(self.z) - 1):
            self.z[i].z.set(self.q[i].z.value)
        self.e[-1].dmu.set(self.eq.dmu.value)
        self.e[-1].dtarget.set(self.eq.dtarget.value)
        y_mu_inf = self.q[-1].z.value

        EFE = 0.
        y_mu = 0.
        if adapt_synapses:
            for ts in range(self.T):
                self.circuit.clamp_input(obs)
                # print("input mean: ", jnp.mean(self.z[0].j.value))

                self.circuit.clamp_target(_lab)
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.e[-1].mu.value

            # print("\n Log before EFE : ", EFE)
            EFE = sum(e.L.value for e in self.e)
            # Example of summing EFE over all error cells
            # EFE = sum(e.L.value for e in self.e + [self.eq])
            # for i, e in enumerate(self.e):
            #     print(f"Layer {i} dimensions: {e.n_units} L.value: {e.L.value}")


            # print("Log after EFE : ", EFE)

            if adapt_synapses:
                # for i, synapse in enumerate(self.W):
                #     print(f"Weights before evolve for W{i+1}: {synapse.weights.value}")
                self.circuit.evolve(t=self.T, dt=1.)
                # for i, synapse in enumerate(self.W):
                #     print(f"Weights after evolve for W{i+1}: {synapse.weights.value}")


        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        return self.q[-2].z.value


    def _get_norm_string(self):
        norms = [f"W{i+1}: {jnp.linalg.norm(W.weights.value)} b{i+1}: {jnp.linalg.norm(W.biases.value)}"
                 for i, W in enumerate(self.W)]
        return " ".join(norms)
