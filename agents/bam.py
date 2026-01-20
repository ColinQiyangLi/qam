import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

from functools import partial
from typing import Any

class BAMAgent(flax.struct.PyTreeNode):
    """Basic adjoint matching.

    This agent uses memoryless SDE with BPTT to train the flow policy to maximize Q
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        rng, sample_rng = jax.random.split(rng)
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
    
    @partial(jax.jit, static_argnames=("flow_steps"))
    def compute_sde(self, obs, rng, grad_params, flow_steps=None):
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        h = 1 / flow_steps

        bc_loss = 0
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config["residual"]:
                    v_res = self.network.select("actor_fast")(obs, x, t, params=grad_params) # bptt by passing in the grad_params
                    v = v_res + actor_slow(obs, x, t)
                    bc_loss += jnp.square(v_res).sum(axis=-1) * 2 / (sigma ** 2)
                else:
                    v = self.network.select("actor_fast")(obs, x, t, params=grad_params)     # bptt by passing in the grad_params
                    v_slow = actor_slow(obs, x, t)
                    bc_loss += jnp.square(v - v_slow).sum(axis=-1) * 2 / (sigma ** 2)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:
                x = x + h * actor_slow(obs, x, t)

        return x, bc_loss.mean()

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, bam_rng = jax.random.split(rng, 4)

        # BC flow-matching loss for the slow actor.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss
        
        # BAM loss for the fast actor (backproping through the sde)
        x, bc_loss = self.compute_sde(batch["observations"], rng=bam_rng, grad_params=grad_params)
        qs = self.network.select("critic")(batch["observations"], x)
        q = (qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)).mean()
        bam_loss = -q * self.config["inv_temp"] + bc_loss

        return actor_loss + bam_loss, {'flow_loss': flow_loss, "bam_loss": bam_loss, "bc_loss": bc_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params


    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng,
    ):
        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        if self.config["inv_temp"] == 0.: # sample from bc directly if inv_temp = 0
            actions = self.compute_flow_actions(observations, noises, model="slow")
        else:
            actions = self.compute_flow_actions(observations, noises, model="slow,fast" if self.config["residual"] else "fast")
        actions = jnp.clip(actions, -1, 1)

        # best-of-n    
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @partial(jax.jit, static_argnames="model")
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
    ):
        actions = noises
        networks = [self.network.select(f'actor_{m}') for m in model.split(",")]
        
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])
            
            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )
        
        # Setup networks
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),  # clip grad norm
                optax.adam(learning_rate=config["lr"]))
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bam',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).
            
            ## Common hyperparamters
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,
            
            ## Q-chunking hyperparameters
            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set
            action_chunking=False,                                      # Use Q-chunking or just n-step return
            
            ## RL hyperparameters
            num_qs=10,       # Critic ensemble size
            rho=0.5,        # Pessimistic backup

            discount=0.99,  # Discount factor.
            tau=0.005,      # Target network update rate.
            flow_steps=10,  # Number of flow steps.
            
            best_of_n=1,    # Best-of-n for computing Q-targets and sampling actions.
            
            ## Main hyperparameter(s)
            inv_temp=0.3,
            
            ## Other design variants
            target_actor=True,  # Use target network for the base policy, pi_beta
            residual=False,     # Learn the residual velocity field v_theta = v - v_beta
        )
    )
    return config
