import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value

class ReBRACAgent(flax.struct.PyTreeNode):
    """Revisited behavior-regularized actor-critic (ReBRAC) agent.

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
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
        next_dist = self.network.select('target_actor')(batch['next_observations'][..., -1, :])
        next_actions = next_dist.mode()
        noise = jnp.clip(
            (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.mode()

        # Q loss.
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0) # jnp.min(qs, axis=0)

        # BC loss.
        mse = jnp.square(actions - batch_actions).sum(axis=-1) * batch["valid"][..., -1]

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        actor_loss = -(lam * q).mean()
        bc_loss = (self.config['alpha_actor'] * mse).mean()

        total_loss = actor_loss + bc_loss
        action_std = dist._distribution.stddev()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'action_std': action_std.mean(),
            'mse': mse.mean(),
        }

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _update(agent, batch, full_update=False):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            agent.target_update(new_network, 'critic')
            agent.target_update(new_network, 'actor')
        return agent.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames="full_update")
    def update(self, batch, full_update=False):
        return self._update(self, batch, full_update=full_update)
    
    @partial(jax.jit, static_argnames="full_update")
    def batch_update(self, batch, full_update=False):
        agent, infos = jax.lax.scan(partial(self._update, full_update=full_update), self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        dist = self.network.select('actor')(observations)
        actions = dist.mode()
        noise = jnp.clip(
            (jax.random.normal(rng, actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        actions = jnp.clip(actions + noise, -1, 1)
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

        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=True,
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rebrac',  # Agent name.
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
            
            ## Main hyperparameter(s)
            actor_noise=0.2,  # Actor noise scale.
            alpha_actor=0.0,  # Actor BC coefficient.
            
            ## Other hyperparameter(s)
            actor_freq=2,           # Actor update frequency.
            actor_fc_scale=0.01,    # Final layer initialization scale for actor.
            actor_noise_clip=0.5,   # Actor noise clipping threshold.
        )
    )
    return config
