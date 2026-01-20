import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, LogParam, ActorVectorField, MLP, TanhNormal

from functools import partial

class DSRLAgent(flax.struct.PyTreeNode):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # Critic loss.
        rng, sample_rng, noise_rng = jax.random.split(rng, 3)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        # Latent critic distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_actions.shape[0], batch_actions.shape[-1]))
        actions = self.sample_flow_actions(batch['observations'], noises=noises)
        actions = jnp.clip(actions, -1, 1)
        target_qs = self.network.select('critic')(batch['observations'], actions)
        qs = self.network.select('z_critic')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((qs - target_qs) ** 2)

        total_loss = critic_loss + distill_loss

        return total_loss, {
            'total_loss': total_loss,
            'critic_loss': critic_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        
        batch_size, action_dim = batch_actions.shape

        # BC flow loss.
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        # Actor loss.
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        actions = actions * self.config['noise_scale']

        qs = self.network.select('z_critic')(batch['observations'], actions)
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = flow_loss + actor_loss + alpha_loss

        action_std = dist._distribution.stddev()
        
        return total_loss, {
            'total_loss': total_loss,
            'flow_loss': flow_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'action_std': action_std.mean(),
            'q': q.mean(),
        }

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
    def _update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor_bc_flow')

        return self.replace(network=new_network, rng=new_rng), info

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
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        dist = self.network.select('actor')(observations)
        noises = dist.sample(seed=rng)
        noises = jnp.clip(noises, -1, 1)
        noises = noises * self.config['noise_scale']

        actions = self.sample_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))
        
        return actions

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        noises,
    ):
        actions = noises
        model = self.network.select('target_actor_bc_flow' if self.config['use_target_latent'] else 'actor_bc_flow')
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = model(observations, actions, t, is_encoded=True)
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

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * full_action_dim
        
        critic_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config["value_layer_norm"], num_ensembles=config["num_qs"])
        actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = TanhNormal(actor_base_cls, full_action_dim)
        
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
        )
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            z_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, full_actions[..., :1])),
            target_actor_bc_flow=(copy.deepcopy(actor_bc_flow_def), (ex_observations, full_actions, full_actions[..., :1])),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_bc_flow'] = params['modules_actor_bc_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dsrl',  # Agent name.
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
            noise_scale=1.0,    # Noise action scale.
            
            ## Other design variants/hyperparameters
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,          # Multiplier to dim(A) for target entropy.
            use_target_latent=True,                 # Use the target BC policy network to form the latent space
        )
    )
    return config
