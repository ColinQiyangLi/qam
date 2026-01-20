import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

from typing import Any

class CGQLAgent(flax.struct.PyTreeNode):
    """Classifier-guidance Q-learning.

    This agent implements a direct application of classifier guidance in the context of Q-learning.
    The Q function acts as the classifier and we use its action gradient as the approximation of the score function.
    To estimate the correct score function, we use three approximations: simple/mse/linex
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        rng, sample_rng, t_rng, noie_rng = jax.random.split(rng, 4)
        
        # t=1 is for noise-free actions
        t1 = jnp.ones_like(batch_actions[..., 0:1])
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions, t1)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        if self.config["separate"]:
            q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        else:
            q = self.network.select('critic')(batch['observations'], batch_actions, t1, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
        
        if self.config["mode"] in ["mse", "linex"]:
            t = jax.random.uniform(t_rng, t1.shape)
            noise = jax.random.normal(noie_rng, batch_actions.shape)
            noisy_actions = (1 - t) * noise + t * batch_actions
            
            noisy_target_q = self.network.select('target_critic')(batch["observations"], batch_actions, t1)
            noisy_q = self.network.select('critic')(batch['observations'], noisy_actions, t, params=grad_params)

            if self.config["mode"] == "mse":
                noisy_critic_loss = (jnp.square(noisy_q - noisy_target_q) * batch['valid'][..., -1]).mean()
            
            elif self.config["mode"] == "linex":
                # Itakura–Saito distance, following TMD: https://people.eecs.berkeley.edu/~vmyers/papers/myers2025offline.pdf
                delta = (-noisy_q + noisy_target_q) * self.config["inv_temp"]
                mask = (delta > self.config["isd_clip"])
                delta_clipped = jnp.where(mask, self.config["isd_clip"], delta)
                noisy_critic_loss = (jnp.where(mask, delta, jnp.exp(delta_clipped) + noisy_q * self.config["inv_temp"]) * batch['valid'][..., -1]).mean()

                info["exp_bias"] = jnp.mean(jnp.exp(noisy_q * self.config["inv_temp"]) - jnp.exp(noisy_target_q * self.config["inv_temp"]))

            info["noisy_critic_loss"] = noisy_critic_loss
            total_loss += noisy_critic_loss * self.config["noisy_coef"]
        
        return total_loss, info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # Flow-matching loss
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])

        return actor_loss, {'flow_loss': actor_loss}

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
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        # update_size = batch["observations"].shape[0]
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
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        
        # query the critic for the noiseless action (at t=1)
        t1 = jnp.ones_like(actions[..., 0:1])
        q = self.network.select("critic")(observations, actions, t1).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        # best-of-n
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        actions = noises

        network = self.network.select('target_actor' if self.config["target_guidance"] else 'actor')
        
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = network(observations, actions, t)
            
            if self.config["inv_temp"] > 0. and i > 0:
                if self.config["mode"] == "simple":
                    score = jax.grad(lambda x, y, t: self.network.select("critic")(x, y, t).mean(axis=0).sum(), 1)(observations, actions, jnp.ones_like(t))
                elif self.config["mode"] in ["mse", "linex"]:
                    score = jax.grad(lambda x, y, t: self.network.select("critic")(x, y, t).mean(axis=0).sum(), 1)(observations, actions, t)
                vels = vels + (score * (1 - t) * self.config["inv_temp"] + actions) * self.config["guidance_coef"] / t
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
        assert config['num_qs'] > 1
        critic_def = ActorVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            action_dim=1,  # auto squeeze when the dimension=1
            num_ensembles=config['num_qs'], # does not support num_qs=1
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )
        
        network_info = dict(
            actor=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )
        
        if config["separate"]:
            network_info.update(dict(
                critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
                d_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions, ex_times)),
            ))
        else:
            network_info.update(dict(
                critic=(copy.deepcopy(critic_def), (ex_observations, full_actions, ex_times)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions, ex_times))
            ))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cgql',  # Agent name.
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
            mode="simple",      # (1) simple: don't learn time-conditioned critic, 
                                # (2) mse:    learn time-conditioned critic by distilling with MSE loss (approx. by dropping exp and log)
                                # (3) linex:  learn time-conditioned critic by distilling with Linex loss 
            noisy_coef=0.001,   # Weight for training the time-conditioned critic
            inv_temp=1.0,       # Inverse temperature ($\tau$)
            guidance_coef=0.1,  # Guidance coefficient ($\vartheta$)

            ## Other design variants and clipping parameters
            isd_clip=5.0,           # For the linex-variant only. This is to clip the exponential in the linex loss for numerical stability
            separate=False,         # Use a separate network for time-conditioned critic
            target_guidance=True,   # Use target network for classifier guidance in the action sampling
        )
    )
    return config
