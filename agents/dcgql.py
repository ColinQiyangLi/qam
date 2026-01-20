from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import flax
import copy
from typing import Any
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

import copy

from utils.diffusion import DDPM, FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from utils.networks import MLP, Value

import ml_collections

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class DCGQLAgent(flax.struct.PyTreeNode):
    """
    Diffusion classifier guidance Q-learning agents.

    Our implementation mostly following the implementations below with minor modifications:
    - https://github.com/Fang-Lin93/DAC/blob/main/agents/dac.py (DAC), and
    - https://github.com/escontra/score_matching_rl/blob/main/jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py (QSM)
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    betas: Any
    alphas: Any
    alpha_hats: Any

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng)
        next_qs = self.network.select("target_critic")(batch["next_observations"][..., -1, :], next_actions)

        next_q = next_qs.mean(axis=0) - next_qs.std(axis=0) * self.config["rho"]
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select("critic")(batch["observations"], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()
        return critic_loss, {"critic_loss": critic_loss, "q": q.mean()}

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        
        key, rng = jax.random.split(rng, 2)
        t = jax.random.randint(key, batch_actions.shape[:-1], 1, self.config["diffusion_steps"] + 1)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, batch_actions.shape)
        
        alpha_hats = self.alpha_hats[t]
        t = jnp.expand_dims(t, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch_actions + alpha_2 * noise_sample
        
        # Compute the critic's action gradient.
        if self.config["use_target_critic_grad"]:
            q_grad = jax.grad(lambda actions: self.network.select("target_critic")(batch['observations'], actions).mean(axis=0).sum())
        else:
            q_grad = jax.grad(lambda actions: self.network.select("critic")(batch['observations'], actions).mean(axis=0).sum())

        eps_pred = self.network.select("actor")(batch['observations'], noisy_actions, t, params=grad_params)

        # Regular ddpm loss as the BC loss.
        bc_loss = (jnp.square(noise_sample - eps_pred).mean(axis=-1) ** batch["valid"][..., -1]).mean()
        
        # QSM/DAC loss.
        if self.config["actor_loss_type"] == "qsm":
            actor_loss = (jnp.square(-self.config["inv_temp"] * q_grad(noisy_actions) - eps_pred).mean(axis=-1) ** batch["valid"][..., -1]).mean()
        elif self.config["actor_loss_type"] == "dac":
            actor_loss = ((alpha_2 * q_grad(noisy_actions) * eps_pred).mean(axis=-1) ** batch["valid"][..., -1]).mean()

        total_loss = bc_loss * self.config["alpha"] + actor_loss
        return total_loss, {"actor_loss": actor_loss}

    
    def ddpm_sampler(self, rng, observations, noise):
        batch_size = observations.shape[0]
        input_time_proto = jnp.ones((*noise.shape[:-1], 1))

        def fn(input_tuple, t):
            current_x, rng_ = input_tuple
            input_time = input_time_proto * t

            eps_pred = self.network.select("actor")(observations, current_x, input_time)

            x0_hat = 1 / jnp.sqrt(self.alpha_hats[t]) * (current_x - jnp.sqrt(1 - self.alpha_hats[t]) * eps_pred)
            if self.config["clip_sampler_before"]:
                x0_hat = jnp.clip(x0_hat, -1, 1)
                current_x = 1 / (1 - self.alpha_hats[t]) * (jnp.sqrt(self.alpha_hats[t - 1]) * (1 - self.alphas[t]) * x0_hat +
                                                    jnp.sqrt(self.alphas[t]) * (1 - self.alpha_hats[t - 1]) * current_x)
            else:
                current_x = x0_hat
            
            rng_, key_ = jax.random.split(rng_, 2)
            z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
            sigmas_t = jnp.sqrt((1 - self.alphas[t]))
            
            current_x = current_x + (t > 1) * (sigmas_t * z)
            if self.config["clip_sampler_after"]:
                current_x = jnp.clip(current_x, -1., 1.)
            return (current_x, rng_), ()

        rng, denoise_key = jax.random.split(rng, 2)
        output_tuple, () = jax.lax.scan(fn, (noise, denoise_key),
            jnp.arange(self.config["diffusion_steps"], 0, -1),
            unroll=self.config["diffusion_steps"])

        action_0, rng = output_tuple
        return action_0
    
    @jax.jit
    def sample_actions(self, observations, rng):
        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        
        if len(observations.shape) == 1:
            need_squeeze = True
            observations = observations[None, :]
        else:
            need_squeeze = False

        noise_key, sampler_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, (observations.shape[0], action_dim))
        actions = self.ddpm_sampler(sampler_key, observations, noise)
        
        if need_squeeze:
            return actions[0]
        return actions
    
    def total_loss(self, batch, grad_params, rng):
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

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(partial(self._update), self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]

        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        preprocess_time_cls = partial(
            FourierFeatures, output_size=config["time_dim"], learnable=True)

        cond_model_cls = partial(
            MLP, hidden_dims=config["actor_hidden_dims"], activations=mish,
            activate_final=False)
        
        base_model_cls = partial(MLP,
            hidden_dims=tuple(list(config["actor_hidden_dims"]) + [full_action_dim]),
            activations=mish, layer_norm=config["actor_layer_norm"],
            activate_final=False)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)
        
        ex_times = jnp.zeros((1,))

        critic_def = Value(hidden_dims=config["value_hidden_dims"], 
                           layer_norm=config["value_layer_norm"], 
                           num_ensembles=config["num_qs"])

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations, full_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx = optax.chain(optax.clip_by_global_norm(max_norm=config["clip_grad_norm"]),  # clip grad norm
                    optax.adam(learning_rate=config["lr"]))
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        
        beta_schedule = config["beta_schedule"]
        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(config["diffusion_steps"]))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, config["diffusion_steps"])
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(config["diffusion_steps"]))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')
        
        betas = jnp.concatenate([jnp.zeros((1,)), betas])

        alphas = 1 - betas
        alpha_hats = jnp.cumprod(alphas)
        config["action_dim"] = action_dim

        return cls(
            rng=rng, network=network, config=flax.core.FrozenDict(**config), 
            alphas=alphas, alpha_hats=alpha_hats, betas=betas
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dcgql',  # Agent name.
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
            num_qs=10,           # Critic ensemble size
            rho=0.5,            # Pessimistic backup

            discount=0.99,      # Discount factor.
            tau=0.005,          # Target network update rate.
            diffusion_steps=10, # Number of diffusion steps.
            time_dim=64,        # Time encoding dimension for the diffusion policy
            beta_schedule='vp', # Diffusion schedule
            
            best_of_n=1,        # Best-of-n for computing Q-targets and sampling actions.
            
            ## Main hyperparameter(s)
            actor_loss_type="qsm",      # Our QSM uses an additional BC loss. Our DAC follows the original implementation as it already has a BC loss.
            clip_sampler_before=False,  # First map it to the noise-free space with one-step approximate,
                                        #  clip it to [-1, 1], and then map it back before applying each diffusion step (used by DAC only)
            clip_sampler_after=False,   # Clip the intermediate noisy action to [-1, 1] after each diffusion step (used by QSM only)
            inv_temp=1.0,               # Inverse temperature for qsm/dac loss.
            alpha=0.0,                  # Weight for BC loss (only used by QSM and not by DAC).

            # Other design variants and clipping parameters
            use_target_critic_grad=True,
            clip_grad_norm=1.,
        )
    )
    return config
