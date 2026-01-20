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
from utils.networks import MLP, TanhNormal, LogParam

class QAMAgent(flax.struct.PyTreeNode):
    """Q-learning with adjoint matching."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
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
    def adj_matching(self, obs, rng, flow_steps=None):
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        h = 1 / flow_steps
        xs = [x]
        ts = []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config["residual"]:
                    v = self.network.select("actor_fast")(obs, x, t) + actor_slow(obs, x, t)
                else:
                    v = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:  # use ODE integration for the last step following the adjoint-matching paper
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        # Compute the critic's action gradient as the adjoint state initialization
        critic_network = "target_critic" if self.config["use_target_grad"] else "critic"
        if self.config['clip_adj']:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
        else:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, y).mean(axis=0).sum(), 1)

        adj = -grad_fn(obs, xs[-1]) * self.config["inv_temp"]
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
        }
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)
            
            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            
            adjs.append(adj)
        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, adj_rng, edit_rng = jax.random.split(rng, 5)

        ## BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss
        
        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")
        
        ## Adjoint-matching
        # Compute the adjoint states
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch["observations"][None], self.config["flow_steps"], axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)

        vf_base = actor_slow(observations, xs, ts)
        
        # Compute the adjoint matching loss
        if self.config["residual"]:
            adj_loss = jnp.sum(jnp.square(vf_fine * 2 / sigmas + sigmas * adjs), axis=-1)
        else:
            adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)

        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        if self.config["fql_alpha"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            fql_noises = jax.random.normal(edit_base_rng, (batch_size, action_dim))
            flow_actions = self.compute_flow_actions(batch["observations"], 
                fql_noises, 
                model="slow,fast" if self.config["residual"] else "fast")
            
            os_actions = self.network.select('one_step_actor')(
                batch["observations"], fql_noises, 
                params=grad_params)
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)
            
            # FQL loss.
            os_actions = jnp.clip(os_actions, -1, 1)
            fql_qs = self.network.select(f'critic')(batch['observations'], actions=os_actions)
            fql_q = jnp.mean(fql_qs, axis=0)
            fql_q_loss = -fql_q.mean()

            info["fql_distill_loss"] = fql_distill_loss
            info["fql_q_loss"] = fql_q_loss

            actor_loss += fql_q_loss + fql_distill_loss * self.config["fql_alpha"]

        if self.config["edit_scale"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            flow_actions = self.compute_flow_actions(batch["observations"], 
                jax.random.normal(edit_base_rng, (batch_size, action_dim)), 
                model="slow,fast" if self.config["residual"] else "fast")
            
            edit_dist = self.network.select('edit_actor')(
                jnp.concatenate((batch["observations"], flow_actions), axis=-1), 
                params=grad_params)
            edit = edit_dist.sample(seed=edit_rng)
            edit_log_probs = edit_dist.log_prob(edit)
            
            edited_actions = flow_actions + edit * self.config["edit_scale"]
            
            # Edit policy loss.
            edited_actions = jnp.clip(edited_actions, -1, 1)
            qs = self.network.select(f'critic')(batch['observations'], actions=edited_actions)
            q = jnp.mean(qs, axis=0)
            edit_q_loss = -q.mean()

            edit_entropy_loss = (edit_log_probs * self.network.select('edit_alpha')()).mean()

            alpha = self.network.select('edit_alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(edit_log_probs).mean()
            edit_alpha_loss = (alpha * (entropy - self.config['edit_target_entropy'])).mean()

            actor_loss += edit_q_loss + edit_entropy_loss + edit_alpha_loss

            info["edit_q_loss"] = edit_q_loss
            info["edit_entropy_loss"] = edit_entropy_loss
            info["edit_alpha_loss"] = edit_alpha_loss
            info["edit_entropy"] = entropy
            info["edit_alpha"] = alpha

        return actor_loss + total_fast_loss, {'flow_loss': flow_loss, "fast_loss": total_fast_loss, **info}

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
        rng, edit_rng = jax.random.split(rng)
        
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

        if self.config["fql_alpha"] > 0.:  # if fql_alpha > 0, use the one-step policy
            actions = self.network.select('one_step_actor')(
                observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:   # otherwise use the flow policy
            if self.config["inv_temp"] == 0.:
                actions = self.compute_flow_actions(observations, noises, model="slow")
            else:
                actions = self.compute_flow_actions(observations, noises, model="slow,fast" if self.config["residual"] else "fast")
            if self.config["edit_scale"] > 0.:  # if there is an edit policy, refine the action further
                edit_dist = self.network.select("edit_actor")(jnp.concatenate((observations, actions), axis=-1))
                actions = actions + edit_dist.sample(seed=edit_rng) * self.config["edit_scale"]
            actions = jnp.clip(actions, -1, 1)
        
        # best-of-n sampling
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

        if config['edit_target_entropy'] is None:
            config['edit_target_entropy'] = -config['edit_target_entropy_multiplier'] * full_action_dim

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
        
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )

        assert (config["fql_alpha"] * config["edit_scale"] == 0.), "Only one of fql_alpha and edit_scale can be non-zero."
        
        if config["fql_alpha"] > 0.:
            network_info.update(dict(
                one_step_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions, None)),
            ))

        if config["edit_scale"] > 0.:
            edit_actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
            edit_actor_def = TanhNormal(edit_actor_base_cls, full_action_dim)
            alpha_def = LogParam()

            network_info.update(dict(
                edit_actor=(edit_actor_def, jnp.concatenate((ex_observations, full_actions), axis=-1)),
                edit_alpha=(alpha_def, ()),
            ))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),  # clip grad norm
                optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
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
            agent_name='qam',  # Agent name.
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
            inv_temp=0.3,   # Inverse temperature that controls the influence of Q for QAM
            fql_alpha=0.,   # If > 0 train a one-step policy that is distilled from the QAM flow policy while maximizing Q
            edit_scale=0.,  # If > 0 train an edit policy that refines the output of the QAM flow policy to maximize Q
            
            ## Other variants/hyperparamter(s)
            target_actor=True,
            residual=False,
            clip_adj=True,
            clip_grad=True,
            use_target_grad=True,
            edit_target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy for edit policy (None for automatic tuning).
            edit_target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy (for edit policy).
        )
    )
    return config

