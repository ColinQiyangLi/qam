from typing import Callable, Optional, Sequence, Any, Type

import flax.linen as nn
import jax.numpy as jnp
import jax
import functools
import tensorflow_probability

import math
from functools import partial

from jax.nn.initializers import zeros, constant

def default_init(scale: Optional[float] = 1.0, *args, **kwargs):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

def orthogonal_init(scale: Optional[float] = jnp.sqrt(2.0), *args, **kwargs):
    return jax.nn.initializers.orthogonal(scale)

def pytorch_init(scale: Optional[float] = 1.0, fan_in=None):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    def _init(key, shape, dtype, fan_in):
        if fan_in is None:
            fan_in = shape[-2]
        bound = math.sqrt(1 / fan_in)
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return partial(_init, fan_in=fan_in)

def uniform_init(bound: Optional[float] = 1.0, *args, **kwargs):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init

def zeros_init(scale: Optional[float] = 1.0, *args, **kwargs):
    return zeros

def constant_init(scale, *args, **kwargs):
    return constant(scale)

INIT_FNS = {
    None: default_init,
    "orthogonal": orthogonal_init,
    "pytorch": pytorch_init,
    "uniform": uniform_init,
}

BIAS_INIT_FNS = {
    None: zeros_init,
    "constant": constant_init,
    "pytorch": pytorch_init,
    "uniform": uniform_init,
}

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

def update_target_network(main_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y,
        main_params, target_params
    )

def value_and_multi_grad(fun, n_outputs, argnums=0):
    def select_output(index):
        def wrapped(*args, **kwargs):
            x, *aux = fun(*args, **kwargs)
            return (x[index], *aux)
        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=True)
        for i in range(n_outputs)
    )
    def multi_grad_fn(*args, **kwargs):
        grads, values = [], []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)
    return multi_grad_fn


def broadcast_concatenate(*arrs):
    shape = jnp.broadcast_shapes(*map(lambda x: x.shape[:-1], arrs))
    return jnp.concatenate(tuple(map(lambda x: jnp.broadcast_to(x, shape=shape + (x.shape[-1],)), arrs)), axis=-1)


"""
Both classes below are taken from the link below (note that the initialization used originally are xavier_uniform)
https://github.com/philippe-eecs/IDQL/blob/main/jaxrl5/networks/resnet.py#L32
"""


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x

class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu
    kernel_init_type: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        init_fn = INIT_FNS[self.kernel_init_type]
        x = nn.Dense(self.hidden_dim, kernel_init=init_fn())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
            
        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=init_fn())(x)
        return x

class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    
    kernel_scale: Optional[float] = None
    kernel_init_type: Optional[str] = None
    kernel_scale_final: Optional[float] = None
    kernel_init_type_final: Optional[str] = None
    
    bias_scale: Optional[float] = None
    bias_init_type: Optional[str] = None
    bias_scale_final: Optional[float] = None
    bias_init_type_final: Optional[str] = None

    @nn.compact
    def __call__(self, args, training: float = False) -> jnp.ndarray:
        
        if type(args) == tuple:
            x = broadcast_concatenate(*args) # broadcast everything together
        else:
            x = args

        for i, size in enumerate(self.hidden_dims):
            init_fn = INIT_FNS[self.kernel_init_type]
            bias_init_fn = BIAS_INIT_FNS[self.bias_init_type]
            kernel_scale = self.kernel_scale
            bias_scale = self.bias_scale
            if i + 1 == len(self.hidden_dims):
                if self.kernel_init_type_final is not None:
                    init_fn = INIT_FNS[self.kernel_init_type_final]
                if self.bias_init_type_final is not None:
                    bias_init_fn = BIAS_INIT_FNS[self.bias_init_type_final]
                if self.kernel_scale_final is not None:
                    kernel_scale = self.kernel_scale_final
                if self.bias_scale_final is not None:
                    bias_scale = self.bias_scale_final

            if kernel_scale:
                kernel_init = init_fn(kernel_scale)
            else:
                kernel_init = init_fn()
            if bias_scale:
                bias_init = bias_init_fn(bias_scale, fan_in=x.shape[-1])
            else:
                bias_init = bias_init_fn(fan_in=x.shape[-1])

            x = nn.Dense(size, kernel_init=kernel_init, bias_init=bias_init)(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def sample_and_log_prob(self, *args, **kwargs):
        x = self.sample(*args, **kwargs)
        return x, self.log_prob(x)

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties

class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    fixed_log_std: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False
    learnable_log_std_multiplier: Optional[float] = None   # learnable log_std multiplier
    learnable_log_std_offset: Optional[float] = None       # learnable log_std offset
    kernel_init_scale: float = 1.0
    kernel_init_type: Optional[str] = None
    bias_init_scale: float = 0.0
    bias_init_type: Optional[str] = None

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        init_fn = INIT_FNS[self.kernel_init_type] # orthogonal_init if self.orthogonal_init else default_init
        bias_init_fn = BIAS_INIT_FNS[self.bias_init_type]
        means = nn.Dense(
            self.action_dim, kernel_init=init_fn(self.kernel_init_scale), bias_init=bias_init_fn(self.bias_init_scale, fan_in=x.shape[-1]), name="OutputDenseMean"
        )(x)
        
        if self.fixed_log_std:
            log_stds = jnp.zeros_like(means)
        else:
            if self.state_dependent_std:
                log_stds = nn.Dense(
                    self.action_dim, kernel_init=init_fn(self.kernel_init_scale), bias_init=bias_init_fn(self.bias_init_scale, fan_in=x.shape[-1]), name="OutputDenseLogStd"
                )(x)
            else:
                log_stds = self.param(
                    "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
                )
        
            if self.learnable_log_std_multiplier is not None:
                log_stds *= self.param("LogStdMul", nn.initializers.constant(self.learnable_log_std_multiplier), (), jnp.float32)
            if self.learnable_log_std_offset is not None:
                log_stds += self.param("LogStdOffset", nn.initializers.constant(self.learnable_log_std_offset), (), jnp.float32)
            
            log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution

TanhNormal = functools.partial(Normal, squash_tanh=True)

class TD3Actor(nn.Module):
    # AKA deterministic tanh policy + fixed-scale truncated gaussian noise
    base_cls: Type[nn.Module]
    action_dim: int
    kernel_init_scale: float = 1.0
    kernel_init_type: Optional[str] = None
    bias_init_scale: float = 0.0
    bias_init_type: Optional[str] = None

    noise_scale: float = 0.2
    noise_clip: float = 0.5

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        init_fn = INIT_FNS[self.kernel_init_type] # orthogonal_init if self.orthogonal_init else default_init
        bias_init_fn = BIAS_INIT_FNS[self.bias_init_type]
        output = nn.Dense(
            self.action_dim, kernel_init=init_fn(self.kernel_init_scale), bias_init=bias_init_fn(self.bias_init_scale, fan_in=x.shape[-1]), name="OutputDenseMean"
        )(x)
        action = jnp.tanh(output)
        
        distribution = tfd.TruncatedNormal(action, self.noise_scale, 
            low=action - self.noise_clip, 
            high=action + self.noise_clip)
        return distribution
