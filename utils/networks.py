from typing import Any, Optional, Sequence, Type

import distrax
import flax.linen as nn
import jax.numpy as jnp

import tensorflow_probability

import functools

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

default_init = nn.initializers.xavier_uniform


def default_init_(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )

class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            # f = jnp.log(10000) / (half_dim - 1)
            # f = jnp.exp(jnp.arange(half_dim) * -f)
            # f = x * f
            f = x * jnp.arange(half_dim)
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, y=None):
        if y is not None:
            x = jnp.concatenate([x, y], axis=-1)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())




class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties

class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    learnable_log_std_multiplier: Optional[float] = None   # learnable log_std multiplier
    learnable_log_std_offset: Optional[float] = None       # learnable log_std offset
    state_dependent_std: bool = True
    squash_tanh: bool = False
    fixed_log_std: Optional[float] = None
    # kernel_init_scale: float = 1.0

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
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

        if self.fixed_log_std is not None:
            log_stds = jnp.ones_like(log_stds) * self.fixed_log_std

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init_(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init_(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """
    action_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    encoder: nn.Module = None
    num_ensembles: int = 1
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.mlp = mlp_class((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)
        if self.action_dim == 1:
            v = v.squeeze(-1)
        return v
