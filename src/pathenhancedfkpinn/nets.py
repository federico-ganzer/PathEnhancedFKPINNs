"""Networks. tanh MLP, width 128, depth 4 (Equinox `depth` = hidden layers).

The network is float32 (parameters and inputs). It takes float64 in and returns float64 out, so
everything around it (labels, residuals, path sums) stays float64.

    net = FieldNet.create(d=1, time=True, lo=[-1.0, 0.0], hi=[1.0, 1.0], key=k)
    u = net(y, tau)          # y (d,), tau scalar -> float64 scalar
    params, static = split(net)

Initialisation "xavier_normal" (default) reproduces the legacy PyTorch MLP: Xavier-normal weights
with tanh gain 5/3, zero biases. "equinox" keeps Equinox's default initialisation.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

ACTIVATIONS: dict[str, Callable] = {
    "tanh": jnp.tanh,
    "sin": jnp.sin,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
}
GAIN = {"tanh": 5.0 / 3.0, "sin": 5.0 / 3.0}


def _xavier_normal(mlp: eqx.nn.MLP, key, gain: float) -> eqx.nn.MLP:
    '''xavier-normal initialisation of weights, zero biases.'''
    keys = jax.random.split(key, len(mlp.layers))
    new_w, new_b = [], []
    for k, layer in zip(keys, mlp.layers):
        fan_out, fan_in = layer.weight.shape
        std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
        new_w.append(std * jax.random.normal(k, layer.weight.shape, layer.weight.dtype))
        new_b.append(jnp.zeros_like(layer.bias))
    mlp = eqx.tree_at(lambda m: [lay.weight for lay in m.layers], mlp, new_w)
    return eqx.tree_at(lambda m: [lay.bias for lay in m.layers], mlp, new_b)


def make_mlp(
    in_size: int, width=128, depth=4, activation="tanh", init="xavier_normal", *, key
) -> eqx.nn.MLP:
    k_net, k_init = jax.random.split(key)
    mlp = eqx.nn.MLP(
        in_size, "scalar", width, depth, activation=ACTIVATIONS[activation], dtype=jnp.float32, key=k_net
    )
    if init == "xavier_normal":
        mlp = _xavier_normal(mlp, k_init, GAIN.get(activation, 1.0))
    elif init != "equinox":
        raise ValueError(f"unknown init {init!r}")
    return mlp


class FieldNet(eqx.Module):
    """u_theta(y, tau): inputs scaled from the box [lo, hi] to [-1, 1], float32 inside.

    lo/hi cover (x_1..x_d, t) for parabolic problems (time=True) or x only (time=False). They are
    static Python floats, so they are never trained."""

    mlp: eqx.nn.MLP
    lo: tuple[float, ...] = eqx.field(static=True)
    hi: tuple[float, ...] = eqx.field(static=True)
    time: bool = eqx.field(static=True)

    @classmethod
    def create(
        cls, d: int, time: bool, lo, hi, width=128, depth=4, activation="tanh",
        init="xavier_normal", *, key,
    ) -> "FieldNet":  # fmt: skip
        n_in = d + int(time)
        lo, hi = tuple(float(v) for v in lo), tuple(float(v) for v in hi)
        if len(lo) != n_in or len(hi) != n_in:
            raise ValueError(f"lo/hi need {n_in} entries, got {len(lo)}/{len(hi)}")
        return cls(make_mlp(n_in, width, depth, activation, init, key=key), lo, hi, time)

    def __call__(self, y: jax.Array, tau=None) -> jax.Array:
        y = jnp.atleast_1d(y)
        z = jnp.concatenate([y, jnp.atleast_1d(jnp.asarray(tau, y.dtype))]) if self.time else y
        lo, hi = jnp.asarray(self.lo, z.dtype), jnp.asarray(self.hi, z.dtype)
        z = 2.0 * (z - lo) / (hi - lo) - 1.0 # scale to [-1, 1] for tanh activations for better training
        return self.mlp(z.astype(jnp.float32)).astype(jnp.float64)


def split(net: eqx.Module):
    """(params, static) with params = the trainable float arrays."""
    return eqx.partition(net, eqx.is_inexact_array) # return params and static parts of the network


def apply_fn(static) -> Callable:
    """u(params, y, tau) -> float64 scalar: the functional form used by the losses."""

    def u(params, y, tau=None):
        return eqx.combine(params, static)(y, tau) # combine params and static parts of the network and call it with inputs y and tau

    return u
