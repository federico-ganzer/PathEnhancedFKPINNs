"""Euler paths: block structure of the random stream."""

import jax.numpy as jnp
import pytest

from pathenhancedfkpinn.keys import derive
from pathenhancedfkpinn.sde.simulate import simulate_paths, zero_drift


def _paths(M, block):
    x0 = jnp.zeros(1, jnp.float64)
    return simulate_paths(zero_drift, 1.0, x0, 0.1, 10, derive(0, "paths", 0), M, block=block)


@pytest.mark.parametrize("M", [100, 125, 250])
def test_fewer_paths_are_prefix(M):
    """M paths are the first M of the block stream, also when M is not a multiple of the block."""
    big, p = _paths(500, 100), _paths(M, 100)
    assert p.X.shape[0] == p.XT.shape[0] == M
    assert jnp.array_equal(p.dW, big.dW[:M]) and jnp.array_equal(p.XT, big.XT[:M])
