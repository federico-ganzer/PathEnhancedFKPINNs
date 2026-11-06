"""Arm E: untruncated path residual E[(Y^theta - Y)^2] on the same paths.

Full Euler sums only for now. The HT variant (product of two independent HT draws) needs a
second node set per path and is not implemented yet; configs that ask for it are rejected.
"""

from __future__ import annotations

import jax.numpy as jnp


def path_mse(R: jnp.ndarray) -> jnp.ndarray:
    """R (B, M') per-path residuals -> scalar."""
    return jnp.mean(R**2)
