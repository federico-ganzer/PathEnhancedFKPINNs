"""Path-signature residual loss for Feynman-Kac supervised PINNs, in JAX.

Importing the package turns on float64 support in JAX. With the flag on, new arrays default to
float64, so every array in this package is created with an explicit dtype (see `precision`).
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
