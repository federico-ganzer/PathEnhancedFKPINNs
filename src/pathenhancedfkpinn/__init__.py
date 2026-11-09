"""Path-signature residual loss for Feynman-Kac supervised PINNs, in JAX.

Importing the package sets two JAX options (see `precision`):
- float64 support on. New arrays then default to float64, so every array in this package is
  created with an explicit dtype.
- full-precision float32 matrix products. On A100/H100 GPUs JAX's default runs float32 matmuls
  as TF32 (10-bit mantissa, about 3 significant digits). CPUs are unaffected.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

__version__ = "0.1.0"
