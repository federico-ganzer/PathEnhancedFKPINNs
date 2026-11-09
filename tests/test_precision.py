"""JAX options the package sets on import."""

import jax

import pathenhancedfkpinn  # noqa: F401


def test_import_sets_jax_options():
    assert jax.config.jax_enable_x64
    assert jax.config.jax_default_matmul_precision == "highest"  # not TF32 on A100/H100
