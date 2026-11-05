"""Counter-based random keys.

Every random draw comes from a key derived as  seed -> purpose -> index -> index ...  with
`jax.random.fold_in`, so any path can be regenerated from (seed, purpose, point, block) without
storing it, and arms within a seed share common random numbers by construction.

    k = derive(seed, "paths", point_index, block_index)
"""

from __future__ import annotations

import jax

# Fixed purpose codes. Never renumber: changing a code changes every stored result.
PURPOSES: dict[str, int] = {
    "init": 0,  # network initialisation
    "coll": 1,  # collocation points
    "fk_points": 2,  # which collocation points become FK points
    "paths": 3,  # Brownian increments of FK paths
    "ht": 4,  # Horvitz-Thompson node draws
    "words": 5,  # word subsampling
    "eval": 6,  # held-out evaluation points
    "train": 7,  # per-step minibatch draws during training
    "ic": 8,  # initial-condition points
    "bc": 9,  # boundary points
    "test": 100,  # unit tests
}


def root(seed: int) -> jax.Array:
    return jax.random.key(seed)


def fold(key: jax.Array, *indices) -> jax.Array:
    """Fold integer indices (Python ints or traced int arrays) into a key, in order."""
    for i in indices:
        key = jax.random.fold_in(key, i)
    return key


def derive(seed: int, purpose: str, *indices) -> jax.Array:
    if purpose not in PURPOSES:
        raise KeyError(f"unknown purpose {purpose!r}; have {sorted(PURPOSES)}")
    return fold(root(seed), PURPOSES[purpose], *indices)
