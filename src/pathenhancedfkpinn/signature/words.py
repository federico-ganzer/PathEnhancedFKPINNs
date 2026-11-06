"""Word sets W_{N,N0}, prefix closure, and the tables that turn signatures into orthogonal features.

Letters: 0 = time, 1..d = Brownian. A word w has n(w) non-zero letters and z(w) zeros.

    W_{N,N0} = {empty} u {w : last letter != 0, 1 <= n(w) <= N, z(w) <= N0}
    |W_{N,N0}| - 1 = sum_{n=1..N} d^n sum_{z=0..N0} C(n-1+z, z)

Two signature layouts are supported (see ito_features.py):
- sparse: one column per word of the prefix closure, column 0 = empty word.
- signax: Signax's flat layout, levels 1..depth in order, each level C-raveled over (d+1)^k.
`WordTables.closure_dense_idx` maps each closure word to its position in the Signax layout.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

import numpy as np

from pathenhancedfkpinn.signature.ito_basis import (
    Word,
    n_letters,
    norm2,
    orthogonal_basis,
    strip0,
    z_letters,
)


def count_words(d: int, N: int, N0: int) -> int:
    """|W_{N,N0}| - 1 (non-empty feature words)."""
    return sum(d**n * sum(comb(n - 1 + z, z) for z in range(N0 + 1)) for n in range(1, N + 1))


def feature_words(d: int, N: int, N0: int) -> list[Word]:
    """Non-empty words of W_{N,N0}, sorted by length then lexicographically.

    Built constructively (zero positions x non-zero letters), so d = 50 does not enumerate
    51^4 tuples."""
    out: list[Word] = []
    letters = range(1, d + 1)
    for n in range(1, N + 1):
        for z in range(N0 + 1):
            L = n + z
            # last slot is non-zero; choose which of the first L-1 slots hold the z zeros
            for zero_pos in itertools.combinations(range(L - 1), z):
                nz_pos = [i for i in range(L) if i not in zero_pos]
                for nz in itertools.product(letters, repeat=n):
                    w = [0] * L
                    for p, a in zip(nz_pos, nz):
                        w[p] = a
                    out.append(tuple(w))
    out.sort(key=lambda w: (len(w), w))
    return out


def prefix_closure(words) -> list[Word]:
    """All non-empty prefixes of the given words, sorted by (length, word)."""
    return sorted({w[:j] for w in words for j in range(1, len(w) + 1)}, key=lambda w: (len(w), w))


def dense_offsets(d: int, depth: int) -> np.ndarray:
    """Start of level k (k = 1..depth) in Signax's flat layout; entry k-1."""
    n = d + 1
    return np.cumsum([0] + [n**k for k in range(1, depth)])


def dense_index(w: Word, d: int) -> int:
    """Position of non-empty word w in Signax's flat signature (C order within each level)."""
    n, k = d + 1, len(w)
    off = sum(n**j for j in range(1, k))
    return off + int(np.ravel_multi_index(tuple(w), (n,) * k))


def dense_size(d: int, depth: int) -> int:
    return sum((d + 1) ** k for k in range(1, depth + 1))


def _canonical(w: Word) -> tuple[Word, dict[int, int]]:
    """Relabel non-zero letters in order of first appearance (1, 2, ...). Returns the canonical
    word and the map canonical letter -> original letter. Ito inner products are invariant under
    injective relabelling of Brownian coordinates, so the basis is computed once per pattern."""
    fwd: dict[int, int] = {}
    for c in w:
        if c != 0 and c not in fwd:
            fwd[c] = len(fwd) + 1
    return tuple(fwd.get(c, 0) for c in w), {v: k for k, v in fwd.items()}


def basis_for_words(words) -> dict[Word, dict[Word, float]]:
    """Orthogonal basis {w: {v: coeff at t=1}} and exact norms, using pattern canonicalisation.

    Returns (basis, norms1) with float coefficients and float norms <p_w, p_w>(1)."""
    classes: dict[Word, list[Word]] = {}
    for w in words:
        classes.setdefault(strip0(w), []).append(w)
    cache: dict[Word, tuple[dict, dict]] = {}
    basis: dict[Word, dict[Word, float]] = {}
    norms1: dict[Word, float] = {}
    for key, cls in classes.items():
        ckey, back = _canonical(key)
        canon_cls = sorted({_canonical(w)[0] for w in cls})
        if ckey not in cache:
            B = orthogonal_basis(canon_cls)
            cache[ckey] = (B, {w: norm2(p) for w, p in B.items()})
        B, nr = cache[ckey]
        relabel = lambda v: tuple(back.get(c, 0) if c != 0 else 0 for c in v)  # noqa: E731
        for w in cls:
            cw = _canonical(w)[0]
            basis[w] = {relabel(v): float(c) for v, c in B[cw].items()}
            norms1[w] = float(nr[cw])
    return basis, norms1


@dataclass(frozen=True)
class WordTables:
    """Everything needed to turn a signature into orthogonal features Phi_w.

    feats             (W,) feature words (non-empty), length-then-lex order
    closure           (K,) prefix closure; signature column j+1 holds closure[j], column 0 = empty
    parent, letter    (K+1,) sparse recursion tables (entry 0 = empty word)
    closure_dense_idx (K,) position of each closure word in Signax's flat layout
    C, zc             (K+1, W) basis coefficients at t = 1 and their time exponents z(w) - z(v)
    nrm1, tpow        (W,) norms <p_w, p_w>(1) and their time exponents 2 z(w) + n(w)
    level             (W,) n(w); zeros (W,) z(w)
    """

    d: int
    N: int
    N0: int
    depth: int
    feats: tuple[Word, ...]
    closure: tuple[Word, ...]
    parent: np.ndarray
    letter: np.ndarray
    closure_dense_idx: np.ndarray
    C: np.ndarray
    zc: np.ndarray
    nrm1: np.ndarray
    tpow: np.ndarray
    level: np.ndarray
    zeros: np.ndarray

    @property
    def n_feats(self) -> int:
        return len(self.feats)


def word_tables(d: int, N: int, N0: int, feats=None) -> WordTables:
    """Tables for W_{N,N0} (or for a given subset `feats`, e.g. a word subsample)."""
    feats = list(feats) if feats is not None else feature_words(d, N, N0)
    closure = prefix_closure(feats)
    words = [()] + closure
    idx = {w: i for i, w in enumerate(words)}
    parent = np.array([0] + [idx[w[:-1]] for w in closure], dtype=np.int64)
    letter = np.array([0] + [w[-1] for w in closure], dtype=np.int64)
    depth = max((len(w) for w in closure), default=0)  # N = 0: empty word only (arm B_0)
    dense_idx = np.array([dense_index(w, d) for w in closure], dtype=np.int64)

    basis, norms1 = basis_for_words(feats)
    W = len(feats)
    C = np.zeros((len(words), W))
    zc = np.zeros((len(words), W))
    for j, w in enumerate(feats):
        for v, c in basis[w].items():
            C[idx[v], j] = c
            zc[idx[v], j] = z_letters(w) - z_letters(v)
    nrm1 = np.array([norms1[w] for w in feats])
    level = np.array([n_letters(w) for w in feats], dtype=np.int64)
    zeros = np.array([z_letters(w) for w in feats], dtype=np.int64)
    tpow = 2 * zeros + level
    return WordTables(
        d=d, N=N, N0=N0, depth=depth, feats=tuple(feats), closure=tuple(closure),
        parent=parent, letter=letter, closure_dense_idx=dense_idx, C=C, zc=zc,
        nrm1=nrm1, tpow=tpow.astype(np.float64), level=level, zeros=zeros,
    )  # fmt: skip
