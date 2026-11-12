"""Exact Ito inner products and the orthogonal signature basis.

Letters: 0 = time, 1..d = Brownian coordinates. Time-augmented Brownian path W_hat_s = (s, W_s).

Ito quasi-shuffle:
    S^{ua} S^{vb} = (S^{ua} S^v) b + (S^u S^{vb}) a + [a == b != 0] (S^u S^v) 0.
E[S^w_t] = t^k / k! if w = 0^k, else 0.  E[S^u_t S^v_t] = c_uv t^m,  m = z(u)+z(v)+(n(u)+n(v))/2.

Pure Python with exact Fractions. Signax has no Ito inner products, so this module does not use it.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial

Word = tuple[int, ...]


def n_letters(w: Word) -> int:
    """n(w): number of non-zero (Brownian) letters."""
    return sum(1 for c in w if c != 0)


def z_letters(w: Word) -> int:
    """z(w): number of zero (time) letters."""
    return sum(1 for c in w if c == 0)


def strip0(w: Word) -> Word:
    return tuple(c for c in w if c != 0)


@lru_cache(maxsize=None)
def qshuffle(u: Word, v: Word) -> dict[Word, int]:
    """Expansion of S^u * S^v as {word: integer coefficient} (Ito quasi-shuffle recursion)."""
    if not u:
        return {v: 1}
    if not v:
        return {u: 1}
    out: dict[Word, int] = {}
    a, b = u[-1], v[-1]
    for w, c in qshuffle(u, v[:-1]).items():
        out[w + (b,)] = out.get(w + (b,), 0) + c
    for w, c in qshuffle(u[:-1], v).items():
        out[w + (a,)] = out.get(w + (a,), 0) + c
    if a == b and a != 0: # Ito correction
        for w, c in qshuffle(u[:-1], v[:-1]).items():
            out[w + (0,)] = out.get(w + (0,), 0) + c
    return out


def E_prod(u: Word, v: Word) -> Fraction:
    """E[S^u_1 S^v_1] as an exact Fraction (t = 1). For general t multiply by t^m."""
    return sum(
        (Fraction(c, factorial(len(w))) for w, c in qshuffle(u, v).items() if all(x == 0 for x in w)),
        Fraction(0),
    )


def orthogonal_basis(words) -> dict[Word, dict[Word, Fraction]]:
    """Gram-Schmidt within each ~0 class (same non-zero letters in order), ordered by number of
    zeros then lexicographically. Returns {w: {v: coefficient at t = 1}}.
    At time t the coefficient of v in p_hat_w scales as t^(z(w) - z(v))."""
    classes: dict[Word, list[Word]] = {}
    for w in words:
        classes.setdefault(strip0(w), []).append(tuple(w))
    basis: dict[Word, dict[Word, Fraction]] = {}
    for cls in classes.values():
        cls.sort(key=lambda w: (z_letters(w), w))
        done: list[Word] = []
        for w in cls:
            p = {w: Fraction(1)}
            for q in done:
                num = sum(cq * E_prod(w, b) for b, cq in basis[q].items())
                den = sum(c1 * c2 * E_prod(a, b) for a, c1 in basis[q].items() for b, c2 in basis[q].items()
                )
                for b, cq in basis[q].items():
                    p[b] = p.get(b, Fraction(0)) - num / den * cq
            basis[w] = {k: c for k, c in p.items() if c != 0}
            done.append(w)
    return basis


def norm2(p: dict[Word, Fraction]) -> Fraction:
    """<p_hat, p_hat> at t = 1. At time t multiply by t^(2 z(w) + n(w))."""
    return sum((c1 * c2 * E_prod(a, b) for a, c1 in p.items() for b, c2 in p.items()), Fraction(0))
