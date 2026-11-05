"""U1: exact Ito Gram and orthogonal basis."""

from fractions import Fraction

from pathenhancedfkpinn.signature.ito_basis import E_prod, norm2, orthogonal_basis
from pathenhancedfkpinn.signature.words import basis_for_words, count_words, feature_words


def test_closed_forms():
    B = orthogonal_basis([(0, 1), (1,), (0, 0, 1), (1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    assert B[(0, 1)] == {(0, 1): 1, (1,): Fraction(-1, 2)} and norm2(B[(0, 1)]) == Fraction(1, 12)
    assert B[(0, 0, 1)] == {(0, 0, 1): 1, (0, 1): Fraction(-1, 2), (1,): Fraction(1, 12)}
    assert norm2(B[(0, 0, 1)]) == Fraction(1, 720)
    assert B[(0, 1, 1)] == {(0, 1, 1): 1, (1, 1): Fraction(-1, 3)} and norm2(B[(0, 1, 1)]) == Fraction(1, 36)
    assert B[(1, 0, 1)] == {(1, 0, 1): 1, (0, 1, 1): Fraction(1, 2), (1, 1): Fraction(-1, 2)}
    assert norm2(B[(1, 0, 1)]) == Fraction(1, 48)
    assert norm2(B[(1, 1)]) == Fraction(1, 2) and norm2(B[(1, 1, 1)]) == Fraction(1, 6)
    assert E_prod((1, 2), (2, 1)) == 0


def test_power_norms():
    for n in range(1, 6):
        assert E_prod((1,) * n, (1,) * n) == Fraction(1, __import__("math").factorial(n))


def test_word_counts():
    assert count_words(1, 3, 2) == len(feature_words(1, 3, 2)) == 19
    assert count_words(1, 5, 2) == len(feature_words(1, 5, 2)) == 55
    assert count_words(10, 2, 2) == len(feature_words(10, 2, 2)) == 630
    assert count_words(50, 2, 2) == 15_150


def test_canonical_relabelling_matches_direct():
    """Pattern-cached basis (used for large d) equals direct Gram-Schmidt."""
    words = feature_words(3, 2, 2)
    direct = orthogonal_basis(words)
    cached, norms = basis_for_words(words)
    for w in words:
        assert {v: float(c) for v, c in direct[w].items()} == cached[w]
        assert float(norm2(direct[w])) == norms[w]
