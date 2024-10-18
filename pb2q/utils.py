"""Common utility functions."""
from collections.abc import Sequence
from typing import Any


def generate_perm(seq: Sequence, _k=None) -> list[tuple[Any]]:
    """Generate all permutations of seq using the Heap's algorithm.

    Because each element in the resulting list of permutations are obtained by swapping two elements
    of the previous element, we are guaranteed to have alternating permutation signs.
    """
    if _k is None:
        seq = list(seq)
        _k = len(seq)

    if _k == 1:
        return [tuple(seq)]

    result = generate_perm(seq, _k - 1)

    if _k % 2 == 0:
        indices = range(_k - 1)
    else:
        indices = [0] * (_k - 1)

    for idx in indices:
        seq[idx], seq[_k - 1] = seq[_k - 1], seq[idx]
        result.extend(generate_perm(seq, _k - 1))

    return result
