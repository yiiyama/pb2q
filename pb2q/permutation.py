"""Common utility functions."""
from collections.abc import Sequence
from typing import Any
from sympy import Expr


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


def order_args(expr: Expr, permutation: Sequence[int]) -> Expr:
    """Order args of an expression.

    Args:
        permutation: Sequence of integers specifying the permutation. i'th arg of the
            returned expr will correspond to the arg numbered permutation[i] of the input.
    """
    np = len(permutation)
    new_args = [expr.args[permutation[i]] for i in range(np)] + list(expr.args[np:])
    return expr.func(*new_args)


def swap_args(expr: Expr, index1: int, index2: int) -> Expr:
    new_args = list(expr.args)
    new_args[index1] = expr.args[index2]
    new_args[index2] = expr.args[index1]
    return expr.func(*new_args)
