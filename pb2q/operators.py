"""Basic field theory operators."""
from sympy import Add, Expr, sqrt
from sympy.physics.quantum import IdentityOperator
from .sympy import SwapOperator


def step_symmetrizer(num_registers: int) -> Expr:
    if num_registers == 1:
        return IdentityOperator()
    ops = [IdentityOperator()]
    ops += [SwapOperator(num_registers - 1, ireg) for ireg in range(num_registers - 1)]
    return Add(*ops) / sqrt(num_registers)
