"""Sympy state classes."""
from typing import Union
import numpy as np
from sympy import Expr, Mul
from sympy.physics.quantum import (Bra, BraBase, Ket, KetBase, IdentityOperator, InnerProduct,
                                   Operator, OrthogonalBra, OrthogonalKet, OuterProduct, StateBase,
                                   TensorProduct)
from sympy.printing.pretty.stringpict import prettyForm


class ProductState(StateBase):  # pylint: disable=abstract-method
    """General abstract quantum product state."""
    _op_priority = 20


class ProductKet(ProductState, KetBase):  # pylint: disable=abstract-method
    """Ket that is interpreted as a tensor product of its argument kets."""

    @classmethod
    def dual_class(cls):
        return ProductBra

    @classmethod
    def atomic_class(cls):
        return Ket

    def __mul__(self, other):
        if (isinstance(other, TensorProduct)
                and all(isinstance(arg, BraBase) for arg in other.args)
                and sum(len(arg.args) for arg in other.args) == len(self.args)):
            other_prod = self.dual_class()(*sum((arg.args for arg in other.args), ()))
            return OuterProduct(self, other_prod)

        if isinstance(other, ProductBra) and len(other.args) == len(self.args):
            return OuterProduct(self, other)

        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, TensorProduct):
            if (all(isinstance(arg, BraBase) for arg in other.args)
                    and sum(len(arg.args) for arg in other.args) == len(self.args)):
                other_prod = self.dual_class()(*sum((arg.args for arg in other.args), ()))
                return InnerProduct(other_prod, self)

            if (all(isinstance(arg, Operator) for arg in other.args)
                    and (nop := len(other.args)) <= len(self.args)):
                args = [op * self.atomic_class()(ket_arg)
                        for op, ket_arg in zip(other.args[:-1], self.args[:nop - 1])]
                # Last op is assumed to apply to all remaining args
                args.append(other.args[-1] * self.atomic_class()(*self.args[nop - 1:]))
                return TensorProduct(*args)

        elif isinstance(other, ProductBra) and len(other.args) == len(self.args):
            return InnerProduct(other, self)

        return Expr.__rmul__(self, other)


class ProductBra(ProductState, BraBase):  # pylint: disable=abstract-method
    """Product Bra in quantum mechanics."""
    @classmethod
    def dual_class(cls):
        return ProductKet

    @classmethod
    def atomic_class(cls):
        return Bra

    def __mul__(self, other):
        if isinstance(other, TensorProduct):
            if (all(isinstance(arg, KetBase) for arg in other.args)
                    and sum(len(arg.args) for arg in other.args) == len(self.args)):
                other_prod = self.dual_class()(*sum((arg.args for arg in other.args), ()))
                return InnerProduct(self, other_prod)

            if (all(isinstance(arg, Operator) for arg in other.args)
                    and (nop := len(other.args)) <= len(self.args)):
                args = [self.atomic_class()(bra_arg) * op
                        for op, bra_arg in zip(other.args[:-1], self.args[:nop - 1])]
                # Last op is assumed to apply to all remaining args
                args.append(self.atomic_class()(*self.args[nop - 1:]) * other.args[-1])
                return TensorProduct(*args)

        elif isinstance(other, ProductKet) and len(other.args) == len(self.args):
            return InnerProduct(self, other)

        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        if (isinstance(other, TensorProduct)
                and all(isinstance(arg, KetBase) for arg in other.args)
                and sum(len(arg.args) for arg in other.args) == len(self.args)):
            other_prod = self.dual_class()(*sum((arg.args for arg in other.args), ()))
            return OuterProduct(other_prod, self)
        if isinstance(other, ProductKet) and len(other.args) == len(self.args):
            return OuterProduct(other, self)
        return Expr.__rmul__(self, other)


class OrthogonalProductKet(OrthogonalKet, ProductKet):  # pylint: disable=abstract-method
    """Orthogonal product ket."""
    @classmethod
    def dual_class(cls):
        return OrthogonalProductBra

    @classmethod
    def atomic_class(cls):
        return OrthogonalKet


class OrthogonalProductBra(OrthogonalBra, ProductBra):  # pylint: disable=abstract-method
    """Orthogonal product ket."""
    @classmethod
    def dual_class(cls):
        return OrthogonalProductKet

    @classmethod
    def atomic_class(cls):
        return OrthogonalBra


def to_product_state(product: TensorProduct):
    if all(isinstance(arg, OrthogonalKet) for arg in product.args):
        return OrthogonalProductKet(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, KetBase) for arg in product.args):
        return ProductKet(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, OrthogonalBra) for arg in product.args):
        return OrthogonalProductBra(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, BraBase) for arg in product.args):
        return ProductBra(*sum((arg.args for arg in product.args), ()))
    return product


class PermutationOperator(Operator):  # pylint: disable=abstract-method
    """Register permutation operator.

    Register indices specification works similarly to the numpy `transpose()` function, but unlike
    the latter, does not require un-permuted indices to be present, with the understanding that
    only the specified registers move. The destination positions are deduced by sorting the given
    indices.

    For example, indices (4, 1, 2) creates a permutation where the original registers in positions
    4, 1, and 2 moves to positions 1, 2, and 4.

    Args:
        indices: Original positions of the registers to move, given in permuted order.
    """
    is_unitary = True

    @classmethod
    def default_args(cls):
        return ('PERM',)

    def __new__(cls, indices: tuple[int, ...]):
        if len(indices) <= 1:
            return IdentityOperator()
        if len(indices) == 2:
            if indices[0] == indices[1]:
                return IdentityOperator()
            return SwapOperator(*indices)
        return super().__new__(cls, indices)

    def __init__(self, indices: tuple[int, ...]):
        super().__init__(*indices)

    def to_swaps(self) -> Mul:
        slots = np.sort(self.args)
        state = list(range(max(self.args) + 1))
        swap_ops = []
        for slot, arg in zip(slots, self.args):
            if arg == state[slot]:
                continue
            target = state.index(arg)
            swap_ops.insert(0, SwapOperator(slot, target))
            state[target] = state[slot]
            state[slot] = arg
        return Mul(*swap_ops)

    @classmethod
    def from_swaps(cls, swap_ops: Mul) -> 'PermutationOperator':
        max_arg = max(sum((op.args for op in swap_ops.args), ()))
        state = list(range(max_arg + 1))
        for swap_op in swap_ops.args[::-1]:
            tmp = state[swap_op[0]]
            state[swap_op[0]] = state[swap_op[1]]
            state[swap_op[1]] = tmp
        indices = tuple(i for i, j in zip(state, range(max_arg + 1)) if i != j)
        return PermutationOperator(indices)

    def _print_operator_name(self, printer, *args):
        return 'PERM'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('PERM')

    def _apply_operator(self, ket: Union[Ket, TensorProduct], **options) -> ProductKet:
        if isinstance(ket, TensorProduct):
            ket = to_product_state(ket)
        if not (isinstance(ket, Ket) and len(ket.args) > max(self.args)):
            raise ValueError('Ket is inconsistent with the permutation')
        new_args = list(ket.args)
        for source, dest in zip(self.args, np.sort(self.args)):
            new_args[dest] = ket.args[source]
        return to_product_state(TensorProduct(*new_args))

    def _eval_inverse(self):
        swaps = self.to_swaps()
        return PermutationOperator.from_swaps(Mul(*swaps.args[::-1]))


class SwapOperator(PermutationOperator):  # pylint: disable=abstract-method
    """Register swap operator."""
    is_hermitian = True

    @classmethod
    def default_args(cls):
        return ('SWAP',)

    def __new__(cls, ireg1: int, ireg2: int):
        super().__new__(cls, (max(ireg1), min(ireg2)))

    def __init__(self, ireg1: int, ireg2: int):
        super().__init__((max(ireg1), min(ireg2)))

    def _print_operator_name(self, printer, *args):
        return 'SWAP'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('SWAP')

    def _eval_inverse(self):
        return self
