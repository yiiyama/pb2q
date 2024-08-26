"""Sympy state classes."""
from typing import Union
import numpy as np
from sympy import Expr, Mul
from sympy.physics.quantum import (Bra, BraBase, Ket, KetBase, IdentityOperator, InnerProduct,
                                   Operator, OrthogonalBra, OrthogonalKet, OuterProduct, StateBase,
                                   TensorProduct)
from sympy.printing.pretty.stringpict import prettyForm


def fn(self, bra, **options):  # pylint: disable=unused-argument
    return InnerProduct(bra, self.ket) * self.bra


OuterProduct._apply_from_right_to = fn


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


def to_product_state(product: TensorProduct) -> ProductState:
    if all(isinstance(arg, OrthogonalKet) for arg in product.args):
        return OrthogonalProductKet(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, KetBase) for arg in product.args):
        return ProductKet(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, OrthogonalBra) for arg in product.args):
        return OrthogonalProductBra(*sum((arg.args for arg in product.args), ()))
    if all(isinstance(arg, BraBase) for arg in product.args):
        return ProductBra(*sum((arg.args for arg in product.args), ()))
    raise ValueError('Cannot convert argument to product state')


def to_tensor_product(state: StateBase) -> TensorProduct:
    if isinstance(state, OrthogonalKet):
        return TensorProduct(*[OrthogonalKet(arg) for arg in state.args])
    if isinstance(state, KetBase):
        return TensorProduct(*[Ket(arg) for arg in state.args])
    if isinstance(state, OrthogonalBra):
        return TensorProduct(*[OrthogonalBra(arg) for arg in state.args])
    if isinstance(state, BraBase):
        return TensorProduct(*[Bra(arg) for arg in state.args])
    raise ValueError('Cannot convert argument to TensorProduct')


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

    def __new__(cls, *args, **kwargs):
        if len(args) <= 1 or args == tuple(sorted(args)):
            return IdentityOperator()
        if len(set(args)) != len(args):
            raise ValueError('Permutation indices must be unique')
        if len(args) == 2:
            return super().__new__(SwapOperator, *args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

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

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathrm{PERM}'

    def _apply_operator(self, ket: Union[KetBase, TensorProduct], **options) -> ProductKet:
        if isinstance(ket, KetBase):
            ket = to_tensor_product(ket)
        if not (isinstance(ket, TensorProduct)
                and all(isinstance(arg, KetBase) and len(arg.args) == 1 for arg in ket.args)):
            raise ValueError('Argument is not a product state')
        if len(ket.args) <= max(self.args):
            raise ValueError('State is inconsistent with the permutation')
        new_kets = list(ket.args)
        for source, dest in zip(self.args, np.sort(self.args)):
            new_kets[dest] = ket.args[source]
        return to_product_state(TensorProduct(*new_kets))

    def _eval_inverse(self):
        swaps = self.to_swaps()
        return PermutationOperator.from_swaps(Mul(*swaps.args[::-1]))


class SwapOperator(PermutationOperator):  # pylint: disable=abstract-method
    """Register swap operator."""
    is_hermitian = True

    @classmethod
    def default_args(cls):
        return ('SWAP',)

    def __new__(cls, *args, **kwargs):
        if not (len(args) == 2 and all(isinstance(arg, int) for arg in args)):
            raise ValueError('SwapOperator requires two integer arguments')
        return super().__new__(cls, *tuple(sorted(args)[::-1]), **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'SWAP'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('SWAP')

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathrm{SWAP}'

    def _eval_inverse(self):
        return self
