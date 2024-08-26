"""Sympy state classes."""
from sympy import Expr
from sympy.physics.quantum import (Bra, BraBase, Ket, KetBase, InnerProduct, Operator,
                                   OrthogonalBra, OrthogonalKet, OuterProduct, StateBase,
                                   TensorProduct)


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
