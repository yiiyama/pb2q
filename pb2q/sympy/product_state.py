# pylint: disable=invalid-name, isinstance-second-argument-not-valid-type
"""States that are also TensorProducts of component system states."""
from sympy import S
from sympy.physics.quantum import BraBase, KetBase, Dagger, State, OuterProduct
from .product_qexpr import ProductQExpr


class ProductState(State, ProductQExpr):
    """General abstract quantum product state."""
    _op_priority = 20

    @property
    def dual(self):
        """Return the dual state of this one."""
        return self.dual_class()._new_rawargs(self.hilbert_space,
                                              *[arg.adjoint() for arg in self.args])


class ProductKet(ProductState, KetBase):
    """Ket of a quantum product state."""
    @classmethod
    def dual_class(cls):
        return ProductBra

    def _eval_innerproduct(self, bra, **hints):
        # TODO: can use Hilbert space check if that's implemented
        if isinstance(bra, ProductBra):
            if len(bra.args) != len(self.args):
                raise ValueError('Cannot multiply a product ket that has a different number of'
                                 ' components.')

            for bra_arg, arg in zip(bra.args, self.args):
                compres = arg._eval_innerproduct(bra_arg, **hints)
                if compres is None:
                    return None
                if compres == 0:
                    return S.Zero

            return S.One

        return super()._eval_innerproduct(bra, **hints)

    def __mul__(self, other):
        if isinstance(other, ProductBra):
            return ProductOuterProduct(self, other)
        return KetBase.__mul__(self, other)


class ProductBra(ProductState, BraBase):
    """Product Bra in quantum mechanics."""
    @classmethod
    def dual_class(cls):
        return ProductKet


class ProductOuterProduct(OuterProduct):
    """OuterProduct of a ProductKet and a ProductBra"""
    def __new__(cls, *args, **old_assumptions):
        if not (len(args) == 2 and isinstance(args[0], ProductKet)
                and isinstance(args[1], ProductBra)):
            raise ValueError(f'Invalid argument for ProductOuterProduct {args}')
        return super().__new__(cls, *args, **old_assumptions)

    def _apply_operator(self, ket, **options):
        if isinstance(ket, ProductKet):
            ip = self.bra * ket
            if options.get('ip_doit', True):
                ip = ip.doit()
            return ip * self.ket
        return super()._apply_operator(ket, **options)

    def _apply_from_right_to(self, bra, **options):  # pylint: disable=unused-argument
        if isinstance(bra, ProductBra):
            ip = bra * self.ket
            if options.get('ip_doit', True):
                ip = ip.doit()
            return ip * self.bra
        return None

    def _eval_adjoint(self):
        return self.func(Dagger(self.bra), Dagger(self.ket))
