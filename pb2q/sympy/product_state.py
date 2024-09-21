# pylint: disable=invalid-name
"""States that are also TensorProducts of component system states."""
from sympy import S, sympify
from sympy.physics.quantum import BraBase, KetBase, State, StateBase, TensorProduct
from sympy.physics.quantum.qexpr import QExpr


class ProductState(State, TensorProduct):
    """General abstract quantum product state."""
    _op_priority = 20

    def __new__(cls, *args):
        args = sympify(args)
        comp_cls = cls.component_class()
        # pylint: disable-next=isinstance-second-argument-not-valid-type
        if comp_cls and not all(isinstance(arg, comp_cls) for arg in args):
            raise ValueError(f'Components of {cls.__name__} must be {comp_cls.__name__}')

        return QExpr.__new__(cls, *args)

    @classmethod
    def component_class(cls) -> type[StateBase]:
        return None

    @property
    def dual(self):
        """Return the dual state of this one."""
        return self.dual_class()._new_rawargs(self.hilbert_space, *[arg.dual for arg in self.args])


class ProductKet(ProductState, KetBase):
    """Ket of a quantum product state."""
    @classmethod
    def dual_class(cls):
        return ProductBra

    def _eval_innerproduct(self, bra, **hints):
        # TODO: can use Hilbert space check if that's implemented
        if isinstance(bra, ProductBra):
            print('bra is product')
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

        return super()._eval_innerproduct(bra)


class ProductBra(ProductState, BraBase):
    """Product Bra in quantum mechanics."""
    @classmethod
    def dual_class(cls):
        return ProductKet
