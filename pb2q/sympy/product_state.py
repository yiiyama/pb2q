# pylint: disable=invalid-name, isinstance-second-argument-not-valid-type
"""States that are also TensorProducts of component system states."""
from sympy import Add, Mul, S, sympify
from sympy.physics.quantum import BraBase, KetBase, State, StateBase, TensorProduct
from sympy.physics.quantum.qexpr import QExpr


class ProductState(State, TensorProduct):
    """General abstract quantum product state."""
    _op_priority = 20

    def __new__(cls, *args):
        args = sympify(args)
        if any(arg == 0 for arg in args):
            return S.Zero
        if not cls._check_components(args):
            raise ValueError(f'{cls.__name__} components must be {cls.component_class().__name__},'
                             f' got {args}')

        return QExpr.__new__(cls, *args)

    @classmethod
    def _check_components(cls, args):
        comp_cls = cls.component_class()
        for arg in args:
            if isinstance(arg, Add):
                for term in arg.args:
                    if isinstance(term, Mul):
                        _, nc = term.args_cnc()
                        if not all(isinstance(op, comp_cls) for op in nc):
                            return False
                    elif not isinstance(term, comp_cls):
                        return False

            elif isinstance(arg, Mul):
                for fact in arg.args:
                    _, nc = fact.args_cnc()
                    if not all(isinstance(op, comp_cls) for op in nc):
                        return False

            elif not isinstance(arg, comp_cls):
                return False

        return True

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
