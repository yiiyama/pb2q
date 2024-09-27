# pylint: disable=invalid-name, isinstance-second-argument-not-valid-type
"""States that are also TensorProducts of component system states."""
from sympy import Add, Mul, S, sympify
from sympy.physics.quantum import (BraBase, KetBase, Dagger, State, StateBase, TensorProduct,
                                   OuterProduct)
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
        if (comp_cls := cls.component_class()) is None:
            return True

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

    def _eval_rewrite(self, rule, args, **hints):
        # Overriding TensorProduct._eval_rewrite which hardcodes TensorProduct construction
        return self.func(*args).expand(tensorproduct=True)

    def doit(self, **hints):
        # Overriding TensorProduct.doit
        return self.func(*[item.doit(**hints) for item in self.args])

    def _eval_expand_tensorproduct(self, **hints):
        # Overriding TensorProduct._eval_expand_tensorproduct
        add_args = []
        for iarg, arg in enumerate(self.args):
            if isinstance(arg, Add):
                for aa in arg.args:
                    tp = self.func(*(self.args[:iarg] + (aa,) + self.args[iarg + 1:]))
                    c_part, nc_part = tp.args_cnc()
                    # Check for TensorProduct object: is the one object in nc_part, if any:
                    # (Note: any other object type to be expanded must be added here)
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    add_args.append(Mul(*c_part) * Mul(*nc_part))
                break

        if add_args:
            return Add(*add_args)
        return self

    # TensorProduct._eval_adjoint also hardcodes but _eval_adjoint of this class is inherited from
    # StateBase


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
