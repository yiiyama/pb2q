# pylint: disable=invalid-name, isinstance-second-argument-not-valid-type
"""QExpr that are also TensorProducts of component objects."""
from sympy import Add, Mul, S, sympify
from sympy.physics.quantum import Dagger, TensorProduct
from sympy.physics.quantum.qexpr import QExpr


class ProductQExpr(QExpr, TensorProduct):
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
    def component_class(cls) -> type[QExpr]:
        return None

    def _eval_adjoint(self):
        return self.func(*[Dagger(arg) for arg in self.args])

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
