"""Operators that are also TensorProducts of component system operators."""
from sympy import Add, Mul
from sympy.physics.quantum import Dagger, Operator, TensorProduct
from sympy.physics.quantum.qexpr import QExpr


class ProductOperator(Operator, TensorProduct):
    """General product operator."""

    def __new__(cls, *args):
        if not cls._check_components(args):
            raise ValueError(f'{cls.__name__} must be a product of Operators, got {args}')
        return QExpr.__new__(cls, *args)

    @classmethod
    def _check_components(cls, args):
        for arg in args:
            if isinstance(arg, Add):
                for term in arg.args:
                    if isinstance(term, Mul):
                        _, nc = term.args_cnc()
                        if not all(isinstance(op, Operator) for op in nc):
                            return False
                    elif not isinstance(term, Operator):
                        return False

            elif isinstance(arg, Mul):
                for fact in arg.args:
                    _, nc = fact.args_cnc()
                    if not all(isinstance(op, Operator) for op in nc):
                        return False

            elif not isinstance(arg, Operator):
                return False

        return True

    @classmethod
    def component_class(cls):
        return None

    def _eval_adjoint(self):
        return self.func(*[Dagger(arg) for arg in self.args])

    def _eval_expand_tensorproduct(self, **hints):
        """Distribute TensorProducts across addition."""
        args = self.args
        add_args = []
        for iarg, arg in enumerate(args):
            if isinstance(arg, Add):
                for aa in arg.args:
                    tp = self.func(*args[:iarg] + (aa,) + args[iarg + 1:])
                    c_part, nc_part = tp.args_cnc()
                    # Check for TensorProduct object: is the one object in nc_part, if any:
                    # (Note: any other object type to be expanded must be added here)
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    add_args.append(Mul(*c_part)*Mul(*nc_part))
                break

        if add_args:
            return Add(*add_args)
        return self
