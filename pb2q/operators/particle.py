"""Particle-level operator representations as sympy objects."""
from sympy.physics.quantum import (BraBase, KetBase, Operator, OrthogonalBra, OrthogonalKet,
                                   OuterProduct)
from sympy.printing.pretty.stringpict import prettyForm


class Control(OuterProduct):
    """Control operator for particle registers."""
    def __new__(cls, *args):
        if len(args) != 2:
            raise ValueError(f'Number of arguments to Control != 2: {args}')
        if all(arg in (0, 1) for arg in args):
            return super().__new__(cls, OrthogonalKet(args[0]), OrthogonalBra(args[1]))
        if (isinstance(args[0], KetBase) and args[0].args[0] in (0, 1)
                and isinstance(args[1], BraBase) and args[1].args[0] in (0, 1)):
            return super().__new__(cls, *args)

        raise ValueError(f'Invalid constructor argument {args} for control')

    def _eval_adjoint(self):
        return Control(self.args[1].args[0], self.args[0].args[0])

    def _print_operator_name(self, printer, *args):
        return 'Ctrl'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('Ctrl')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathfrak{C}'

    def _sympystr(self, printer, *args):
        return Operator._sympystr(self, printer, *args)

    def _sympyrepr(self, printer, *args):
        return Operator._sympyrepr(self, printer, *args)

    def _pretty(self, printer, *args):
        return Operator._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return r'%s_{%s%s}' % (
            self._print_operator_name_latex(printer, *args),
            self.args[0].args[0],
            self.args[1].args[0]
        )
