"""Particle-level operator representations as sympy objects."""
from sympy import Expr
from sympy.physics.quantum import (BraBase, KetBase, Operator, OrthogonalBra, OrthogonalKet,
                                   OuterProduct)
from sympy.printing.pretty.stringpict import prettyForm

from ..momentum import Momentum
from ..states import ParticleState


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


class ParticleEnergy(Operator):
    """Particle-level free Hamiltonian."""
    is_hermitian = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(f'ParticleEnergy takes one argument (mass), got {args}')

        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'hpart'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('hpart')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'h_{\mathrm{part}}'

    def _print_contents(self, printer, *args):
        return self._print_operator_name(printer, *args)

    def _print_contents_latex(self, printer, *args):
        return self._print_operator_name_latex(printer, *args)

    def _apply_operator(self, state: ParticleState, **options) -> Expr:
        return Momentum(state.args[1].args[0], mass=self.args[0]).energy() * state
