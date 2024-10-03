# pylint: disable=invalid-name, unused-argument
"""Free evolution operators."""
from sympy import Add, Expr, I, S, exp, sympify
from sympy.physics.quantum import UnitaryOperator
from sympy.printing.pretty.stringpict import prettyForm
from ..states import ParticleKet
from .particle import PresenceProjection, AbsenceProjection


class FreeEvolution(UnitaryOperator):
    """Abstract free evolution operator."""
    def __new__(cls, time, **kwargs):
        time = sympify(time)
        if not time.is_real:
            raise ValueError('ParticleEvolution operator accepts a real number as argument, got'
                             f' {time}')
        return super().__new__(cls, time, **kwargs)

    @classmethod
    def default_args(cls):
        return ('t',)

    @property
    def time(self):
        return self.args[0]


class ParticleFreeEvolution(FreeEvolution):
    """Particle level free evolution operator."""
    def _print_operator_name(self, printer, *args):
        return 'uf'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('uf')

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathfrak{u}_{f}'

    def _apply_operator_ParticleKet(self, rhs: ParticleKet, **options) -> Expr:
        return exp(-I * self.time * rhs.momentum.energy) * rhs

    def _apply_operator_PresenceProjection(self, rhs: PresenceProjection, **options) -> Expr:
        return self

    def _apply_operator_AbsenceProjection(self, rhs: AbsenceProjection, **options) -> Expr:
        return S.Zero


class FieldFreeEvolution(FreeEvolution):
    """Field-level free evolution operator."""
    def _print_operator_name(self, printer, *args):
        return 'Uf'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('Uf')

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathcal{U}_{f}'

    def _apply_operator_FieldKet(self, rhs: ParticleKet, **options) -> Expr:
        exponents = []
        for particle in rhs.args:
            if particle.is_null_state:
                break
            exponents.append(particle.momentum.energy)

        return exp(-I * self.time * Add(*exponents)) * rhs
