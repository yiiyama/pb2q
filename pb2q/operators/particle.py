# pylint: disable=invalid-name, unused-argument
"""Particle-level operator representations as sympy objects."""
from sympy import Expr, S
from sympy.physics.quantum import (BraBase, Dagger, KetBase, HermitianOperator, Operator,
                                   OrthogonalBra, OrthogonalKet, OuterProduct)
from sympy.printing.pretty.stringpict import prettyForm

from ..momentum import Momentum
from ..states import ParticleBra, ParticleKet, ParticleState


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


class Projection(HermitianOperator):
    """Projector to the occupied state of a particle register."""
    is_unitary = False

    projection = None

    def __new__(cls):
        return super().__new__(cls)

    @classmethod
    def default_args(cls):
        return ()

    def _apply_operator_ParticleKet(self, ket, **options):
        if ket.is_null_state:
            return S.Zero if self.projection else ket
        return ket if self.projection else S.Zero

    def _apply_operator_ParticleOuterProduct(self, other, **options):
        if other.ket.is_null_state:
            return S.Zero if self.projection else other
        return other if self.projection else S.Zero

    def _apply_from_right_to(self, other, **options):
        if isinstance(other, ParticleBra):
            bra = other
        elif isinstance(other, ParticleOuterProduct):
            bra = other.bra
        else:
            return None

        if bra.is_null_state:
            return S.Zero if self.projection else other
        return other if self.projection else S.Zero

    def _print_contents(self, printer, *args):
        return f'P{self.projection}'

    def _print_contents_pretty(self, printer, *args):
        return prettyForm(f'P{self.projection}')

    def _print_contents_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathbb{P}_{%d}' % self.projection

    def _eval_power(self, exp):
        if exp.is_integer and exp.is_positive:
            return self
        return super()._eval_power(exp)


class PresenceProjection(Projection):
    """Projector to the occupied state of a particle register."""
    projection = 1

    def _apply_operator_AbsenceProjection(self, other, **options):
        return S.Zero


class AbsenceProjection(Projection):
    """Projector to the unoccupied state of a particle register."""
    projection = 0

    def _apply_operator_PresenceProjection(self, other, **options):
        return S.Zero


class ParticleOuterProduct(OuterProduct):
    """OuterProduct of a ParticleKet and a ParticleBra."""
    def __new__(cls, *args, **old_assumptions):
        if not (len(args) == 2 and isinstance(args[0], ParticleKet)
                and isinstance(args[1], ParticleBra)):
            raise ValueError(f'Invalid argument for ProductOuterProduct {args}')
        if args[0].is_null_state and args[1].is_null_state:
            return AbsenceProjection()

        return super().__new__(cls, *args, **old_assumptions)

    def _apply_operator_ParticleKet(self, ket, **options):
        ip = self.bra * ket
        if options.get('ip_doit', True):
            ip = ip.doit()
        return ip * self.ket

    def _apply_operator_PresenceProjection(self, other, **options):
        if self.bra.is_null_state:
            return S.Zero
        return self

    def _apply_operator_AbsenceProjection(self, other, **options):
        if self.bra.is_null_state:
            return self
        return S.Zero

    def _apply_from_right_to(self, other, **options):
        ip_doit = options.get('ip_doit', True)

        if isinstance(other, ParticleBra):
            ip = other * self.ket
            if ip_doit:
                ip = ip.doit()
            return ip * self.bra
        if isinstance(other, ParticleOuterProduct):
            ip = other.bra * self.ket
            if ip_doit:
                ip = ip.doit()
            return ip * (other.ket * self.bra)
        if isinstance(other, PresenceProjection):
            if self.ket.is_null_state:
                return S.Zero
            return self
        if isinstance(other, AbsenceProjection):
            if self.ket.is_null_state:
                return self
            return S.Zero

        return None

    def _eval_adjoint(self):
        return self.func(Dagger(self.bra), Dagger(self.ket))


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

    def _apply_operator(self, rhs: Expr, **options) -> Expr:
        if isinstance(rhs, ParticleState):
            return Momentum(rhs.args[1].args[0], mass=self.args[0]).energy() * rhs
        return super()._apply_operator(rhs, **options)
