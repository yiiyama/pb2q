"""Field register swaps and symmetrizations."""
from typing import Union
from sympy import Add, Expr, Pow, sqrt, sympify
from sympy.physics.quantum import IdentityOperator, Operator
from sympy.printing.pretty.stringpict import prettyForm

from ..states import FieldState
from ..sympy import ProductKet
from .field import FieldOperator


class ParticleSwap(Operator):
    """Particle-level swap operator implemented as a sympy Operator."""
    is_hermitian = True
    is_unitary = True

    @classmethod
    def default_args(cls):
        return ('PSWAP',)

    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (len(args) == 2 and all(arg.is_integer for arg in args)):
            raise ValueError('ParticleSwap requires two integer arguments (index1, index2),'
                             f' got {args}')
        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'PSWAP'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('PSWAP')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathrm{PSWAP}'

    def _print_contents(self, printer, *args):
        return f'{self._print_operator_name(printer, *args)}({self.args[0]},{self.args[1]})'

    def _print_contents_latex(self, printer, *args):
        return r'%s\left({%s}, {%s}\right)' % (
            (self._print_operator_name_latex(printer, *args),) + self.args
        )

    @staticmethod
    def swap_particles(
        state: Union[FieldState, FieldOperator],
        index1: int,
        index2: int
    ) -> FieldState:
        particle_states = list(state.args)
        particle_states[index1] = state.args[index2]
        particle_states[index2] = state.args[index1]
        return state.func(*particle_states)

    def _apply_operator(self, state: FieldState, **options) -> ProductKet:
        return self.swap_particles(state, *self.args)  # pylint: disable=no-value-for-parameter

    def _eval_power(self, exp):
        return IdentityOperator() if exp % 2 == 0 else self

    def _eval_inverse(self):
        return self


class StepSymmetrizerBase(Operator):
    """Step-(anti)symmetrizer of a bosonic (fermionic) field register."""
    is_hermitian = True
    _sign = 0

    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (len(args) == 1 and args[0].is_integer and args[0] > 0):
            raise ValueError('Step(Anti)Symmetrizer requires one integer argument (updated number'
                             ' of particles)')
        if args[0] == 1:
            return IdentityOperator()

        return super().__new__(cls, *args, **kwargs)

    def _print_contents(self, printer, *args):
        return f'{self._print_operator_name(printer, *args)}({self.args[0]}<-{self.args[0]-1})'

    def _print_contents_latex(self, printer, *args):
        return r'%s\left({%s}\leftarrow{%s}\right)' % (
            self._print_operator_name_latex(printer, *args), self.args[0], self.args[0] - 1
        )

    def _apply_operator(self, state: FieldState, **options) -> Expr:
        new_num = self.args[0]
        result_states = [state]
        for ipart in range(new_num - 1):
            result_states.append(
                self._sign * ParticleSwap.swap_particles(state, new_num - 1, ipart)
            )
        return Add(*result_states) / sqrt(new_num)

    def _eval_power(self, exp):
        return Pow(sqrt(self.args[0]), exp - 1) * self

    def _eval_rewrite(self, rule, args, **hints):
        new_num = self.args[0]  # pylint: disable=unbalanced-tuple-unpacking
        if rule == ParticleSwap:
            if new_num == 1:
                return IdentityOperator()

            ops = [IdentityOperator()]
            ops += [self._sign * ParticleSwap(new_num - 1, ipart) for ipart in range(new_num - 1)]
            return Add(*ops) / sqrt(new_num)
        return None


class StepSymmetrizer(StepSymmetrizerBase):
    """Step-symmetrizer of a bosonic field register."""
    _sign = 1

    def _print_operator_name(self, printer, *args):
        return 'S'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('S')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathcal{S}'


class StepAntisymmetrizer(StepSymmetrizerBase):
    """Step-antisymmetrizer of a fermionic field register."""
    _sign = -1

    def _print_operator_name(self, printer, *args):
        return 'A'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('A')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathcal{A}'
