# pylint: disable=consider-using-f-string, invalid-name, unused-argument
"""Field register swaps and symmetrizations."""
from collections.abc import Sequence
from typing import Any, Union
from sympy import Add, Expr, Pow, factorial, sqrt, sympify
from sympy.physics.quantum import HermitianOperator, IdentityOperator, UnitaryOperator
from sympy.printing.pretty.stringpict import prettyForm

from ..states import FieldState
from .field import FieldOperator


class ParticlePermutation(HermitianOperator, UnitaryOperator):
    """Particle-level permutation operator.

    Arguments of this operator must be unique contiguous integers >= 0. i'th particle of the
    returned state will correspond to the particle numbered permutation[i] of the input.
    """
    @classmethod
    def default_args(cls):
        return ('PPERM',)

    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (all(arg.is_integer for arg in args) and set(args) == set(range(len(args)))):
            raise ValueError('ParticlePermutation requires a sequence of unique integers')
        if args == tuple(range(len(args))):
            return IdentityOperator()

        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'PPERM'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('PPERM')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathrm{PPERM}'

    def _print_contents(self, printer, *args):
        return '%s(%s)' % (
            self._print_operator_name(printer, *args),
            ','.join(f'{arg}' for arg in self.args)
        )

    def _print_contents_latex(self, printer, *args):
        return r'%s\left(%s\right)' % (
            self._print_operator_name_latex(printer, *args),
            ','.join(f'{arg}' for arg in self.args)
        )

    @staticmethod
    def order_particles(
        state: Union[FieldState, FieldOperator],
        permutation: Sequence[int]
    ) -> FieldState:
        """Order particles in a FieldState or FieldOperator.

        Args:
            permutation: Sequence of integers specifying the permutation. i'th particle of the
                returned state will correspond to the particle numbered permutation[i] of the input.
        """
        np = len(permutation)
        particle_states = [state.args[permutation[i]] for i in range(np)] + list(state.args[np:])
        return state.func(*particle_states)

    def _apply_operator(self, rhs: Expr, **options) -> Expr:
        if isinstance(rhs, (FieldState, FieldOperator)):
            return self.order_particles(rhs, self.args)  # pylint: disable=no-value-for-parameter
        return super()._apply_operator(rhs, **options)

    def _apply_operator_ParticlePermutation(self, rhs: 'ParticlePermutation', **options) -> Expr:
        new_indices = [rhs_arg[self_arg] for self_arg, rhs_arg in zip(self.args, rhs.args)]
        return ParticlePermutation(*new_indices)


class ParticleSwap(HermitianOperator, UnitaryOperator):
    """Particle-level swap operator implemented as a sympy Operator."""
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

    def _apply_operator(self, rhs: Expr, **options) -> Expr:
        if isinstance(rhs, (FieldState, FieldOperator)):
            return self.swap_particles(rhs, *self.args)  # pylint: disable=no-value-for-parameter
        return super()._apply_operator(rhs, **options)

    def _apply_operator_ParticleSwap(self, rhs: 'ParticleSwap', **options) -> Expr:
        if set(rhs.args) == set(self.args):
            # Note that case rhs.args == self.args is actually covered by _eval_power
            return IdentityOperator()
        indices = list(range(max(rhs.args + self.args) + 1))
        indices[rhs.args[0]], indices[rhs.args[1]] = indices[rhs.args[1]], indices[rhs.args[0]]
        indices[self.args[0]], indices[self.args[1]] = indices[self.args[1]], indices[self.args[0]]
        return ParticlePermutation(*indices)

    def _eval_power(self, exp):
        """Capturing return of unity and converting to I."""
        result = super()._eval_power(exp)
        if result == 1:
            return IdentityOperator()
        return result


class StepSymmetrizerBase(HermitianOperator):
    """Step-(anti)symmetrizer of a bosonic (fermionic) field register.

    S/A_n = 1/sqrt(n) * [I +/- sum_{j=0}^{n-2} P_{n-1, j}]
    """
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

    def _apply_operator(self, rhs: Expr, **options) -> Expr:
        if isinstance(rhs, (FieldState, FieldOperator)):
            new_num = self.args[0]
            result_states = [rhs]
            for ipart in range(new_num - 1):
                result_states.append(
                    self._sign * ParticleSwap.swap_particles(rhs, new_num - 1, ipart)
                )
            return Add(*result_states) / sqrt(new_num)

        return super()._apply_operator(rhs, **options)

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


class SymmetrizerBase(HermitianOperator):
    """(Anti-)symmetrizer of a bosonic (fermionic) field register.

    SS/AA_n = 1/sqrt(n!) * (sum of permutations)
    """
    _sign = 0

    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (len(args) == 1 and args[0].is_integer and args[0] > 0):
            raise ValueError('Step(Anti)Symmetrizer requires one integer argument (number of'
                             ' particles)')
        if args[0] == 1:
            return IdentityOperator()

        return super().__new__(cls, *args, **kwargs)

    def _print_contents(self, printer, *args):
        return f'{self._print_operator_name(printer, *args)}({self.args[0]})'

    def _print_contents_latex(self, printer, *args):
        return f'{self._print_operator_name_latex(printer, *args)}({self.args[0]})'

    def _apply_operator(self, rhs: Expr, **options) -> Expr:
        if isinstance(rhs, FieldState):
            result_states = []
            sign = 1
            for perm in generate_perm(range(self.args[0])):
                result_states.append(sign * ParticlePermutation.order_particles(rhs, perm))
                sign *= self._sign

            return Add(*result_states) / factorial(self.args[0])

        return super()._apply_operator(rhs, **options)

    def _eval_power(self, exp):
        if exp.is_integer and exp.is_positive:
            return self
        return super()._eval_power(exp)

    def _eval_rewrite(self, rule, args, **hints):
        num = self.args[0]
        if rule == ParticlePermutation:
            if num == 1:
                return IdentityOperator()

            ops = [(self._sign ** ip) * ParticlePermutation(perm)
                   for ip, perm in enumerate(generate_perm(range(num)))]
            return Add(*ops) / factorial(num)

        return None


def generate_perm(seq: Sequence, _k=None) -> list[tuple[Any]]:
    """Generate all permutations of seq using the Heap's algorithm.

    Because each element in the resulting list of permutations are obtained by swapping two elements
    of the previous element, we are guaranteed to have alternating permutation signs.
    """
    if _k is None:
        seq = list(seq)
        _k = len(seq)

    if _k == 1:
        return [tuple(seq)]

    result = generate_perm(seq, _k - 1)

    if _k % 2 == 0:
        indices = range(_k - 1)
    else:
        indices = [0] * (_k - 1)

    for idx in indices:
        seq[idx], seq[_k - 1] = seq[_k - 1], seq[idx]
        result.extend(generate_perm(seq, _k - 1))

    return result
