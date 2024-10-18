# pylint: disable=consider-using-f-string, invalid-name, unused-argument
"""Field register swaps and symmetrizations."""
from sympy import Add, Expr, factorial, sqrt, sympify
from sympy.physics.quantum import HermitianOperator, IdentityOperator, UnitaryOperator
from sympy.printing.pretty.stringpict import prettyForm

from ..states import FieldKet, FieldBra, AntisymmetricFieldKet, SymmetricFieldKet
from ..permutation import generate_perm, order_args, swap_args


class ParticlePermutation(HermitianOperator, UnitaryOperator):
    """Particle-level permutation operator.

    Arguments of this operator must be unique contiguous integers >= 0. i'th particle of the
    returned state will correspond to the particle numbered permutation[i] of the input.
    """
    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (all(arg.is_integer for arg in args) and set(args) == set(range(len(args)))):
            raise ValueError('ParticlePermutation requires a sequence of unique integers')
        if args == tuple(range(len(args))):
            return IdentityOperator()

        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def default_args(cls):
        return ('PPERM',)

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

    def _apply_operator_FieldKet(self, rhs: FieldKet, **options) -> Expr:
        return order_args(rhs, self.args)  # pylint: disable=no-value-for-parameter

    def _apply_operator_ParticlePermutation(self, rhs: 'ParticlePermutation', **options) -> Expr:
        new_indices = [rhs_arg[self_arg] for self_arg, rhs_arg in zip(self.args, rhs.args)]
        return ParticlePermutation(*new_indices)

    def _apply_from_right_to(self, lhs: Expr, **options) -> Expr:
        if isinstance(lhs, FieldBra):
            return order_args(lhs, self.args)
        return None


class ParticleSwap(HermitianOperator, UnitaryOperator):
    """Particle-level swap operator implemented as a sympy Operator."""
    def __new__(cls, index1, index2, **kwargs):
        args = sympify((index1, index2))
        if not all(arg.is_integer for arg in args):
            raise ValueError('ParticleSwap requires two integer arguments (index1, index2), got'
                             f' {args}')
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def default_args(cls):
        return ('PSWAP',)

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

    def _apply_operator_FieldKet(self, rhs: FieldKet, **options) -> Expr:
        return swap_args(rhs, self.args[0], self.args[1])

    def _apply_operator_ParticleSwap(self, rhs: 'ParticleSwap', **options) -> Expr:
        if set(rhs.args) == set(self.args):
            # Note that case rhs.args == self.args is actually covered by _eval_power
            return IdentityOperator()
        indices = list(range(max(rhs.args + self.args) + 1))
        indices[rhs.args[0]], indices[rhs.args[1]] = indices[rhs.args[1]], indices[rhs.args[0]]
        indices[self.args[0]], indices[self.args[1]] = indices[self.args[1]], indices[self.args[0]]
        return ParticlePermutation(*indices)

    def _apply_from_right_to(self, lhs: Expr, **options) -> Expr:
        if isinstance(lhs, FieldBra):
            return swap_args(lhs, self.args[0], self.args[1])
        return None

    def _eval_power(self, exp):
        """Capturing return of unity and converting to I."""
        if (result := super()._eval_power(exp)) == 1:
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

    def _apply_operator_FieldKet(self, rhs: FieldKet, **options) -> Expr:
        new_num = self.args[0]
        result_states = [rhs]
        for ipart in range(new_num - 1):
            result_states.append(
                self._sign * swap_args(rhs, new_num - 1, ipart)
            )
        return Add(*result_states) / sqrt(new_num)

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

    def _apply_operator_SymmetricFieldKet(self, ket: SymmetricFieldKet, **options) -> Expr:
        if ket.right_filled and ket.nocc >= self.args[0]:
            return sqrt(self.args[0]) * ket
        raise ValueError('StepSymmetrizer violates right filling')

    def _print_operator_name(self, printer, *args):
        return 'S'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('S')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathcal{S}'


class StepAntisymmetrizer(StepSymmetrizerBase):
    """Step-antisymmetrizer of a fermionic field register."""
    _sign = -1

    def _apply_operator_AntisymmetricFieldKet(self, ket: AntisymmetricFieldKet, **options) -> Expr:
        if ket.right_filled and ket.nocc >= self.args[0]:
            return sqrt(self.args[0]) * ket
        raise ValueError('StepAntisymmetrizer violates right filling')

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

    def _apply_operator_FieldKet(self, rhs: FieldKet, **options) -> Expr:
        result_states = []
        sign = 1
        for perm in generate_perm(range(self.args[0])):
            result_states.append(sign * order_args(rhs, perm))
            sign *= self._sign

        return Add(*result_states) / sqrt(factorial(self.args[0]))

    def _eval_power(self, exp):
        if exp.is_integer and exp.is_positive:
            return self * sqrt(self.args[0])
        return super()._eval_power(exp)

    def _eval_rewrite(self, rule, args, **hints):
        num = self.args[0]
        if rule == ParticlePermutation:
            if num == 1:
                return IdentityOperator()

            ops = [(self._sign ** ip) * ParticlePermutation(perm)
                   for ip, perm in enumerate(generate_perm(range(num)))]
            return Add(*ops) / sqrt(factorial(num))

        return None


class Symmetrizer(SymmetrizerBase):
    """Full symmetrizer."""
    _sign = 1

    def _print_operator_name(self, printer, *args):
        return 'SS'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('SS')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\bar{\mathcal{S}}'


class Antisymmetrizer(SymmetrizerBase):
    """Full antisymmetrizer."""
    _sign = -1

    def _print_operator_name(self, printer, *args):
        return 'AA'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('AA')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\bar{\mathcal{A}}'
