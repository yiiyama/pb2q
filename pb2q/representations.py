"""State and operator representations as sympy objects."""

from collections.abc import Sequence
from sympy import Add, Expr, Pow, sqrt, sympify
from sympy.core.containers import Tuple
from sympy.physics.quantum import (Dagger, IdentityOperator, KetBase, Operator, OrthogonalBra,
                                   OrthogonalKet, OuterProduct, TensorProduct)
from sympy.printing.pretty.stringpict import prettyForm

from .sympy import OrthogonalProductKet, ProductKet


class UniverseState(TensorProduct, KetBase):
    """TensorProduct of FieldStates."""
    def __new__(cls, *args):
        if not all(isinstance(arg, FieldState) for arg in args):
            raise ValueError(f'FieldState must be a product of ParticleStates, got {args}')
        return super().__new__(cls, *args)

    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % arg._sympystr(printer, *args)) for arg in self.args)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if printer._use_unicode:
                next_pform = prettyForm(*next_pform.parens(
                    left='\N{MATHEMATICAL LEFT WHITE SQUARE BRACKET}',
                    right='\N{MATHEMATICAL LEFT WHITE SQUARE BRACKET}')
                )
            else:
                next_pform = prettyForm(*next_pform.parens(left='[', right=']'))
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        return r'\otimes'.join(fr'\left\llbracket {arg._latex(printer, *args)} \right\rrbracket'
                               for arg in self.args)


class FieldState(TensorProduct, KetBase):
    """TensorProduct of ParticleStates."""
    def __new__(cls, *args):
        if not all(isinstance(arg, ParticleState) for arg in args):
            raise ValueError(f'FieldState must be a product of ParticleStates, got {args}')
        return super().__new__(cls, *args)

    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % arg._sympystr(printer, *args)) for arg in self.args)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            next_pform = prettyForm(*next_pform.parens(left='{', right='}'))
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        return r'\otimes'.join((r'\left\{ %s \right\}' % arg._latex(printer, *args))
                               for arg in self.args)


class FieldOperator(TensorProduct):
    """TensorProduct of particle-level operators."""
    def __new__(cls, *args):
        if not all(isinstance(arg, (Operator, TensorProduct)) for arg in args):
            raise ValueError(f'FieldOperator must be a product of Operators, got {args}')
        return super().__new__(cls, *args)

    def _eval_adjoint(self):
        return FieldOperator(*[Dagger(arg) for arg in self.args])

    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % arg._sympystr(printer, *args)) for arg in self.args)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            next_pform = prettyForm(*next_pform.parens(left='{', right='}'))
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        return r'\otimes'.join((r'\left\{ %s \right\}' % arg._latex(printer, *args))
                               for arg in self.args)


class ParticleState(TensorProduct, KetBase):
    """TensorProduct of a presence ket and a quantum number product ket."""
    def __new__(cls, *args):
        args = sympify(args)
        if len(args) == 1 and isinstance(args[0], TensorProduct) and len(args[0].args) == 2:
            args = (args[0].args[0].args[0], args[0].args[1].args)
        if not (len(args) == 2 and args[0] in (0, 1) and isinstance(args[1], (Sequence, Tuple))):
            raise ValueError(f'Invalid constructor argument for ParticleState: {args}')

        return super().__new__(cls, OrthogonalKet(args[0]), OrthogonalProductKet(*args[1]))

    def _print_contents(self, printer, *args):
        if self.args[0].args[0] == 0:
            return '_O_'
        return self.args[1]._print_contents(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        if self.args[0].args[0] == 0:
            if printer._use_unicode:
                return prettyForm('\N{GREEK CAPITAL LETTER OMEGA}')
            else:
                return prettyForm('_O_')
        return self.args[1]._print_contents_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        if self.args[0].args[0] == 0:
            return r'\Omega'
        return self.args[1]._print_contents_latex(printer, *args)

    def _sympystr(self, printer, *args):
        return KetBase._sympystr(self, printer, *args)

    def _pretty(self, printer, *args):
        return KetBase._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return KetBase._latex(self, printer, *args)


class Control(OuterProduct):
    """Control operator for particle registers."""
    def __new__(cls, *args):
        if not (len(args) == 2 and all(arg in (0, 1) for arg in args)):
            raise ValueError(f'Invalid constructor argument {args} for control')

        return super().__new__(cls, OrthogonalKet(args[0]), OrthogonalBra(args[1]))

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
        return (f'{self._print_operator_name(printer, *args)}({self.args[0]},{self.args[1]})')

    def _print_contents_latex(self, printer, *args):
        return r'%s\left({%s}, {%s}\right)' % (
            (self._print_operator_name_latex(printer, *args),) + self.args
        )

    @staticmethod
    def swap_particles(
        state: FieldState,
        index1: int,
        index2: int
    ) -> FieldState:
        particle_states = list(state.args)
        particle_states[index1] = state.args[index2]
        particle_states[index2] = state.args[index1]
        return FieldState(*particle_states)

    def _apply_operator(self, state: FieldState, **options) -> ProductKet:
        return self.swap_particles(state, *self.args)  # pylint: disable=no-value-for-parameter

    def _eval_power(self, exp):
        return IdentityOperator() if exp % 2 == 0 else self

    def _eval_inverse(self):
        return self


class StepSymmetrizer(Operator):
    """Step-symmetrizer of a bosonic field register."""
    is_hermitian = True

    def __new__(cls, *args, **kwargs):
        args = sympify(args)
        if not (len(args) == 1 and args[0].is_integer and args[0] > 0):
            raise ValueError('StepSymmetrizer requires one integer argument (updated number of'
                             ' particles)')
        if args[0] == 1:
            return IdentityOperator()

        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'S'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('S')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathcal{S}'

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
                ParticleSwap.swap_particles(state, new_num - 1, ipart)
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
            ops += [ParticleSwap(new_num - 1, ipart) for ipart in range(new_num - 1)]
            return Add(*ops) / sqrt(new_num)
        return None
