# pylint: disable=no-member, unused-argument, invalid-name, too-few-public-methods
"""State representations as sympy objects."""

from collections.abc import Sequence
from numbers import Integral
from sympy import Add, Expr, Mul, S, sympify
from sympy.core.containers import Tuple
from sympy.physics.quantum import BraBase, KetBase, OrthogonalBra, StateBase, TensorProduct
from sympy.printing.pretty.stringpict import prettyForm

from .sympy import OrthogonalProductBra, OrthogonalProductKet


class UniverseState(TensorProduct):
    """TensorProduct of FieldStates."""
    def __new__(cls, *args):
        # Comparing lbrackets instead of checking class inheritance!!
        if all(isinstance(arg, cls.field_state_class()) for arg in args):
            # Using Expr.__new__ instead of super().__new__ because TensorProduct.__new__ returns
            # arg[0] if len(arg) == 1
            return Expr.__new__(cls, *args)
        raise ValueError(f'UniverseState must be a product of FieldStates, got {args}')

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
                    right='\N{MATHEMATICAL RIGHT WHITE SQUARE BRACKET}')
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


class UniverseKet(UniverseState, KetBase):
    """TensorProduct of FieldKets."""
    @classmethod
    def field_state_class(cls):
        return FieldKet

    @classmethod
    def dual_class(cls):
        return UniverseBra

    def _eval_innerproduct_UniverseBra(self, bra, **hints):
        if len(self.args) != len(bra.args):
            raise ValueError('Cannot take an inner product of states from different universes')

        for others, mine in zip(bra.args, self.args):
            if mine._eval_innerproduct(others) == 0:
                return S.Zero

        return S.One


class UniverseBra(UniverseState, BraBase):
    """TensorProduct of FieldBras."""
    @classmethod
    def field_state_class(cls):
        return FieldBra

    @classmethod
    def dual_class(cls):
        return UniverseKet


class FieldState(TensorProduct):
    """TensorProduct of ParticleStates."""
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], TensorProduct):
            # Type-casting form (Single TensorProduct argument)
            args = args[0].args

        null_state_class = cls.null_state_class()
        particle_state_class = cls.particle_state_class()
        if all(isinstance(arg, (null_state_class, particle_state_class)) for arg in args):
            return Expr.__new__(cls, *args)
        raise ValueError(f'FieldState must be a product of ParticleStates, got {args}')

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


class FieldKet(FieldState, KetBase):
    """TensorProduct of ParticleKets."""
    @classmethod
    def null_state_class(cls):
        return NullKet

    @classmethod
    def particle_state_class(cls):
        return ParticleKet

    @classmethod
    def dual_class(cls):
        return FieldBra

    def _eval_innerproduct_FieldBra(self, bra, **hints):
        if len(self.args) != len(bra.args):
            raise ValueError('Cannot evaluate an inner product of states of different fields.')

        for others, mine in zip(bra.args, self.args):
            if mine._eval_innerproduct(others) == 0:
                return S.Zero

        return S.One


class FieldBra(FieldState, BraBase):
    """TensorProduct of ParticleBras."""
    @classmethod
    def null_state_class(cls):
        return NullBra

    @classmethod
    def particle_state_class(cls):
        return ParticleBra

    @classmethod
    def dual_class(cls):
        return FieldKet


def as_field_state(expr: Add):
    """Convert a linear combination of TensorProducts to a linear combination of FieldStates."""
    if isinstance(expr, Add):
        terms = expr.args
    else:
        terms = [expr]

    output_args = []
    for term in terms:
        # term is coefficient * tp
        c_part, new_args = TensorProduct.flatten(sympify((term,)))
        output_args.append(Mul(*c_part) * FieldState(new_args[0]))

    return Add(*output_args)


class NullState(StateBase):
    """Representation of unoccupied state."""
    def _print_contents(self, printer, *args):
        return '_O_'

    def _print_contents_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{GREEK CAPITAL LETTER OMEGA}')
        return prettyForm('_O_')

    def _print_contents_latex(self, printer, *args):
        return r'\Omega'


class NullKet(NullState, KetBase):
    """Ket representing the unoccupied state."""
    def __new__(cls):
        return super().__new__(cls, 0)

    @classmethod
    def dual_class(cls):
        return NullBra

    def _eval_innerproduct_NullBra(self, bra, **hints):
        return S.One

    def _eval_innerproduct_ParticleBra(self, bra, **hints):
        return S.Zero


class NullBra(NullState, OrthogonalBra):
    """Bra representing the unoccupied state."""
    def __new__(cls):
        return super().__new__(cls, 0)

    @classmethod
    def dual_class(cls):
        return NullKet


class ParticleState(TensorProduct):
    """TensorProduct of a momentum state and a quantum number product state."""
    def __new__(cls, *args):
        args = sympify(args)
        if len(args) == 1 and isinstance(args[0], TensorProduct):
            # Type-casting form (Single TensorProduct argument)
            momentum, qnumber = args[0].args
            args = (momentum.args, qnumber.args)
        if not (len(args) == 2
                and isinstance(args[0], (Integral, Sequence, Tuple))
                and isinstance(args[1], (Sequence, Tuple))):
            raise ValueError(f'Invalid constructor argument for ParticleState: {args}')

        return super().__new__(
            cls,
            cls.momentum_state_class()(*args[0]),
            cls.qnumber_state_class()(*args[1])
        )

    def _print_contents(self, printer, *args):
        return ';'.join(arg._print_contents(printer, *args) for arg in self.args)

    def _print_contents_pretty(self, printer, *args):
        pform = self.args[0]._print_contents_pretty(printer, *args)
        pform = prettyForm(*pform.right(','))
        pform = prettyForm(*pform.right(self.args[1]._print_contents_pretty(printer, *args)))
        return pform

    def _print_contents_latex(self, printer, *args):
        return '; '.join(arg._print_contents_latex(printer, *args) for arg in self.args)

    def _sympystr(self, printer, *args):
        return super(TensorProduct, self)._sympystr(printer, *args)  # pylint: disable

    def _pretty(self, printer, *args):
        return super(TensorProduct, self)._pretty(printer, *args)

    def _latex(self, printer, *args):
        return super(TensorProduct, self)._latex(printer, *args)


class ParticleKet(ParticleState, KetBase):
    """ParticleState ket."""
    @classmethod
    def momentum_state_class(cls):
        return MomentumKet

    @classmethod
    def qnumber_state_class(cls):
        return QNumberKet

    @classmethod
    def dual_class(cls):
        return ParticleBra

    def _eval_innerproduct_ParticleBra(self, bra, **hints):
        for others, mine in zip(bra.args, self.args):
            if mine._eval_innerproduct(others) == 0:
                return S.Zero

        return S.One

    def _eval_innerproduct_NullBra(self, bra, **hints):
        return S.Zero


class ParticleBra(ParticleState, BraBase):
    """ParticleState ket."""
    @classmethod
    def momentum_state_class(cls):
        return MomentumBra

    @classmethod
    def qnumber_state_class(cls):
        return QNumberBra

    @classmethod
    def dual_class(cls):
        return ParticleKet


class MomentumPrinterMixin:
    """Mixin class for printing momentum values."""
    _label_separator = ','


class MomentumKet(MomentumPrinterMixin, OrthogonalProductKet):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumBra


class MomentumBra(MomentumPrinterMixin, OrthogonalProductBra):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumKet


class QNumberPrinterMixin:
    """Mixin class for printing momentum values."""
    _label_separator = ';'


class QNumberKet(QNumberPrinterMixin, OrthogonalProductKet):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberBra


class QNumberBra(QNumberPrinterMixin, OrthogonalProductBra):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberKet
