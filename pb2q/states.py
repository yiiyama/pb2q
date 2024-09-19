# pylint: disable=no-member, unused-argument, invalid-name, too-few-public-methods
"""State representations as sympy objects."""

from collections.abc import Sequence
from numbers import Integral
from sympy import Add, Expr, Mul, S, sympify
from sympy.core.containers import Tuple
from sympy.physics.quantum import (BraBase, KetBase, OrthogonalBra, OrthogonalKet, StateBase,
                                   TensorProduct)
from sympy.physics.quantum.qexpr import QExpr
from sympy.printing.pretty.stringpict import prettyForm

from .sympy.product_state import ProductState, ProductKet, ProductBra


class UniverseState(ProductState):
    """TensorProduct of FieldStates."""
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
        return r'\otimes'.join(fr'\llbracket {arg._latex(printer, *args)} \rrbracket'
                               for arg in self.args)


class UniverseKet(UniverseState, ProductKet):
    """TensorProduct of FieldKets."""
    @classmethod
    def dual_class(cls):
        """Used for type tests (e.g. in OuterProduct)."""
        return UniverseBra

    @classmethod
    def component_class(cls) -> type[KetBase]:
        return FieldKet


class UniverseBra(UniverseState, ProductBra):
    """TensorProduct of FieldBras."""
    @classmethod
    def dual_class(cls):
        return UniverseKet

    @classmethod
    def component_class(cls):
        return FieldBra


class FieldState(ProductState):
    """TensorProduct of ParticleStates."""
    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % arg._sympystr(printer, *args)) for arg in self.args)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            # next_pform = prettyForm(*next_pform.parens(left='{', right='}'))
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        # return r'\otimes'.join((r'\left\{ %s \right\}' % arg._latex(printer, *args))
        #                        for arg in self.args)
        return r'\otimes'.join(arg._latex(printer, *args) for arg in self.args)


class FieldKet(FieldState, ProductKet):
    """TensorProduct of ParticleKets."""
    @classmethod
    def dual_class(cls):
        return FieldBra

    @classmethod
    def component_class(cls):
        return ParticleKet


class FieldBra(FieldState, ProductBra):
    """TensorProduct of ParticleBras."""
    @classmethod
    def dual_class(cls):
        return FieldKet

    @classmethod
    def component_class(cls):
        return ParticleBra


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


class ParticleState(StateBase, TensorProduct):
    """TensorProduct of a momentum state and a quantum number product state."""
    def __new__(cls, *args):
        if len(args) == 1 and args[0] is None:
            return cls.null_state_class()()

        args = sympify(args)
        if len(args) == 1 and isinstance(args[0], TensorProduct):
            # Type-casting form (Single TensorProduct argument)
            args = args[0].args

        if len(args) == 2:
            pcls = cls.momentum_state_class()
            qcls = cls.qnumber_state_class()
            if isinstance(args[0], (Integral, Sequence, Tuple)):
                args = (pcls(*args[0]), args[1])
            if isinstance(args[1], (Sequence, Tuple)):
                args = (args[0], qcls(*args[1]))

            if (isinstance(args[0], pcls) and isinstance(args[1], qcls)):
                return QExpr.__new__(cls, *args)

        raise ValueError(f'Invalid constructor arguments for ParticleState {args}')

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
    def dual_class(cls):
        return ParticleBra

    @classmethod
    def null_state_class(cls):
        return NullKet

    @classmethod
    def momentum_state_class(cls):
        return MomentumKet

    @classmethod
    def qnumber_state_class(cls):
        return QNumberKet

    def _eval_innerproduct(self, bra, **hints):
        # TODO: can use Hilbert space check if that's implemented
        if isinstance(bra, NullBra):
            return S.Zero

        if isinstance(bra, ParticleBra):
            for bra_arg, arg in zip(bra.args, self.args):
                compres = arg._eval_innerproduct(bra_arg, **hints)
                if compres is None:
                    return None
                if compres == 0:
                    return S.Zero

        return S.One


class ParticleBra(ParticleState, BraBase):
    """ParticleState bra."""
    @classmethod
    def dual_class(cls):
        return ParticleKet

    @classmethod
    def momentum_state_class(cls):
        return MomentumBra

    @classmethod
    def qnumber_state_class(cls):
        return QNumberBra


class NullState(StateBase):
    """Representation of unoccupied state."""
    def __new__(cls):
        return Expr.__new__(cls)

    def _print_contents(self, printer, *args):
        return '_O_'

    def _print_contents_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{GREEK CAPITAL LETTER OMEGA}')
        return prettyForm('_O_')

    def _print_contents_latex(self, printer, *args):
        return r'\Omega'


class NullKet(NullState, ParticleKet):
    """Ket representing the unoccupied state."""
    @classmethod
    def dual_class(cls):
        return NullBra

    def _eval_innerproduct(self, bra, **hints):
        if isinstance(bra, NullBra):
            return S.One
        if isinstance(bra, ParticleBra):
            return S.Zero


class NullBra(NullState, OrthogonalBra):
    """Bra representing the unoccupied state."""
    @classmethod
    def dual_class(cls):
        return NullKet


class MomentumState(ProductState):
    """Generic momentum state."""
    _label_separator = ','

    def __new__(cls, *args):
        args = sympify(args)
        if len(args) == 1 and isinstance(args[0], Integral):
            args = (args,)

        return super().__new__(cls, *args)


class MomentumKet(MomentumState, ProductKet):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumBra

    @classmethod
    def component_class(cls):
        return OrthogonalKet


class MomentumBra(MomentumState, ProductBra):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumKet

    @classmethod
    def component_class(cls):
        return OrthogonalBra


class QNumberState(ProductState):
    """Generic quantum number state."""
    _label_separator = ';'

    def __new__(cls, *args):
        args = sympify(args)
        if len(args) == 1 and isinstance(args[0], Integral):
            args = (args,)

        return super().__new__(cls, *args)


class QNumberKet(QNumberState, ProductKet):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberBra

    @classmethod
    def component_class(cls):
        return OrthogonalKet


class QNumberBra(QNumberState, ProductBra):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberKet

    @classmethod
    def component_class(cls):
        return OrthogonalBra
