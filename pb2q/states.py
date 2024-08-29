"""State representations as sympy objects."""

from collections.abc import Sequence
from sympy import Add, Mul, sympify
from sympy.core.containers import Tuple
from sympy.physics.quantum import KetBase, OrthogonalKet, TensorProduct
from sympy.printing.pretty.stringpict import prettyForm

from .sympy import OrthogonalProductKet


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
        if len(args) == 1 and isinstance(args[0], TensorProduct):
            args = args[0].args
        if all(isinstance(arg, TensorProduct) for arg in args):
            return super().__new__(cls, *[ParticleState(arg) for arg in args])

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
