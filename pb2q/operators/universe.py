"""Universe-level operator representations as sympy objects."""

from sympy.printing.pretty.stringpict import prettyForm
from ..sympy import ProductOperator


class UniverseOperator(ProductOperator):
    """Universe-level operator."""

    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % printer._print(arg, *args)) for arg in self.args)

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
        return r'\otimes'.join(fr'\llbracket {printer._print(arg, *args)} \rrbracket'
                               for arg in self.args)
