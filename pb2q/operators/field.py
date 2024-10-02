"""Field-level operator representations as sympy objects."""
import logging
from sympy.printing.pretty.stringpict import prettyForm
from ..sympy import ProductOperator

LOG = logging.getLogger(__name__)


class FieldOperator(ProductOperator):
    """Field-level operator."""
    def _sympystr(self, printer, *args):
        return 'x'.join(printer._print(arg, *args) for arg in reversed(self.args))

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            # next_pform = prettyForm(*next_pform.parens(left='{', right='}'))
            pform = prettyForm(*pform.left(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.left('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.left('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        # return r'\otimes'.join((r'\left\{ %s \right\}' % arg._latex(printer, *args))
        #                        for arg in self.args)
        return r'\otimes'.join(printer._print(arg, *args) for arg in reversed(self.args))
