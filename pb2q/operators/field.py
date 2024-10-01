"""Field-level operator representations as sympy objects."""

from sympy import Add, Mul, S
from sympy.physics.quantum import TensorProduct, qapply, tensor_product_simp
from sympy.printing.pretty.stringpict import prettyForm
from ..sympy import ProductOperator


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


def apply_field_op(expr: Mul):
    """Expand the product and apply tensor_product_simp + qapply on each term, then reinterpret as
    field state / op."""
    if not isinstance(expr, Mul):
        return expr

    output_args = []
    for term in expr.expand().args:
        if not isinstance(term, Mul):
            output_args.append(term)
            continue

        output_term = S.One
        tps = []
        for factor in term.args:
            if isinstance(factor, TensorProduct):
                tps.append(factor)
            elif len(tps) != 0:
                output_term *= tensor_product_simp(Mul(*tps))
                tps = []
            else:
                output_term *= factor
        if len(tps) != 0:
            output_term *= tensor_product_simp(Mul(*tps))

        print(output_term)
        output_args.append(qapply(output_term))

    return Add(*output_args)
