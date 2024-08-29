"""Field-level operator representations as sympy objects."""

from sympy import Add, Mul, S
from sympy.physics.quantum import Dagger, Operator, TensorProduct, qapply, tensor_product_simp
from sympy.printing.pretty.stringpict import prettyForm


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
