"""Universe-level operator representations as sympy objects."""

from sympy import Add, Mul
from sympy.physics.quantum import Dagger, Operator, TensorProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.printing.pretty.stringpict import prettyForm


class UniverseOperator(Operator, TensorProduct):
    """TensorProduct of particle-level operators."""
    def __new__(cls, *args):
        if not cls._check_field_ops(args):
            raise ValueError(f'UniverseOperator must be a product of FieldOperators, got {args}')
        return QExpr.__new__(cls, *args)

    @staticmethod
    def _check_field_ops(args):
        for arg in args:
            if isinstance(arg, Add):
                print(arg, 'is Add')
                for term in arg.args:
                    if isinstance(term, Mul):
                        print(term, 'is Mul')
                        _, nc = term.args_cnc()
                        if not all(isinstance(op, Operator) for op in nc):
                            print('nc', nc, [type(op) for op in nc])
                            return False
                    elif not isinstance(term, Operator):
                        print(term, 'is not instance')
                        return False

            elif not isinstance(arg, Operator):
                print(arg, 'is not instance')
                return False

        return True

    def _eval_adjoint(self):
        return UniverseOperator(*[Dagger(arg) for arg in self.args])

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
