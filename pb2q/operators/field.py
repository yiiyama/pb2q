# pylint: disable=import-outside-toplevel
"""Field-level operator representations as sympy objects."""

from sympy import Add, Mul, S, Expr, sqrt
from sympy.physics.quantum import TensorProduct
from sympy.printing.pretty.stringpict import prettyForm
from ..sympy import ProductOperator
from .particle import PresenceProjection, ParticleOuterProduct


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


def project_physical(expr: Expr, **options) -> Expr:
    """Project a FieldOperator to the physical (right-filled) subspace."""
    from .symm import StepSymmetrizerBase

    symmetry = options.get('symmetry')
    expr = expr.expand(tensorproduct=True, commutator=True)

    if isinstance(expr, Add):
        terms = []
        for arg in expr.args:
            term = project_physical(arg, **options)
            if term != 0:
                terms.append(term)

        return Add(*terms).expand()

    if isinstance(expr, Mul):
        c_part, nc_part = expr.args_cnc()
        ops = []
        for op in nc_part:
            op = project_physical(op, **options)
            if op == 0:
                return S.Zero
            ops.append(op)
        return Mul(*(c_part + ops))

    if isinstance(expr, StepSymmetrizerBase):
        if expr._sign == symmetry:
            return sqrt(expr.args[0])
        if symmetry is not None:
            return S.Zero

    if not isinstance(expr, TensorProduct):
        return expr

    if len(expr.args) == 0:
        return S.Zero

    def _right_present(part_op):
        return (isinstance(part_op, PresenceProjection)
                or (isinstance(part_op, ParticleOuterProduct) and not part_op.bra.is_null_state))

    def _left_present(part_op):
        return (isinstance(part_op, PresenceProjection)
                or (isinstance(part_op, ParticleOuterProduct) and not part_op.ket.is_null_state))

    right_present = _right_present(expr.args[0])
    left_present = _left_present(expr.args[0])
    for part_op in expr.args[1:]:
        if _right_present(part_op):
            if not right_present:
                return S.Zero
        else:
            right_present = False

        if _left_present(part_op):
            if not left_present:
                return S.Zero
        else:
            left_present = False

    return expr
