"""Representation of momentum."""

from sympy import Add, Expr, sqrt
from sympy.printing.pretty.stringpict import prettyForm
from .sympy import OrthogonalProductBra, OrthogonalProductKet


class MomentumKet(OrthogonalProductKet):
    """Ket class for printing the momentum as a tuple."""
    _label_separator = ','

    def _print_contents(self, printer, *args):
        return f'[{self._print_label(printer, *args)}]'
    
    def _print_contents_pretty(self, printer, *args):
        pform = super()._print_contents_pretty(printer, *args)
        pform = prettyForm(*pform.left('['))
        pform = prettyForm(*pform.right(']'))
        return pform
    
    def _print_contents_latex(self, printer, *args):
        return fr'\left[ {super()._print_contents_latex(printer, *args)} \right]'


class Momentum:
    """Converter between momentum numerical values and integral representations."""

    def __init__(self, rep: tuple[int, ...], mass: float = 0.):
        self.rep = rep
        self.mass = mass

    def numeric(self) -> Expr:
        return (0.,) * len(self.rep)

    def energy(self) -> Expr:
        return sqrt(self.mass ** 2 + Add(*[p ** 2 for p in self.numeric()]))
