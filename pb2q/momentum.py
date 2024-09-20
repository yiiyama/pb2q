"""Representation of momentum."""

from sympy import Add, Expr, sqrt


class Momentum:
    """Converter between momentum numerical values and integral representations."""

    def __init__(self, rep: tuple[int, ...], mass: float = 0.):
        self.rep = rep
        self.mass = mass

    def numeric(self) -> Expr:
        return (0.,) * len(self.rep)

    def energy(self) -> Expr:
        return sqrt(self.mass ** 2 + Add(*[p ** 2 for p in self.numeric()]))
