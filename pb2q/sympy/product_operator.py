"""Operators that are also TensorProducts of component system operators."""
from sympy.physics.quantum import Operator
from .product_qexpr import ProductQExpr


class ProductOperator(Operator, ProductQExpr):
    """General product operator."""
