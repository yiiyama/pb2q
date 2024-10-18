# pylint: disable=no-member, unused-argument, invalid-name, too-few-public-methods
"""State representations as sympy objects."""

from collections import Counter
from collections.abc import Sequence
from numbers import Integral
from sympy import Add, Function, Mul, S, factorial, sqrt, sympify
from sympy.core.containers import Tuple
from sympy.physics.quantum import KetBase, OrthogonalBra, OrthogonalKet, TensorProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.printing.pretty.stringpict import prettyForm

from .sympy.product_state import ProductState, ProductKet, ProductBra
from .utils import generate_perm


class UniverseState(ProductState):
    """TensorProduct of FieldStates."""
    def _sympystr(self, printer, *args):
        return 'x'.join(('{%s}' % printer._print(arg, *args)) for arg in reversed(self.args))

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
            pform = prettyForm(*pform.left(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.left('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.left('x' + ' '))

        return pform

    def _latex(self, printer, *args):
        return r'\otimes'.join(fr'\llbracket {printer._print(arg, *args)} \rrbracket'
                               for arg in reversed(self.args))

    @property
    def fields(self):
        return self.args


class UniverseKet(UniverseState, ProductKet):
    """TensorProduct of FieldKets."""
    @classmethod
    def dual_class(cls):
        """Used for type tests (e.g. in OuterProduct)."""
        return UniverseBra

    @classmethod
    def component_class(cls) -> type[KetBase]:
        return FieldKet


class UniverseBra(UniverseState, ProductBra):
    """TensorProduct of FieldBras."""
    @classmethod
    def dual_class(cls):
        return UniverseKet

    @classmethod
    def component_class(cls):
        return FieldBra


class FieldState(ProductState):
    """TensorProduct of ParticleStates."""
    __slots__ = ('_n',)

    def __new__(cls, *args):
        obj = super().__new__(cls, *args)
        obj._nocc = sum(1 if not arg.is_null_state else 0 for arg in args)
        return obj

    @property
    def nocc(self):
        return self._nocc

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

    @property
    def particles(self):
        return self.args


class FieldKet(FieldState, ProductKet):
    """TensorProduct of ParticleKets."""
    @classmethod
    def dual_class(cls):
        return FieldBra

    @classmethod
    def component_class(cls):
        return ParticleKet


class FieldBra(FieldState, ProductBra):
    """TensorProduct of ParticleBras."""
    @classmethod
    def dual_class(cls):
        return FieldKet

    @classmethod
    def component_class(cls):
        return ParticleBra


class SymmetricFieldStateBase(FieldState):
    """FieldState with implicit symmetrization."""
    __slots__ = ('right_filled',)

    def __new__(cls, *args, right_filled=True):
        obj = super().__new__(cls, *args)
        obj.right_filled = right_filled
        if right_filled and not all(arg.is_null_state for arg in args[obj.nocc:]):
            raise ValueError('Field state arguments are not right-filled')
        return obj

    def doit(self, **hints):
        # pylint: disable-next=import-outside-toplevel
        from .operators.symm import ParticlePermutation

        npart = self.nocc if self.right_filled else len(self.args)
        permutations = [perm + tuple(range(npart, len(self.args)))
                        for perm in generate_perm(range(npart))]

        norm = sqrt(factorial(npart))
        for cnt in Counter(self.args[:npart]).values():
            if cnt != 1:
                norm /= sqrt(factorial(cnt))

        terms = []
        for ip, perm in enumerate(permutations):
            if ip == 0:
                terms.append(self.field_state(*self.args))
            else:
                terms.append(
                    (self._sign ** ip) * ParticlePermutation.order_particles(terms[0], perm)
                )

        return (Add(*terms) / norm).expand()

    def _sympystr(self, printer, *args):
        return (f'{self.lbracket}{super()._sympystr(printer, *args)}{self.rbracket}'
                + ('S' if self._sign == 1 else 'A'))

    def _pretty(self, printer, *args):
        pform = super()._pretty(printer, *args)
        lbracket, rbracket = self._pretty_brackets(pform.height(), printer._use_unicode)
        pform = prettyForm(*pform.parens(left=lbracket, right=rbracket))
        return prettyForm(*pform.right('S' if self._sign == 1 else 'A'))

    def _latex(self, printer, *args):
        return '{{%s%s%s}_%s}' % (
            self.lbracket_latex,
            super()._latex(printer, *args),
            self.rbracket_latex,
            'S' if self._sign == 1 else 'A'
        )


class SymmetricFieldState(SymmetricFieldStateBase):
    """Symmetric FieldState."""
    _sign = 1


class SymmetricFieldKet(SymmetricFieldState, ProductKet):
    """Symmetric TensorProduct of ParticleKets."""
    field_state = FieldKet

    @classmethod
    def dual_class(cls):
        return SymmetricFieldBra

    @classmethod
    def component_class(cls):
        return ParticleKet


class SymmetricFieldBra(SymmetricFieldState, ProductBra):
    """Symmetric TensorProduct of ParticleBras."""
    field_state = FieldBra

    @classmethod
    def dual_class(cls):
        return SymmetricFieldKet

    @classmethod
    def component_class(cls):
        return ParticleBra


class AntisymmetricFieldState(SymmetricFieldStateBase):
    """Antisymmetric FieldState."""
    _sign = -1


class AntisymmetricFieldKet(AntisymmetricFieldState, ProductKet):
    """Antisymmetric TensorProduct of ParticleKets."""
    field_state = FieldKet

    @classmethod
    def dual_class(cls):
        return AntisymmetricFieldBra

    @classmethod
    def component_class(cls):
        return ParticleKet


class AntisymmetricFieldBra(AntisymmetricFieldState, ProductBra):
    """Antisymmetric TensorProduct of ParticleBras."""
    field_state = FieldBra

    @classmethod
    def dual_class(cls):
        return AntisymmetricFieldKet

    @classmethod
    def component_class(cls):
        return ParticleBra


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


class ParticleState(ProductState):
    """TensorProduct of a momentum state and a quantum number product state."""
    def __new__(cls, *args):
        if not args or (len(args) == 1 and args[0] == 0):
            return QExpr.__new__(cls, S.Zero)

        args = sympify(args)
        if len(args) == 2:
            pcls = cls.momentum_state_class()
            qcls = cls.qnumber_state_class()
            if isinstance(args[0], Integral):
                args = (pcls(args[0]), args[1])
            elif isinstance(args[0], (Sequence, Tuple)):
                args = (pcls(*args[0]), args[1])
            if isinstance(args[1], Integral):
                args = (args[0], qcls(args[1]))
            elif isinstance(args[1], (Sequence, Tuple)):
                args = (args[0], qcls(*args[1]))

            if (isinstance(args[0], pcls) and isinstance(args[1], qcls)):
                return QExpr.__new__(cls, *args)

        raise ValueError(f'Invalid constructor arguments for ParticleState {args}')

    @classmethod
    def default_args(cls):
        return None

    def _print_contents(self, printer, *args):
        if self.is_null_state:
            return '_O_'
        return ':'.join(arg._print_contents(printer, *args) for arg in self.args)

    def _print_contents_pretty(self, printer, *args):
        if self.is_null_state:
            if printer._use_unicode:
                return prettyForm('\N{GREEK CAPITAL LETTER OMEGA}')
            return prettyForm('_O_')

        pform = self.args[0]._print_contents_pretty(printer, *args)
        pform = prettyForm(*pform.right(':'))
        pform = prettyForm(*pform.right(self.args[1]._print_contents_pretty(printer, *args)))
        return pform

    def _print_contents_latex(self, printer, *args):
        if self.is_null_state:
            return r'\Omega'
        return ': '.join(arg._print_contents_latex(printer, *args) for arg in self.args)

    @property
    def momentum(self):
        if self.is_null_state:
            return None
        return self.args[0]

    @property
    def qnumber(self):
        if self.is_null_state:
            return None
        return self.args[1]

    @property
    def is_null_state(self):
        return self.args[0] == 0


class ParticleKet(ParticleState, ProductKet):
    """ParticleState ket."""
    @classmethod
    def dual_class(cls):
        return ParticleBra

    @classmethod
    def momentum_state_class(cls):
        return MomentumKet

    @classmethod
    def qnumber_state_class(cls):
        return QNumberKet

    def _eval_innerproduct(self, bra, **hints):
        if isinstance(bra, ParticleBra):
            if self.is_null_state:
                if bra.is_null_state:
                    return S.One
                return S.Zero

            if bra.is_null_state:
                return S.Zero

        return super()._eval_innerproduct(bra, **hints)

    def __mul__(self, other):
        # pylint: disable-next=import-outside-toplevel
        from pb2q.operators import ParticleOuterProduct
        if isinstance(other, ParticleBra):
            return ParticleOuterProduct(self, other)
        return KetBase.__mul__(self, other)


class ParticleBra(ParticleState, ProductBra):
    """ParticleState bra."""
    @classmethod
    def dual_class(cls):
        return ParticleKet

    @classmethod
    def momentum_state_class(cls):
        return MomentumBra

    @classmethod
    def qnumber_state_class(cls):
        return QNumberBra


class QNumberState(ProductState):
    """Generic quantum number state."""
    _label_separator = ';'

    def __new__(cls, *args):
        comp_cls = cls.component_class()
        statified = []
        for arg in args:
            # pylint: disable-next=isinstance-second-argument-not-valid-type
            if isinstance(arg, comp_cls):
                statified.append(arg)
            else:
                statified.append(comp_cls(arg))  # pylint: disable=not-callable

        return super().__new__(cls, *statified)

    def _print_contents(self, printer, *args):
        return self._label_separator.join(arg._print_contents(printer, *args) for arg in self.args)

    def _print_contents_pretty(self, printer, *args):
        pform = self.args[0]._print_contents_pretty(printer, *args)
        for arg in self.args[1:]:
            pform = prettyForm(*pform.right(self._label_separator))
            pform = prettyForm(*pform.right(arg._print_contents_pretty(printer, *args)))
        return pform

    def _print_contents_latex(self, printer, *args):
        return self._label_separator.join(
            arg._print_contents_latex(printer, *args) for arg in self.args
        )

    @property
    def component(self):
        return self.args


class QNumberKet(QNumberState, ProductKet):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberBra

    @classmethod
    def component_class(cls):
        return OrthogonalKet


class QNumberBra(QNumberState, ProductBra):
    """Quantum number ket."""
    @classmethod
    def dual_class(cls):
        return QNumberKet

    @classmethod
    def component_class(cls):
        return OrthogonalBra


class MomentumState(QNumberState):
    """Generic momentum state."""
    _label_separator = ','

    def __getitem__(self, index):
        return self.args[index].args[0]

    @property
    def num_dimensions(self):
        return len(self.args)

    @property
    def x(self):
        return self.args[0]

    @property
    def y(self):
        return self.args[1]

    @property
    def z(self):
        return self.args[2]

    @property
    def energy(self):
        # pylint: disable-next=not-callable
        return Function('E')(sympify(tuple(p.args[0] for p in self.args)))


class MomentumKet(MomentumState, ProductKet):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumBra

    @classmethod
    def component_class(cls):
        return OrthogonalKet


class MomentumBra(MomentumState, ProductBra):
    """Momentum ket."""
    @classmethod
    def dual_class(cls):
        return MomentumKet

    @classmethod
    def component_class(cls):
        return OrthogonalBra
