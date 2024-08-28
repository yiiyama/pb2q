"""Registers."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from sympy import Add, Expr, S, sqrt
from sympy.physics.quantum import (Dagger, IdentityOperator, Ket, KetBase, Operator, OrthogonalBra,
                                   OrthogonalKet, OuterProduct, TensorProduct)
from sympy.printing.pretty.stringpict import prettyForm
from .field import FieldDefinition
from .sympy import (IdentityProduct, OrthogonalProductBra, OrthogonalProductKet,
                    PermutationOperator, ProductKet, to_product_state, to_tensor_product)
from .representations import UniverseState, FieldState, ParticleState


class RegisterBase(ABC):
    """Base register class."""
    @classmethod
    @abstractmethod
    def size(cls) -> int:
        """Return the total number of physical registers."""

    @classmethod
    @abstractmethod
    def initial_state(cls) -> Expr:
        """Return the zero state."""

    def __init__(self):
        self._name = type(self).__name__

    @property
    def name(self):
        return self._name


class Register(RegisterBase):
    """Fixed-dimension register class."""
    dimension: int = 0

    @classmethod
    def size(cls) -> int:
        return 1

    @classmethod
    def initial_state(cls) -> Expr:
        return OrthogonalProductKet(0)


class CompoundRegister(RegisterBase):
    """A register that consists of other registers."""
    @classmethod
    def initial_state(cls) -> Expr:
        return OrthogonalProductKet(*((0,) * cls.size()))

    def interpret(self, state: Expr) -> str:  # pylint: disable=unused-argument
        return ''


class Universe(CompoundRegister):
    """Total physical register."""
    _singleton: 'Universe' = None

    @classmethod
    def size(cls) -> int:
        return sum(field.size() for field in cls._singleton.fields.values())

    @classmethod
    def initial_state(cls) -> Expr:
        return UniverseState(*[field.initial_state() for field in cls._singleton.fields.values()])

    def __init__(
        self,
        fields: list[FieldDefinition],
        spatial_dimension: int
    ):
        super().__init__()
        self.spatial_dimension = spatial_dimension
        self.fields = {definition.name: self.make_field(definition) for definition in fields}
        Universe._singleton = self

    def make_field(self, definition: FieldDefinition) -> 'Field':
        momentum = Momentum.of_dimension(self.spatial_dimension)
        spin = Spin.of_spin(definition.spin)
        quantum_numbers = [type(name, (Register,), {'dimension': dimension})
                           for name, dimension in definition.quantum_numbers]
        attributes = {'momentum': momentum}
        if spin is not None:
            attributes['spin'] = spin
        attributes.update({
            'quantum_numbers': quantum_numbers,
            'max_particles': definition.max_particles
        })
        field_cls = type(definition.name, (Field,), attributes)

        particle_cls = type(definition.name + 'Particle', (Particle,), {'field': field_cls})
        field_cls.particle = particle_cls

        return field_cls()


class Field(CompoundRegister):
    """Register for a single field species."""
    momentum: type['Momentum'] = None
    spin: type['Spin'] = None
    quantum_numbers: list[type['Register']] = None
    max_particles: int = None
    particle: type['Particle'] = None

    @classmethod
    def size(cls) -> int:
        return cls.particle.size() * cls.max_particles

    @classmethod
    def initial_state(cls) -> Expr:
        return FieldState(*[cls.particle.initial_state() for _ in range(cls.max_particles)])

    def __init__(self):
        super().__init__()
        # pylint: disable-next=not-callable
        self.particles = [self.particle(index=ipart) for ipart in range(self.max_particles)]

    @classmethod
    def annihilation_op(
        cls,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        ann_op = 0
        for ipart in range(cls.max_particles):
            zeroed_particles = cls.max_particles - ipart - 1
            annihilator_args = list(cls.projection_op(0, zeroed_particles).args)
            annihilator_args.append(cls.particle.annihilation_op(momentum, spin, **quantum_numbers))
            annihilator_args.extend(cls.projection_op(1, ipart).args)
            annihilator = TensorProduct(*annihilator_args)
            ann_op += annihilator * StepSymmetrizer(cls.particle.size(), ipart + 1)

        return ann_op

    @classmethod
    def creation_op(
        cls,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(cls.annihilation_op(momentum, spin, **quantum_numbers))

    @classmethod
    def projection_op(cls, state: int, num_particles: int) -> TensorProduct:
        if num_particles == 0:
            return S.One
        proj = OrthogonalProductKet(state) * OrthogonalProductBra(state)
        identity = IdentityProduct(cls.particle.size() - 1)
        args = []
        for _ in range(num_particles):
            args.append(proj)
            args.append(identity)
        return TensorProduct(*args)


class Particle(CompoundRegister):
    """Register for a single particle."""
    field: 'Field' = None

    @classmethod
    def size(cls) -> int:
        return (1 + int(cls.field.spin is not None) + cls.field.momentum.spatial_dimension
                + len(cls.field.quantum_numbers))

    @classmethod
    def initial_state(cls) -> Expr:
        return ParticleState(0, (0,) * (cls.size() - 1))

    def __init__(
        self,
        index: Optional[int] = None
    ):
        super().__init__()
        if index is not None:
            self._name += f'[{index}]'

        self.index = index

        self.registers = [
            Occupancy(),
            self.field.momentum()
        ]
        if self.field.spin is not None:
            self.registers.append(self.field.spin())
        self.registers.extend(reg_cls() for reg_cls in self.field.quantum_numbers)

    @property
    def occupancy(self) -> Register:
        return self.registers[0]

    @property
    def momentum(self) -> Register:
        return self.registers[1]

    @property
    def spin(self) -> Register:
        if self.field.spin is None:
            return None
        return self.registers[2]

    def quantum_number(self, name) -> Register:
        return next(reg for reg in self.registers if reg.name == name)

    @classmethod
    def annihilation_op(
        cls,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Expr:
        if not isinstance(momentum, tuple):
            momentum = (momentum,)
        # control = OuterProduct(OrthogonalKet(0), OrthogonalBra(1))
        control = Control(0, 1)
        source_labels = momentum
        if spin is not None:
            source_labels += (spin,)
        source_labels += tuple(quantum_numbers[reg_cls.__name__]
                               for reg_cls in cls.field.quantum_numbers)
        scrap = OuterProduct(
            OrthogonalProductKet(*((0,) * len(source_labels))),
            OrthogonalProductBra(*source_labels)
        )
        return TensorProduct(control, scrap)

    @classmethod
    def creation_op(
        cls,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(cls.annihilation_op(momentum, spin, **quantum_numbers))


class Occupancy(Register):
    """Single-bit P/A register."""
    dimension = 2


class MomentumComponent(RegisterBase):
    """Single momentum component."""
    @classmethod
    def size(cls) -> int:
        return 1

    @classmethod
    def initial_state(cls) -> Ket:
        return OrthogonalProductKet(0)

    def __init__(self, dname: str):
        super().__init__()
        self._name = 'Momentum_' + dname


class Momentum(CompoundRegister):
    """Momentum register."""
    spatial_dimension: int = None
    _subclasses = []

    @classmethod
    def size(cls) -> int:
        return cls.spatial_dimension

    @classmethod
    def of_dimension(cls, spatial_dimension: int) -> type['Momentum']:
        return cls._subclasses[spatial_dimension - 1]

    def __init__(self):
        super().__init__()
        self._name = 'Momentum'
        self.components = [MomentumComponent(dname)
                           for dname in ['x', 'y', 'z'][:self.spatial_dimension]]

    def __getitem__(self, index) -> MomentumComponent:
        return self.components[index]


Momentum._subclasses.extend(type(f'Momentum{i}D', (Momentum,), {'spatial_dimension': i})
                            for i in range(1, 4))


class Spin(Register):
    """Spin register."""
    spin: int = None
    _subclasses = []

    @classmethod
    def of_spin(cls, spin: int) -> type['Spin']:
        if spin == 0:
            return None
        return cls._subclasses[spin - 1]

    def __init__(self):
        super().__init__()
        self._name = 'Spin'


Spin._subclasses.extend(type(f'Spin{i}Hal' + ('f' if i == 1 else 'ves'), (Spin,), {'spin': i})
                        for i in range(1, 4))


class Control(OuterProduct):
    """Control operator for particle registers."""
    def __new__(cls, *args):
        if not (len(args) == 2 and all(arg in (0, 1) for arg in args)):
            raise ValueError(f'Invalid constructor argument {args} for control')

        return super().__new__(cls, OrthogonalKet(args[0]), OrthogonalBra(args[1]))

    def _eval_adjoint(self):
        return Control(self.args[1].args[0], self.args[0].args[0])

    def _print_operator_name(self, printer, *args):
        return 'Ctrl'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('Ctrl')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathfrak{C}'

    def _sympystr(self, printer, *args):
        return Operator._sympystr(self, printer, *args)

    def _sympyrepr(self, printer, *args):
        return Operator._sympyrepr(self, printer, *args)

    def _pretty(self, printer, *args):
        return Operator._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return r'%s_{%s%s}' % (
            self._print_operator_name_latex(printer, *args),
            self.args[0].args[0],
            self.args[1].args[0]
        )


class ParticleSwap(Operator):
    """Particle-level swap operator implemented as a sympy Operator."""
    is_hermitian = True
    is_unitary = True

    @classmethod
    def default_args(cls):
        return ('PSWAP',)

    def __new__(cls, *args, **kwargs):
        if not (len(args) == 3 and all(isinstance(arg, int) for arg in args)):
            raise ValueError('ParticleSwap requires three integer arguments (particle size, index1,'
                             ' index2)')
        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'PSWAP'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('PSWAP')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathrm{PSWAP}'

    def _print_contents(self, printer, *args):
        return (f'{self._print_operator_name(printer, *args)}[{self.args[0]}]'
                f'({self.args[1]},{self.args[2]})')

    def _print_contents_latex(self, printer, *args):
        return r'%s_{%s}\left({%s}, {%s}\right)' % (
            (self._print_operator_name_latex(printer, *args),) + self.args
        )

    @staticmethod
    def swap_particles(
        ket: Union[KetBase, TensorProduct],
        size: int,
        index1: int,
        index2: int
    ) -> ProductKet:
        if isinstance(ket, KetBase):
            ket = to_tensor_product(ket)
        if not (isinstance(ket, TensorProduct)
                and all(isinstance(arg, KetBase) and len(arg.args) == 1 for arg in ket.args)):
            raise ValueError(f'Argument {ket} is not a product state')

        if len(ket.args) <= max(index1, index2) * size:
            raise ValueError(f'State {ket} is inconsistent with the permutation')
        new_kets = list(ket.args)
        for source, dest in zip((index1, index2), (index2, index1)):
            for ireg in range(size):
                new_kets[dest * size + ireg] = ket.args[source * size + ireg]

        return to_product_state(TensorProduct(*new_kets))

    def _apply_operator(self, ket: Union[KetBase, TensorProduct], **options) -> ProductKet:
        return self.swap_particles(ket, *self.args)  # pylint: disable=no-value-for-parameter

    def _eval_inverse(self):
        return self

    def _eval_rewrite(self, rule, args, **hints):
        size = args[0]
        index1, index2 = sorted(args[1:])
        if rule == Register:
            return PermutationOperator(*(
                tuple(range(index2 * size, (index2 + 1) * size))
                + tuple(range(index1 * size, (index1 + 1) * size))
            ))
        return None


class StepSymmetrizer(Operator):
    """Step-symmetrizer of a bosonic field register."""
    is_hermitian = True

    def __new__(cls, *args, **kwargs):
        if not (len(args) == 2 and all(isinstance(arg, int) for arg in args)):
            raise ValueError('StepSymmetrizer requires two integer arguments (particle size,'
                             ' updated number of particles)')
        return super().__new__(cls, *args, **kwargs)

    def _print_operator_name(self, printer, *args):
        return 'S'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('S')

    def _print_operator_name_latex(self, printer, *args):  # pylint: disable=unused-argument
        return r'\mathcal{S}'

    def _print_contents(self, printer, *args):
        return (f'{self._print_operator_name(printer, *args)}[{self.args[0]}]'
                f'({self.args[1]}<-{self.args[1]-1})')

    def _print_contents_latex(self, printer, *args):
        return r'%s_{%s}\left({%s}\leftarrow{%s}\right)' % (
            self._print_operator_name_latex(printer, *args), self.args[0],
            self.args[1], self.args[1] - 1
        )

    def _apply_operator(self, ket: Union[KetBase, TensorProduct], **options) -> Expr:
        particle_size, new_num_particles = self.args  # pylint: disable=unbalanced-tuple-unpacking
        result_states = [ket]
        if isinstance(ket, KetBase):
            ket = to_tensor_product(ket)
        for ipart in range(new_num_particles - 1):
            result_states.append(
                ParticleSwap.swap_particles(ket, particle_size, new_num_particles - 1, ipart)
            )
        return Add(*result_states) / sqrt(new_num_particles)

    def _eval_inverse(self):
        return self

    def _eval_rewrite(self, rule, args, **hints):
        particle_size, new_num_particles = self.args  # pylint: disable=unbalanced-tuple-unpacking
        if rule == ParticleSwap:
            if new_num_particles == 1:
                return IdentityOperator()

            ops = [IdentityOperator()]
            ops += [ParticleSwap(particle_size, new_num_particles - 1, ipart)
                    for ipart in range(new_num_particles - 1)]
            return Add(*ops) / sqrt(new_num_particles)
        return None
