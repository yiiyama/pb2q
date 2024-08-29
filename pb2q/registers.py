"""Registers."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from sympy import Expr, S
from sympy.physics.quantum import (Dagger, Ket, Operator, OuterProduct,
                                   TensorProduct)
from .field import FieldDefinition
from .sympy import IdentityProduct, OrthogonalProductBra, OrthogonalProductKet
from .states import (Control, FieldOperator, FieldState, ParticleState, StepAntisymmetrizer,
                     StepSymmetrizer, UniverseState)


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
        ann_op = S.Zero
        for ipart in range(cls.max_particles):
            num_unoccupied = cls.max_particles - ipart - 1
            args = [cls.particle.projection_op(0) for _ in range(num_unoccupied)]
            args.append(cls.particle.annihilation_op(momentum, spin, **quantum_numbers))
            args.extend(cls.particle.projection_op(1) for _ in range(ipart))
            annihilator = FieldOperator(*args)
            if ipart > 0:
                if cls.spin.spin % 2 == 0:
                    annihilator *= StepSymmetrizer(ipart + 1)
                else:
                    annihilator *= StepAntisymmetrizer(ipart + 1)
            ann_op += annihilator

        return ann_op

    @classmethod
    def creation_op(
        cls,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(cls.annihilation_op(momentum, spin, **quantum_numbers))


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
        if cls.field.spin is not None:
            if spin is None:
                raise ValueError('Spin value missing')
            source_labels += (spin,)
        try:
            source_labels += tuple(quantum_numbers[reg_cls.__name__]
                                   for reg_cls in cls.field.quantum_numbers)
        except KeyError as exc:
            raise ValueError('Quantum number missing') from exc

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

    @classmethod
    def projection_op(
        cls,
        state: int
    ):
        return TensorProduct(Control(state, state), IdentityProduct(cls.size() - 1))


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
