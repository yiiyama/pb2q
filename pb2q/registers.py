"""Representation of registers."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from sympy import S
from sympy.physics.quantum import Dagger, IdentityOperator, Ket, Operator, TensorProduct
from .field import FieldDefinition
from .sympy import OrthogonalProductBra, OrthogonalProductKet


class RegisterBase(ABC):
    """Base register class."""
    @classmethod
    @abstractmethod
    def size(cls) -> int:
        """Return the total number of physical registers."""

    @classmethod
    @abstractmethod
    def initial_state(cls) -> Ket:
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
    def initial_state(cls) -> Ket:
        return OrthogonalProductKet(0)


class CompoundRegister(RegisterBase):
    """A register that consists of other registers."""
    @classmethod
    def initial_state(cls) -> Ket:
        return OrthogonalProductKet(*((0,) * cls.size()))

    def interpret(self, state: Ket) -> str:
        return ''


class Universe(CompoundRegister):
    """Total physical register."""
    _singleton: 'Universe' = None

    @classmethod
    def size(cls) -> int:
        return sum(field.size for field in cls._singleton.fields.values())

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
        field_cls = type(
            definition.name,
            (Field,),
            {
                'momentum': momentum,
                'spin': spin,
                'quantum_numbers': quantum_numbers,
                'max_particles': definition.max_particles
            }
        )
        particle_cls = type(
            definition.name + 'Particle',
            (Particle,),
            {
                'field': field_cls
            }
        )
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
            ann_op += TensorProduct(*annihilator_args) * step_symmetrizer(ipart)

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
        identities = [IdentityOperator() for _ in range(cls.particle.size() - 1)]
        args = []
        for _ in range(num_particles):
            args.append(proj)
            args.extend(identities)
        return TensorProduct(*args)


class Particle(CompoundRegister):
    """Register for a single particle."""
    field: 'Field' = None

    @classmethod
    def size(cls) -> int:
        return 2 + cls.field.momentum.spatial_dimension + len(cls.field.quantum_numbers)

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
            self.field.momentum(),
            self.field.spin()
        ] + [reg_cls() for reg_cls in self.field.quantum_numbers]

    @property
    def occupancy(self) -> Register:
        return self.registers[0]

    @property
    def momentum(self) -> Register:
        return self.registers[1]

    @property
    def spin(self) -> Register:
        return self.registers[2]

    def quantum_number(self, name) -> Register:
        return next(reg for reg in self.registers if reg.name == name)

    @classmethod
    def annihilation_op(
        cls,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        if not isinstance(momentum, tuple):
            momentum = (momentum,)
        source_labels = (1,) + momentum
        if spin is not None:
            source_labels += (spin,)
        source_labels += tuple(quantum_numbers[reg_cls.__name__]
                               for reg_cls in cls.field.quantum_numbers)
        return cls.initial_state() * OrthogonalProductBra(*source_labels)

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
        return cls._subclasses[spin]

    def __init__(self):
        super().__init__()
        self._name = 'Spin'


Spin._subclasses.extend(type(f'Spin{i}Halves', (Spin,), {'spin': i})
                        for i in range(2))
