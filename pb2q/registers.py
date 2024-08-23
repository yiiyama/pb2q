"""Representation of registers."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from sympy.physics.quantum import Dagger, Ket, Operator, OrthogonalBra, OrthogonalKet, TensorProduct
from .field import FieldDefinition


class RegisterBase(ABC):
    """Base register class."""
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the total number of physical registers."""

    @abstractmethod
    def initial_state(self) -> Ket:
        """Return the zero state."""


class Register(RegisterBase):
    """Fixed-dimension register class."""
    def __init__(self, name: str, dimension: int):
        super().__init__(name)
        self.dimension = dimension

    @property
    def size(self) -> int:
        return 1

    def initial_state(self) -> Ket:
        return OrthogonalKet(0)


class CompoundRegister(RegisterBase):
    """A register that consists of other registers."""
    def __init__(self, name: str):
        super().__init__(name)
        self._components = []

    def initial_state(self) -> Ket:
        return OrthogonalKet(*((0,) * self.size))

    def interpret(self, state: Ket) -> str:
        return ''


class Universe(CompoundRegister):
    """Total physical register."""
    def __init__(
        self,
        fields: list[FieldDefinition],
        spatial_dimension: int
    ):
        super().__init__('universe')
        self.spatial_dimension = spatial_dimension
        self.fields = {definition.name: Field(definition, universe=self)
                       for definition in fields}

    @property
    def size(self) -> int:
        return sum(field.size for field in self.fields.values())


class Field(CompoundRegister):
    """Register for a single field species."""
    def __init__(
        self,
        definition: FieldDefinition,
        spatial_dimension: Optional[int] = None,
        universe: Optional[Universe] = None
    ):
        super().__init__(definition.name)
        self.spin = definition.spin
        self.max_particles = definition.max_particles
        self.quantum_numbers = definition.quantum_numbers

        if universe is None:
            if spatial_dimension is None:
                raise RuntimeError('spatial_dimension or universe required')
            self.universe = Universe([], spatial_dimension)
            self.universe.fields[self.name] = self
        else:
            self.universe = universe

        self.particles = [Particle(self, index=ipart) for ipart in range(self.max_particles)]

    @property
    def size(self) -> int:
        return self.particle_size * len(self.particles)

    @property
    def particle_size(self) -> int:
        return 2 + self.universe.spatial_dimension + len(self.quantum_numbers)

    def particle_initial_state(self) -> Ket:
        return OrthogonalKet(*((0,) * self.particle_size))

    def annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        ann_op = 0
        for ipart, particle in enumerate(self.particles):
            op_target = ()
            op_source = ()
            zeroed_particles = self.max_particles - ipart - 1
            if zeroed_particles != 0:
                proj = self.null_projection_op(zeroed_particles)
                op_target += proj.ket.args
                op_source += proj.bra.args
            p_op = particle.annihilation_op(momentum, spin, **quantum_numbers)
            op_target += p_op.ket.args
            op_source += p_op.bra.args
            if ipart != 0:
                target, source = id_projector(ipart + 1)
                op_target += target
                op_source += source
            ann_op += (OrthogonalKet(*op_target) * OrthogonalBra(*op_source)) * symmetrizer(ipart)

        return ann_op

    def creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))

    def particle_annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        source_labels = (1,) + momentum
        if spin is not None:
            source_labels += (spin,)
        source_labels += tuple(quantum_numbers[name] for name, _ in self.quantum_numbers)
        return self.particle_initial_state() * OrthogonalBra(*source_labels)

    def particle_creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))


class Particle(CompoundRegister):
    """Register for a single particle."""
    @staticmethod
    def register_size(
        field: FieldDefinition,
        spatial_dimension: int
    ) -> int:
        return 2 + spatial_dimension + len(field.quantum_numbers)

    def __init__(
        self,
        field: Union[FieldDefinition, Field],
        spatial_dimension: Optional[int] = None,
        index: Optional[int] = None
    ):
        name = field.name
        if index is not None:
            name += f'[{index}]'
        super().__init__(name)

        self.index = index

        if isinstance(field, Field):
            self.field = field
        else:
            if spatial_dimension is None:
                raise RuntimeError('spatial_dimension required when constructing from'
                                   ' FieldDefinition')
            self.field = Field(field, spatial_dimension=spatial_dimension)

        self.registers = [
            Occupancy(),
            Momentum(self.field.universe.spatial_dimension),
            Spin(self.field.spin)
        ] + [Register(name, dim) for name, dim in self.field.quantum_numbers]

    def __getattr__(self, name: str) -> Any:
        try:
            return next(reg for reg in self.registers[3:] if reg.name == name)
        except StopIteration as exc:
            raise AttributeError from exc

    @property
    def size(self) -> int:
        return len(self.registers)

    @property
    def occupancy(self) -> Register:
        return self.registers[0]

    @property
    def momentum(self) -> Register:
        return self.registers[1]

    @property
    def spin(self) -> Register:
        return self.registers[2]

    def annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return self.field.particle_annihilation_op(momentum, spin, **quantum_numbers)

    def creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return self.field.particle_creation_op(momentum, spin, **quantum_numbers)


class Occupancy(Register):
    """Single-bit P/A register."""
    def __init__(self):
        super().__init__('occupancy', 2)


class MomentumComponent(RegisterBase):
    """Single momentum component."""
    @property
    def size(self) -> int:
        return 1

    def initial_state(self) -> Ket:
        return OrthogonalKet(0)


class Momentum(CompoundRegister):
    """Momentum register."""
    def __init__(self, spatial_dimension: int):
        super().__init__('momentum')
        self.components = [MomentumComponent(dname)
                           for dname in ['x', 'y', 'z'][:spatial_dimension]]

    def __getitem__(self, index) -> MomentumComponent:
        return self.components[index]

    @property
    def size(self) -> int:
        return len(self.components)


class Spin(Register):
    """Spin register."""
    def __init__(self, spin: int):
        super().__init__('spin', spin + 1)
