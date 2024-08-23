"""Representation of registers."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from sympy.physics.quantum import Dagger, Ket, Operator, OrthogonalBra, OrthogonalKet
from .field import FieldDefinition


class RegisterBase(ABC):
    """Base register class."""
    def __init__(
        self,
        name: str,
        position: int,
    ):
        self.name = name
        self.position = position

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the register size (base 2)"""

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the total number of physical registers."""

    @abstractmethod
    def initial_state(self) -> Ket:
        """Return the zero state."""


class Register(RegisterBase):
    """Physical register class."""
    @staticmethod
    def min_register_size(dimension: int) -> int:
        return int(np.ceil(np.log2(dimension)).astype(int))

    def __init__(
        self,
        name: str,
        position: int,
        dimension: int
    ):
        super().__init__(name, position)
        self.dimension = dimension

    @property
    def count(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return Register.min_register_size(self.dimension)

    def initial_state(self) -> Ket:
        return OrthogonalKet(0)


class CompoundRegister(RegisterBase):
    """A register that consists of other registers."""
    def initial_state(self) -> Ket:
        return OrthogonalKet(*((0,) * self.count))

    def interpret(self, state: Ket) -> str:
        return ''


class Universe(CompoundRegister):
    """Total physical register."""
    def __init__(
        self,
        fields: list[FieldDefinition],
        spatial_dimension: int,
        momentum_precision: int
    ):
        self.fields = {}
        pos = 0
        for definition in fields:
            field = Field(definition, spatial_dimension, momentum_precision, pos)
            self.fields[field.name] = field
            pos += field.size

        super().__init__('universe', 0)

    @property
    def size(self) -> int:
        return sum(field.size for field in self.fields.values())

    @property
    def count(self) -> int:
        return sum(field.count for field in self.fields.values())


class Field(CompoundRegister):
    """Register for a single field species."""
    def __init__(
        self,
        definition: FieldDefinition,
        spatial_dimension: int,
        momentum_precision: int,
        position: int = 0
    ):
        self._definition = definition
        self._spatial_dimension = spatial_dimension
        self._momentum_precision = momentum_precision

        self.particles = []
        pos = position
        for ipart in range(definition.max_particles):
            self.particles.append(
                Particle(definition, spatial_dimension, momentum_precision, pos, ipart)
            )
            pos += self.particles[-1].size

        super().__init__(definition.name, position)

    @property
    def size(self) -> int:
        return self.particle_size * len(self.particles)

    @property
    def count(self) -> int:
        return self.particle_count * len(self.particles)

    @property
    def particle_size(self) -> int:
        return Particle.register_size(self._definition, self._spatial_dimension,
                                      self._momentum_precision)

    @property
    def particle_count(self) -> int:
        return 2 + self._spatial_dimension + len(self._definition.quantum_numbers)

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
            zeroed_particles = self._definition.max_particles - ipart - 1
            if zeroed_particles != 0:
                proj = null_projector(zeroed_particles)
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


class Particle(CompoundRegister):
    """Register for a single particle."""
    @staticmethod
    def register_size(
        field: FieldDefinition,
        spatial_dimension: int,
        momentum_precision: int
    ) -> int:
        return (1 + MomentumRegister.register_size(spatial_dimension, momentum_precision)
                + SpinRegister.register_size(field.spin)
                + sum(Register.min_register_size(dim) for _, dim in field.quantum_numbers))

    @staticmethod
    def register_count(
        field: FieldDefinition,
        spatial_dimension: int
    ) -> int:
        return 2 + spatial_dimension + len(field.quantum_numbers)

    def __init__(
        self,
        field: FieldDefinition,
        spatial_dimension: int,
        momentum_precision: int,
        position: int = 0,
        index: Optional[int] = None
    ):
        pos = position
        
        pos += self.registers[-1].size
        self.registers.append(MomentumRegister(
            pos,
            spatial_dimension,
            momentum_precision
        ))
        pos += self.registers[-1].size
        pos += 1
        self.registers.append(SpinRegister(pos, field.spin))
        for name, dim in field.quantum_numbers:
            self.registers.append(Register(name, pos, dim))
            pos += self.registers[-1].size

        name = field.name
        if index is not None:
            name += f'[{index}]'

        super().__init__(name, position)

    def __getattr__(self, name: str) -> Any:
        try:
            return next(reg for reg in self.registers[3:] if reg.name == name)
        except StopIteration as exc:
            raise AttributeError from exc

    @property
    def size(self) -> int:
        return sum(reg.size for reg in self.registers)

    @property
    def count(self) -> int:
        return sum(reg.count for reg in self.registers)

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
        source_labels = (1,) + momentum
        if spin is not None:
            source_labels += (spin,)
        source_labels += tuple(quantum_numbers[reg.name] for reg in self.registers[3:])
        return self.initial_state() * OrthogonalBra(*source_labels)

    def creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))


class OccupancyRegister(Register):
    """Single-bit P/A register."""
    def __init__(self, position: int):
        super().__init__('occupancy', position, 2)

    @property
    def size(self) -> int:
        return 1


class MomentumRegister(CompoundRegister):
    """Momentum register."""
    @staticmethod
    def register_size(spatial_dimension: int, precision: int) -> int:
        return (1 + precision) * spatial_dimension

    def __init__(self, position: int, spatial_dimension: int, precision: int):
        self._regs = []
        pos = position
        for dname in ['x', 'y', 'z'][:spatial_dimension]:
            self._regs.append(
                Register(dname, pos, 2 ** (precision + 1))
            )
            pos += self._regs[-1].size

        super().__init__('momentum', position)

    def __getitem__(self, index) -> Register:
        return self._regs[index]

    @property
    def size(self) -> int:
        return MomentumRegister.register_size(len(self._regs), self._regs[0].size - 1)

    @property
    def count(self) -> int:
        return len(self._regs)


class SpinRegister(Register):
    """Spin register."""
    @staticmethod
    def register_size(spin: int) -> int:
        return Register.min_register_size(spin + 1)

    def __init__(self, position: int, spin: int):
        super().__init__('spin', position, spin + 1)
