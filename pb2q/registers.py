"""Registers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Union
from sympy import Add, Expr, S, factorial
from sympy.physics.quantum import Dagger, Ket, IdentityOperator, Operator
from .field import FieldDefinition
from .operators import (PresenceProjection, AbsenceProjection, FieldOperator, StepAntisymmetrizer,
                        StepSymmetrizer, UniverseOperator, generate_perm)
from .states import FieldKet, MomentumKet, ParticleKet, QNumberKet, UniverseKet


class RegisterBase(ABC):
    """Base register class."""
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        """Return the total number of physical registers."""
        return None

    @abstractmethod
    def null_state(self) -> Expr:
        """Return the zero state."""

    def interpret(self, state: Expr) -> str:  # pylint: disable=unused-argument
        """Give a representation string of a register state."""
        return ''


class Register(RegisterBase):
    """Fixed-dimension register class."""
    @property
    def size(self) -> int:
        return 1


class CompoundRegister(RegisterBase):
    """A register that consists of other registers."""


class Universe(CompoundRegister):
    """Register representing the entire universe."""
    def __init__(
        self,
        fields: list[Union['Field', FieldDefinition]],
        spatial_dimension: int
    ):
        super().__init__('Universe')
        self.spatial_dimension = spatial_dimension
        self.fields = {f.name: f if isinstance(f, Field) else Field(f, self) for f in fields}

    @property
    def size(self) -> int:
        return sum(field.size() for field in self.fields.values())

    def null_state(self) -> Expr:
        return UniverseKet(*[field.null_state() for field in self.fields.values()])

    def annihilation_op(
        self,
        field_name: str,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        ops = []
        for name, field in self.fields.items():
            if name == field_name:
                ops.append(field.annihilation_op(momentum, spin, **quantum_numbers))
            else:
                ops.append(IdentityOperator())
        return UniverseOperator(*ops)

    def creation_op(
        self,
        field_name: str,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return Dagger(self.annihilation_op(field_name, momentum, spin, **quantum_numbers))


class Field(CompoundRegister):
    """Register for a single field species."""
    def __init__(
        self,
        definition: FieldDefinition,
        universe: Optional[Universe] = None,
        spatial_dimension: Optional[int] = None
    ):
        super().__init__(definition.name)
        if universe:
            self._universe = universe
        else:
            if not spatial_dimension:
                raise ValueError('Field needs a spatial dimension if not given a universe')
            self._universe = Universe([self], spatial_dimension)

        self.max_particles = definition.max_particles
        self.momentum = Momentum(self._universe.spatial_dimension)
        self.spin = Spin(definition.spin)
        self.quantum_numbers = {name: QNumber(name, dim)
                                for name, dim in definition.quantum_numbers}

        self.particle = Particle(self)

    @property
    def size(self) -> int:
        return self.particle.size * self.max_particles

    def state(self, particle_args: Iterable[tuple]) -> Expr:
        """Return a linear combination of FieldKets corresponding to the symmetrized tensorproducts
        of the given particle states.

        Args:
            particle_args: List of arguments to ParticleKet.
        """
        if (np := len(particle_args)) > self.max_particles:
            raise ValueError('Too many particle state arguments')

        result = S.Zero
        for ip, perm in enumerate(generate_perm(range(np))):
            particle_states = [self.particle.state(*particle_args[idx]) for idx in perm]
            particle_states += [self.particle.null_state() for _ in range(self.max_particles - np)]
            ket = FieldKet(*particle_states)
            if self.spin.spin % 2 != 0 and ip % 2 == 1:
                ket *= -1
            result += ket

        return result / factorial(np)

    def null_state(self) -> Expr:
        return FieldKet(*[self.particle.null_state() for _ in range(self.max_particles)])

    def annihilation_part_op(
        self,
        ipart: int,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        """Return the annihilation op of ipart-th particle register."""
        args = [PresenceProjection() for _ in range(ipart)]
        args.append(self.particle.annihilation_op(momentum, spin, **quantum_numbers))
        args.extend(AbsenceProjection() for _ in range(ipart + 1, self.max_particles))
        annihilator = FieldOperator(*args)
        if ipart > 0:
            if self.spin.spin % 2 == 0:
                annihilator *= StepSymmetrizer(ipart + 1)
            else:
                annihilator *= StepAntisymmetrizer(ipart + 1)
        return annihilator

    def creation_part_op(
        self,
        ipart: int,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return Dagger(self.annihilation_part_op(ipart, momentum, spin, **quantum_numbers))

    def annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return Add(*[self.annihilation_part_op(ipart, momentum, spin, **quantum_numbers)
                     for ipart in range(self.max_particles)])

    def creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))


class Particle(CompoundRegister):
    """Register for a single particle."""
    def __init__(
        self,
        field: Union[Field, FieldDefinition],
        spatial_dimension: Optional[int] = None
    ):
        super().__init__(f'{field.name}Particle')
        if isinstance(field, Field):
            self._field = field
        else:
            self._field = Field(field, spatial_dimension=spatial_dimension)

    @property
    def size(self) -> int:
        # Momentum counts as one register
        return (2 + int(self._field.spin.spin != 0) + len(self._field.quantum_numbers))

    def state(
        self,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Expr:
        if isinstance(momentum, int):
            momentum = (momentum,)

        try:
            qnumber = tuple(quantum_numbers[name] for name in self._field.quantum_numbers)
        except KeyError as exc:
            raise ValueError('Quantum number missing') from exc

        if self._field.spin.spin != 0:
            if (spin is None or abs(spin) > self._field.spin.spin
                    or spin % 2 != self._field.spin.spin % 2):
                raise ValueError(f'Invalid spin value {spin}')
            qnumber = (spin,) + qnumber

        return ParticleKet(momentum, qnumber)

    def null_state(self) -> Expr:
        return ParticleKet()

    def null_state_args(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        momentum = self._field.momentum.null_state_args()
        qnumber = ()
        if self._field.spin.spin != 0:
            qnumber += (0,)
        qnumber += (0,) * len(self._field.quantum_numbers)
        return momentum, qnumber

    def annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return self.null_state() * self.state(momentum, spin, **quantum_numbers).dual

    def creation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))


class Momentum(Register):
    """Momentum register."""
    def __init__(self, spatial_dimension: int):
        super().__init__('Momentum')
        self.spatial_dimension = spatial_dimension

    def null_state(self) -> Ket:
        return MomentumKet(*self.null_state_args())

    def null_state_args(self) -> tuple[int, ...]:
        return (0,) * self.spatial_dimension


class QNumber(Register):
    """Generic quantum number register."""
    def __init__(self, name: str, dim: int):
        super().__init__(name)
        self.dim = dim

    def null_state(self) -> Ket:
        return QNumberKet(0)


class Spin(QNumber):
    """Spin register."""
    def __init__(self, spin: int):
        super().__init__('Spin', spin + 1)

    @property
    def spin(self) -> int:
        return self.dim - 1
