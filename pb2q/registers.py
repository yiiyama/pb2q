"""Registers."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from sympy import Expr, S
from sympy.physics.quantum import Dagger, Ket, Operator, OuterProduct, TensorProduct
from .field import FieldDefinition
from .momentum import MomentumKet
from .operators import Control, FieldOperator, StepAntisymmetrizer, StepSymmetrizer
from .sympy import IdentityProduct, OrthogonalProductBra, OrthogonalProductKet
from .states import FieldKet, NullKet, ParticleKet, QNumberKet, UniverseKet


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
    """Total physical register."""
    def __init__(
        self,
        fields: list[Union['Field', FieldDefinition]],
        spatial_dimension: int
    ):
        super().__init__('Universe')
        self.spatial_dimension = spatial_dimension
        self.fields = {f.name: f if isinstance(f, Field) else Field(f) for f in fields}

    @property
    def size(self) -> int:
        return sum(field.size() for field in self.fields.values())

    def null_state(self) -> Expr:
        return UniverseKet(*[field.null_state() for field in self.fields.values()])


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

    def null_state(self) -> Expr:
        return FieldKet(*[NullKet() for _ in range(self.max_particles)])

    def annihilation_op(
        self,
        momentum: tuple[int, ...],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Operator:
        ann_op = S.Zero
        for ipart in range(self.max_particles):
            num_unoccupied = self.max_particles - ipart - 1
            args = [self.particle.projection_op(0) for _ in range(num_unoccupied)]
            args.append(self.particle.annihilation_op(momentum, spin, **quantum_numbers))
            args.extend(self.particle.projection_op(1) for _ in range(ipart))
            annihilator = FieldOperator(*args)
            if ipart > 0:
                if self.spin.spin % 2 == 0:
                    annihilator *= StepSymmetrizer(ipart + 1)
                else:
                    annihilator *= StepAntisymmetrizer(ipart + 1)
            ann_op += annihilator

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

    def null_state(self) -> Expr:
        return ParticleKet(0, *self.null_state_args())

    def null_state_args(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        momentum = self._field.momentum.null_state_args()
        qnumber = ()
        if self._field.spin.spin != 0:
            qnumber += (0,)
        qnumber += (0,) * len(self._field.quantum_numbers)
        return momentum, qnumber

    def annihilation_op(
        self,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ) -> Expr:
        if not isinstance(momentum, tuple):
            momentum = (momentum,)

        # control = OuterProduct(OrthogonalKet(0), OrthogonalBra(1))
        control = Control(0, 1)
        source_labels = momentum
        if self._field.spin.spin != 0:
            if spin is None:
                raise ValueError('Spin value missing')
            source_labels += (spin,)
        try:
            source_labels += tuple(quantum_numbers[name] for name in self._field.quantum_numbers)
        except KeyError as exc:
            raise ValueError('Quantum number missing') from exc

        scrap = OuterProduct(
            OrthogonalProductKet(*self.null_state_args()),
            OrthogonalProductBra(*source_labels)
        )
        return TensorProduct(control, scrap)

    def creation_op(
        self,
        momentum: Union[int, tuple[int, ...]],
        spin: Optional[int] = None,
        **quantum_numbers
    ):
        return Dagger(self.annihilation_op(momentum, spin, **quantum_numbers))

    def projection_op(
        self,
        state: int
    ):
        return TensorProduct(Control(state, state), IdentityProduct(self.size - 1))


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
