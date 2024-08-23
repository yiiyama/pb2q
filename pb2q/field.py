"""Field definition."""

from dataclasses import dataclass, field


@dataclass
class FieldDefinition:
    """Field definition."""
    name: str
    spin: int
    max_particles: int
    quantum_numbers: list[tuple[str, int]] = field(default_factory=list)
