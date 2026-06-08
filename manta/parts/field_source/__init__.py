"""Field-source parts — parts that EMIT a field disturbance.

Most parts READ fields (a Magnetometer reads the MagField, a Mass reads
the GravityField). A *field source* is the dual: a part bolted to a craft
that CONTRIBUTES a disturbance to a world field, so other craft feel /
see it. The emitted disturbance rides the carrying craft (its pose tracks
the body), built and registered onto the world's field at compile time by
`World._register_field_sources` (idempotent, like planet registration).

  * `GravitySource`  — a point-mass pull (simulate a massive body).
  * `MagneticSource` — a body-fixed dipole (motors, magnets).
  * `OpticalSource`  — a semantic ellipsoid so cameras can see the craft.

A source adds no wrench of its own (it is a field emitter, not a force on
its carrier); give the carrier real `Mass`/`Thruster` parts for dynamics.
"""

from .base import FieldSource
from .gravity_source import GravitySource
from .magnetic_source import MagneticSource
from .optical_source import OpticalSource

__all__ = ["FieldSource", "GravitySource", "MagneticSource", "OpticalSource"]
