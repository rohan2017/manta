"""Inter-craft couplings.

A `Coupling` joins two crafts so their dynamics are evaluated in the
same compiled tick (so per-tick forces between them are symbolically
consistent, and an EKF over the connected component sees full
cross-craft Jacobians). `World.compile` detects coupled connected
components and routes them to `compile_world_tick`.

The base `Coupling` ABC lives in `.base`; concrete couplings live in
sibling modules (`.tether`, …). Import via:
    from manta.couplings import Coupling, Tether   # or
    from manta import Coupling
"""

from .base import Coupling
from .tether import Tether

__all__ = ["Coupling", "Tether"]
