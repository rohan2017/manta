"""Inter-craft couplings.

A `Coupling` joins two crafts so their dynamics are evaluated in the
same compiled tick (so per-tick forces between them are symbolically
consistent, and an EKF over the connected component sees full
cross-craft Jacobians). `World.compile` detects coupled connected
components and routes them to `compile_coupled_tick`.

The base `Coupling` ABC lives in `manta_next.world`; concrete couplings
live here.
"""

from .base import Coupling
from .tether import Tether

__all__ = ["Coupling", "Tether"]
