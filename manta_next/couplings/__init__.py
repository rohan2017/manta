"""Inter-craft couplings.

A `Coupling` joins two crafts so their dynamics are evaluated in the
same compiled tick (so per-tick forces between them are symbolically
consistent, and an EKF over the connected component sees full
cross-craft Jacobians). `World.compile` detects coupled connected
components and routes them to `compile_coupled_tick`.

The base `Coupling` ABC lives in `.base`; concrete couplings live in
sibling modules (`.tether`, …). `manta_next.world` re-exports
`Coupling` for backward compatibility with `from manta_next.world
import Coupling` — that re-export is scheduled for removal in a
follow-up cleanup.
"""

from .base import Coupling
from .tether import Tether

__all__ = ["Coupling", "Tether"]
