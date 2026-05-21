"""manta_next.ir — IR primitives.

Public surface:
  * Frames        — Frame base + stock frame tags.
  * Types         — Scalar, Vec3[F], Mat3[A,B], Quat[A,B].
  * Graph         — context manager + compile / jacobian / summary.
  * FrameError    — raised on frame-tag mismatch with source-location info.

Manifold helpers (SO3, R3, RigidBody) live in `manta_next.math.manifold`.
"""

from .frames import (
    Frame,
    WorldFrame,
    PlanetFrame,
    AnchorFrame,
    CraftFrame,
    PartFrame,
    FrameError,
)
from .types import Scalar, Vec3, Mat3, Quat
from .graph import Graph

__all__ = [
    "Frame",
    "WorldFrame", "PlanetFrame", "AnchorFrame", "CraftFrame", "PartFrame",
    "FrameError",
    "Scalar", "Vec3", "Mat3", "Quat",
    "Graph",
]
