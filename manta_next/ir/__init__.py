"""manta_next.ir — IR primitives.

Public surface:
  * Frames        — Frame base + stock frame tags.
  * Types         — Scalar, Vec3[F], Mat3[A,B], Quat[A,B].
  * Manifolds     — SO3, R3, RigidBody (boxplus/boxminus on top of types).
  * Wrench        — frame-tagged (force, torque) pair.
  * Graph         — context manager + compile / jacobian / summary.
  * FrameError    — raised on frame-tag mismatch with source-location info.
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
from .manifold import SO3, R3, RigidBody
from .wrench import Wrench
from .graph import Graph

__all__ = [
    "Frame",
    "WorldFrame", "PlanetFrame", "AnchorFrame", "CraftFrame", "PartFrame",
    "FrameError",
    "Scalar", "Vec3", "Mat3", "Quat",
    "SO3", "R3", "RigidBody",
    "Wrench",
    "Graph",
]
