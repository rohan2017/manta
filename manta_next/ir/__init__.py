"""manta_next.ir — IR primitives.

Public surface:
  * Frames        — Frame base + stock frame tags.
  * Types         — Scalar, Vec3[F], Mat3[A,B], Quat[A,B] (and Ori alias).
  * Graph         — context manager + compile / jacobian / summary.
  * Manifold      — SO3, R3, RigidBody for state declarations.
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
from .types import Scalar, Vec3, Mat3, Quat, Ori
from .graph import Graph
from . import manifold

__all__ = [
    "Frame",
    "WorldFrame", "PlanetFrame", "AnchorFrame", "CraftFrame", "PartFrame",
    "FrameError",
    "Scalar", "Vec3", "Mat3", "Quat", "Ori",
    "Graph",
    "manifold",
]
