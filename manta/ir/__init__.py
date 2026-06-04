"""manta.ir — IR primitives.

Public surface:
  * Frames        — Frame base + stock frame tags.
  * Types         — Scalar, Vec3[F], Mat3[A,B], Quat[A,B].
  * Manifolds     — Manifold + ScalarManifold / R3Manifold / SO3Manifold
                    (metadata + boxplus/boxminus on top of types).
  * StateSpec     — flat-vector state layout (slots over manifolds),
                    with SlotSet (POSE / TWIST / ALL) aliases.
  * Wrench        — frame-tagged (force, torque) pair.
  * Graph         — context manager + compile / jacobian / summary.
  * FrameError    — raised on frame-tag mismatch with source-location info.
"""

from .frames import (
    Frame,
    WorldFrame,
    PlanetFrame,
    CraftFrame,
    PartFrame,
    FrameError,
)
from .types import Scalar, Vec3, Mat3, Quat
from .manifold import Manifold, ScalarManifold, R3Manifold, SO3Manifold
from .state_spec import (
    ALL, POSE, TWIST, SlotSet, StateSlot, StateSpec, resolve_slotset,
)
from .wrench import Wrench
from .graph import Graph

__all__ = [
    "Frame",
    "WorldFrame", "PlanetFrame", "CraftFrame", "PartFrame",
    "FrameError",
    "Scalar", "Vec3", "Mat3", "Quat",
    "Manifold", "ScalarManifold", "R3Manifold", "SO3Manifold",
    "StateSpec", "StateSlot", "SlotSet", "POSE", "TWIST", "ALL",
    "resolve_slotset",
    "Wrench",
    "Graph",
]
