"""manta.ir — IR primitives.

Public surface:
  * Frames        — Frame base + stock frame tags.
  * Types         — Scalar, Vec3[F], Mat3[A,B], Quat[A,B], VecN[n].
  * Manifolds     — Manifold + ScalarManifold / R3Manifold / SO3Manifold /
                    RnManifold (metadata + boxplus/boxminus on top of
                    types), plus `manifold_from_shortcut` for the
                    `"R<n>"` string grammar that Noise/Parameter take.
  * StateSpec     — flat-vector state layout (slots over manifolds),
                    with SlotSet (POSE / TWIST / ALL) aliases.
  * Wrench        — frame-tagged (force, torque) pair.
  * Graph         — context manager + compile / jacobian / summary.
  * FrameError    — raised on frame-tag mismatch with source-location info.
  * Module        — the typed transform↔backend contract (`Module`,
                    `StateLayout`/`StateField`, `Port`/`PortField`, `Role`,
                    `EntryPoint`, `StateRef`/`PortRef`, `Hosting`,
                    `entry_ident`) — every transform emits one and every
                    target lowers one; a custom backend imports it from
                    here.
"""

from .frames import (
    Frame,
    WorldFrame,
    PlanetFrame,
    CraftFrame,
    PartFrame,
    ParentFrame,
    FrameError,
)
from .types import Scalar, Vec3, Mat3, Quat, VecN
from .manifold import (
    Manifold, ScalarManifold, R3Manifold, RnManifold, SO3Manifold,
    manifold_from_shortcut,
)
from .state_spec import (
    ALL, POSE, TWIST, SlotSet, StateSlot, StateSpec, resolve_slotset,
)
from .wrench import Wrench
from .graph import Graph
from .module import (
    EntryPoint, Hosting, Module, ModuleKind, Port, PortField, PortRef, Role,
    StateField, StateLayout, StateRef, entry_ident,
)

__all__ = [
    "Frame",
    "WorldFrame", "PlanetFrame", "CraftFrame", "PartFrame", "ParentFrame",
    "FrameError",
    "Scalar", "Vec3", "Mat3", "Quat", "VecN",
    "Manifold", "ScalarManifold", "R3Manifold", "RnManifold",
    "SO3Manifold", "manifold_from_shortcut",
    "StateSpec", "StateSlot", "SlotSet", "POSE", "TWIST", "ALL",
    "resolve_slotset",
    "Wrench",
    "Graph",
    "Module", "ModuleKind", "StateLayout", "StateField", "Port", "PortField", "Role",
    "EntryPoint", "StateRef", "PortRef", "Hosting", "entry_ident",
]
