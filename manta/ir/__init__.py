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
    CraftFrame,
    Frame,
    FrameError,
    ParentFrame,
    PartFrame,
    PlanetFrame,
    WorldFrame,
)
from .graph import Graph
from .manifold import (
    Manifold,
    R3Manifold,
    RnManifold,
    ScalarManifold,
    SO3Manifold,
    manifold_from_shortcut,
)
from .module import (
    EntryPoint,
    Hosting,
    Module,
    ModuleKind,
    Port,
    PortField,
    PortRef,
    Role,
    StateField,
    StateLayout,
    StateRef,
    entry_ident,
)
from .state_spec import (
    ALL,
    POSE,
    TWIST,
    SlotSet,
    StateSlot,
    StateSpec,
    resolve_slotset,
)
from .types import Mat3, Quat, Scalar, Vec3, VecN
from .wrench import Wrench

__all__ = [
    "ALL",
    "POSE",
    "TWIST",
    "CraftFrame",
    "EntryPoint",
    "Frame",
    "FrameError",
    "Graph",
    "Hosting",
    "Manifold",
    "Mat3",
    "Module",
    "ModuleKind",
    "ParentFrame",
    "PartFrame",
    "PlanetFrame",
    "Port",
    "PortField",
    "PortRef",
    "Quat",
    "R3Manifold",
    "RnManifold",
    "Role",
    "SO3Manifold",
    "Scalar",
    "ScalarManifold",
    "SlotSet",
    "StateField",
    "StateLayout",
    "StateRef",
    "StateSlot",
    "StateSpec",
    "Vec3",
    "VecN",
    "WorldFrame",
    "Wrench",
    "entry_ident",
    "manifold_from_shortcut",
    "resolve_slotset",
]
