"""Math primitives — manifold helpers used by state declarations,
the rigid-body integrator, and the EKF's tangent-space machinery.

Public surface:
  * `SO3`        — rotation in SO(3), backed by a unit quaternion.
                   Carries boxplus/boxminus (tangent ↔ ambient).
  * `R3`         — Euclidean 3-space marker manifold (boxplus = +).
  * `RigidBody`  — R3 × SO3 × R3 × R3 composite (the standard 13-D
                   ambient / 12-D tangent rigid-body state).

These don't depend on the `ir.graph` / `ir.frames` infrastructure
beyond `Quat`/`Vec3` — they're pure math on those typed wrappers.
"""

from .manifold import SO3, R3, RigidBody, _so3_exp, _so3_log
from .wrench import Wrench

__all__ = ["SO3", "R3", "RigidBody", "_so3_exp", "_so3_log", "Wrench"]
