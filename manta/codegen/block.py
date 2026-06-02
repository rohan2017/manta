"""Block — the contract every lowerable IR satisfies.

A *block* is any IR a `Target` can lower to a runtime: the whole-world
analysis transforms (`Sim`, `EKF`, `LQR`) and the freestanding
signal-processing blocks (PID, Madgwick/Mahony, the IMU strapdown
integrator). A backend dispatches on a block's `RUNTIME_KIND` — a member
of the small closed set below — rather than on its Python type. So adding
a block that reuses an existing runtime shape (e.g. another ``recurrence``
filter) costs **no backend code at all**: the backend already has a
`lower_recurrence` handler, and the new block just declares the kind plus
its own kernels.

Runtime kinds (a backend supports a kind iff it defines ``lower_<kind>``):

  * ``"sim"``        — forward-dynamics tick; state threaded by the caller.
  * ``"ekf"``        — stateful Kalman filter (predict + measurement update).
  * ``"lqr"``        — stateless baked feedback law ``u(x)``.
  * ``"recurrence"`` — stateful manifold recurrence ``x' = f(x, u, dt)``
                       with an optional readout ``y = h(x)``; the shape
                       shared by PID, Madgwick/Mahony, and the IMU
                       strapdown integrator. One generic runtime per
                       backend serves every block of this kind.

The kinds live here as string constants so the set stays closed and
greppable; `RUNTIME_KIND` on an IR class names which one it is. This is
the seam that keeps the backend dispatch open for extension (new kind →
new handler) but closed for modification (existing kinds untouched).
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

# --- The closed set of runtime kinds a Target switches on -------------------
KIND_SIM:        str = "sim"
KIND_EKF:        str = "ekf"
KIND_LQR:        str = "lqr"
KIND_RECURRENCE: str = "recurrence"

ALL_KINDS: tuple[str, ...] = (KIND_SIM, KIND_EKF, KIND_LQR, KIND_RECURRENCE)


@runtime_checkable
class Block(Protocol):
    """Structural type for a lowerable IR.

    The only universal requirement is a `RUNTIME_KIND` class attribute
    naming which backend handler lowers it. The richer per-kind contract
    (e.g. the kernels + state spec + ports a ``recurrence`` block exposes)
    is documented on that kind's base class, not here — different kinds
    carry genuinely different payloads.
    """

    RUNTIME_KIND: ClassVar[str]


def block_kind(obj) -> str | None:
    """Return a block's `RUNTIME_KIND`, or `None` if it isn't a block.

    Reads the class attribute (kinds are per-class, not per-instance), so
    a plain object that merely happens to have an instance attribute named
    `RUNTIME_KIND` is not mistaken for a block.
    """
    return getattr(type(obj), "RUNTIME_KIND", None)
