"""Wrench — paired force + torque in a single frame.

A `Wrench[Frame]` is the natural type for a part's contribution to its
craft's rigid-body dynamics. The force and torque must share a frame —
typically the craft body frame after the part's static transform has
folded the force-at-offset into a torque about the craft origin.
"""

from __future__ import annotations

from .frames import FrameError, _capture_user_source, _validate_frame
from .types import Vec3


class Wrench:
    """Force + torque pair sharing a frame.

    Construction::

        w = Wrench(force=Vec3[F](...), torque=Vec3[F](...))
        w = Wrench.zero(F)
    """

    __slots__ = ("force", "torque", "_frame")

    def __init__(self, force: Vec3, torque: Vec3) -> None:
        if not isinstance(force, Vec3) or not isinstance(torque, Vec3):
            raise TypeError(
                f"Wrench: force and torque must be Vec3; "
                f"got {type(force).__name__}, {type(torque).__name__}")
        if force._frame is not torque._frame:
            raise FrameError(
                "Wrench",
                expected=f"both components in the same frame "
                         f"({force._frame.__name__})",
                got=f"force={force._frame.__name__}, "
                    f"torque={torque._frame.__name__}",
                source=_capture_user_source(),
            )
        self.force = force
        self.torque = torque
        self._frame = force._frame

    @property
    def frame(self):
        return self._frame

    # --- Factories --------------------------------------------------------

    @classmethod
    def zero(cls, frame) -> "Wrench":
        _validate_frame("Wrench.zero", frame)
        return cls(
            force=Vec3[frame].constant((0.0, 0.0, 0.0)),
            torque=Vec3[frame].constant((0.0, 0.0, 0.0)),
        )

    # --- Operators --------------------------------------------------------

    def __add__(self, other: "Wrench") -> "Wrench":
        if not isinstance(other, Wrench):
            return NotImplemented
        if self._frame is not other._frame:
            raise FrameError(
                "Wrench + Wrench",
                expected=f"same frame ({self._frame.__name__})",
                got=f"{self._frame.__name__} vs {other._frame.__name__}",
                source=_capture_user_source(),
            )
        return Wrench(self.force + other.force, self.torque + other.torque)

    def __neg__(self) -> "Wrench":
        return Wrench(-self.force, -self.torque)

    def __repr__(self) -> str:
        return f"<Wrench[{self._frame.__name__}]>"
