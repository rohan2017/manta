"""Frame tags.

A Frame is a static type-marker. It carries no runtime data — its identity is
its class. Frame tags participate in IR type checking: a `Vec3[SceneFrame]`
cannot be cross-producted with a `Vec3[CraftFrame]` without rotating one of
them through a `Quat[SceneFrame, CraftFrame]` first.

Stock frames cover the manta hierarchy `World → Planet → Anchor → Craft →
Part`. Users declaring their own frames just subclass `Frame`:

    class WheelFrame(Frame):
        '''Body frame of a steering wheel; rotates about CraftFrame.+y.'''

Two frames are "the same" iff they are the same Python class (identity).
"""

from __future__ import annotations

import traceback


class Frame:
    """Marker base class. Subclass to declare a new frame tag.

    Frames are never instantiated; they are used purely as types passed to
    IR-value generics like ``Vec3[SomeFrame]``. A frame's identity is its
    class object.
    """

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__doc__:
            cls.__doc__ = f"{cls.__name__} frame tag."

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} is a frame tag (type), not a value — "
            f"do not instantiate it.")


# ----- Stock frames ---------------------------------------------------------

class WorldFrame(Frame):
    """The inertial reference. The integrator advances state in this frame."""

class PlanetFrame(Frame):
    """A planet's body frame (rotates with the planet). Optional layer."""

class AnchorFrame(Frame):
    """A floating-origin coordinate root for a coupling component. One per
    connected (craft + coupling) group at compile time. Replaces the
    legacy `SceneFrame`. Use this as the parent for craft-local frames."""

# Backwards-compatible alias: legacy manta used SceneFrame for this role.
# Kept so M0 example code reads naturally; AnchorFrame is preferred going
# forward.
SceneFrame = AnchorFrame

class CraftFrame(Frame):
    """A single craft's body frame. Rotates and translates relative to its
    AnchorFrame."""

class PartFrame(Frame):
    """A part's local frame, rigidly attached to its parent (which may be
    CraftFrame or another PartFrame via a static or joint transform)."""


# ----- Error type -----------------------------------------------------------

class FrameError(TypeError):
    """Raised when an IR op's operands have incompatible frame tags.

    Carries the operation name, the offending frames, and a source-location
    string captured at op-build time so the message points at the user's
    `.py:line` (not at the manta IR internals).
    """

    def __init__(self, op: str, *, expected: str, got: str,
                 source: str | None = None):
        msg = f"{op}: frame mismatch — expected {expected}, got {got}"
        if source:
            msg += f"\n  at {source}"
        super().__init__(msg)
        self.op = op
        self.expected = expected
        self.got = got
        self.source = source


def _capture_user_source() -> str | None:
    """Walk the stack and return the topmost frame outside of manta_next.
    Used to attach a user-code source location to errors raised inside IR
    ops. Returns `None` if no frame outside manta_next is found."""
    stack = traceback.extract_stack()
    # Walk from the caller end (most recent) inward, skipping any frame
    # whose file path contains 'manta_next/ir/'.
    for entry in reversed(stack):
        if "manta_next/ir/" in entry.filename:
            continue
        if "manta_next\\ir\\" in entry.filename:    # windows
            continue
        return f"{entry.filename}:{entry.lineno}  in {entry.name}"
    return None


def _check_frame(name: str, *, expected, got, op: str) -> None:
    """Raise a FrameError if `got` is not the same class as `expected`.

    Both `expected` and `got` should be subclasses of Frame. Identity is
    by class equality (Python `is`).
    """
    if expected is got:
        return
    raise FrameError(
        op,
        expected=f"{name}={expected.__name__}",
        got=f"{name}={got.__name__}",
        source=_capture_user_source(),
    )


def _format_frame(frame) -> str:
    if frame is None:
        return "<unframed>"
    if isinstance(frame, type) and issubclass(frame, Frame):
        return frame.__name__
    return repr(frame)


def _is_frame(f) -> bool:
    return isinstance(f, type) and issubclass(f, Frame)


def _validate_frame(name: str, f) -> None:
    """Raise TypeError unless `f` is a Frame subclass."""
    if not _is_frame(f):
        raise TypeError(
            f"{name}: frame argument must be a Frame subclass, got {f!r}")
