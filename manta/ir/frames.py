"""Frame tags.

A Frame is a static type-marker. It carries no runtime data — its identity is
its class. Frame tags participate in IR type checking: a `Vec3[WorldFrame]`
cannot be cross-producted with a `Vec3[CraftFrame]` without rotating one of
them through a `Quat[WorldFrame, CraftFrame]` first.

Stock frames cover the manta hierarchy `World → Planet → Craft → Part`.
Users declaring their own frames just subclass `Frame`:

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
    """The inertial reference. The integrator writes Newton-Euler in this
    frame — `m·a = F`, no pseudo-forces. Craft state (position, velocity,
    orientation) lives here. Choose it inertial: e.g., for a drone above
    a launch pad, treat WorldFrame as Earth-Centered Inertial (or as a
    locally non-rotating tangent plane if you're OK with ignoring Earth's
    rotation at the timescales of interest)."""


class PlanetFrame(Frame):
    """A planet's body-fixed frame — rotates with the planet, parameterized
    by a transform from `WorldFrame`. Surface-fixed objects (launch pads,
    ground geometry, atmospheric models) are naturally authored here.
    Coriolis / centrifugal effects emerge automatically when something
    expressed in PlanetFrame is read in WorldFrame; the integrator
    itself stays inertial."""


class CraftFrame(Frame):
    """A single craft's body frame. Rotates and translates relative to
    WorldFrame."""

class PartFrame(Frame):
    """A part's own body frame — origin at the part's mount point, axes
    fixed to the part. Equals CraftFrame for a part mounted on the craft
    root. Inside a `Part.update()` it denotes *this* part's frame; the
    framework maps it to CraftFrame (and on to WorldFrame) afterwards.
    Vectors a part authors natively (thrust, sensor axes) live here."""


class ParentFrame(Frame):
    """The frame of a part's immediate parent (the parent's PartFrame).
    For a part on the craft root this is CraftFrame; for a part on a
    joint's rotor it is the joint's own frame. Used by the relative-motion
    accessors (`ctx.velocity[ParentFrame]` = how the part moves relative
    to its parent)."""


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
    """Walk the stack and return the topmost frame outside of manta.
    Used to attach a user-code source location to errors raised inside IR
    ops. Returns `None` if no frame outside manta is found."""
    stack = traceback.extract_stack()
    # Walk from the caller end (most recent) inward, skipping any frame
    # whose file path contains 'manta/ir/'.
    for entry in reversed(stack):
        if "manta/ir/" in entry.filename:
            continue
        if "manta\\ir\\" in entry.filename:    # windows
            continue
        return f"{entry.filename}:{entry.lineno}  in {entry.name}"
    return None


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
