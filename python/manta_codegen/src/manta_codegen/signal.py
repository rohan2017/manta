"""Signal — declarative metadata for a value the user can publish/subscribe.

Each PartDescriptor declares its bindable signals as a class-level table:

    class IMU(PartDescriptor):
        cpp_class = "manta::parts::IMU"
        signals = [
            vec3_out_signal("last_accel", "last_accel"),
            vec3_out_signal("last_gyro",  "last_gyro"),
        ]

The PartDescriptor base class' __init__ walks `signals` and attaches each
one as a BoundSignal attribute on the instance:

    imu = IMU("imu")
    imu.last_accel    # → BoundSignal handle the user passes to c.publish(...)

The signals table is the single authoritative description of what data a
part exposes — codegen consults it to emit publishers/subscribers, and a
sync test (tests/test_codegen_signals — phase 2e) compiles each signal's
C++ expression to verify the part's actual API matches.

Not a wire format. Signals only carry enough information for codegen to
emit the right read/write call into the user's C++; the actual wire
encoding (JSON today, binary later) is the emitter's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import PartDescriptor


@dataclass(frozen=True)
class Signal:
    """Class-level descriptor of a publishable/subscribable value on a Part.

    Args:
        name: Python attribute name (e.g. "last_accel"). Becomes an attribute
              on each PartDescriptor instance after construction.
        direction: "out" = the part exposes a value the user publishes;
                   "in"  = the part accepts a value the user subscribes-into.
        n_floats: Wire-format width — number of float scalars in the payload.
        cpp_read_exprs: For "out" signals, a tuple of n_floats C++ expressions,
                        each yielding one float component. The placeholder
                        `{accessor}` is substituted with the part accessor at
                        emit time (e.g. "craft.imu()"). Empty for "in" signals.
        cpp_write_stmt: For "in" signals, a single C++ statement to apply the
                        decoded payload. Placeholders: `{accessor}` and
                        `{v0}`..`{v{n_floats-1}}`. Empty for "out" signals.
    """
    name: str
    direction: str   # "out" | "in"
    n_floats: int
    cpp_read_exprs: tuple[str, ...] = ()
    cpp_write_stmt: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("out", "in"):
            raise ValueError(
                f"Signal {self.name!r}: direction must be 'out' or 'in', got "
                f"{self.direction!r}")
        if self.n_floats < 1:
            raise ValueError(
                f"Signal {self.name!r}: n_floats must be >= 1, got {self.n_floats}")
        if self.direction == "out":
            if len(self.cpp_read_exprs) != self.n_floats:
                raise ValueError(
                    f"Signal {self.name!r}: out signals need exactly n_floats "
                    f"({self.n_floats}) cpp_read_exprs, got {len(self.cpp_read_exprs)}")
            if self.cpp_write_stmt:
                raise ValueError(
                    f"Signal {self.name!r}: out signals must not set cpp_write_stmt")
        else:
            if not self.cpp_write_stmt:
                raise ValueError(
                    f"Signal {self.name!r}: in signals must set cpp_write_stmt")
            if self.cpp_read_exprs:
                raise ValueError(
                    f"Signal {self.name!r}: in signals must not set cpp_read_exprs")


@dataclass(frozen=True)
class BoundSignal:
    """A per-instance handle returned by `partdesc.<signal_name>` access. Carries
    enough information for codegen to emit the right C++ call without going
    back to the part descriptor — protocol emitters consume BoundSignal,
    not Signal."""
    part_name: str       # e.g. "imu" — the Part's name within its Craft
    signal: Signal       # class-level metadata

    @property
    def name(self) -> str:
        return self.signal.name

    @property
    def direction(self) -> str:
        return self.signal.direction


# ---------------------------------------------------------------------------
# Helpers for common shapes

def scalar_out_signal(name: str, getter_method: str) -> Signal:
    """Single-float output: `{accessor}.<getter_method>()`."""
    return Signal(
        name=name,
        direction="out",
        n_floats=1,
        cpp_read_exprs=(f"{{accessor}}.{getter_method}()",),
    )


def vec3_out_signal(name: str, getter_method: str) -> Signal:
    """3-float output reading a Vec3<...>: `{accessor}.<getter_method>().raw()(i)`."""
    return Signal(
        name=name,
        direction="out",
        n_floats=3,
        cpp_read_exprs=tuple(
            f"{{accessor}}.{getter_method}().raw()({i})" for i in range(3)),
    )


def scalar_in_signal(name: str, setter_method: str) -> Signal:
    """Single-float input: calls `{accessor}.<setter_method>(v0)`."""
    return Signal(
        name=name,
        direction="in",
        n_floats=1,
        cpp_write_stmt=f"{{accessor}}.{setter_method}({{v0}});",
    )
