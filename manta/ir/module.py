"""Module — the single backend-neutral IR every transform lowers to.

A **Module** is the whole ontology in one object: a typed **State** plus a
set of named **Functions**, exposed through typed **EntryPoint** methods.
`Sim`, `EKF`, `LQR`, and every recurrence filter are just Modules with
different state + entry points, so a backend implements exactly one generic
`lower_module(Module)` (plus a way to run a `ca.Function`) and gets every
feature for free.

The IR is *typed*: every kernel argument is a `StateRef` or a `PortRef` —
nothing else — and every Port carries a `Role`. Backends dispatch on those
types/roles; there are no name conventions ("dt", "z_…") and no lowering
flags. A transform builds *honest kernels* (each kernel takes exactly the
arguments it uses; specializations like a dt-free measurement or an
auto-process-noise predict are baked symbolically at construction), so the
entry set handed to a backend is final — lowering just lowers.

This module is pure data — it imports nothing heavy and knows nothing about
any backend or transform. Each transform's `.module()` produces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from types import MappingProxyType
from typing import Any, Mapping


def entry_ident(name: str) -> str:
    """A dotted manta name (`craft.part.out`) → the flat identifier used in
    entry-point/kernel names and C++ symbols. The single home of the
    dot→underscore convention: transforms name their entries with it and
    runtimes/backends look entries up with it, so they can never drift."""
    return name.replace(".", "_")


def check_name(name: str, *, who: str) -> str:
    """Require a model-object name to be a valid identifier.

    Craft/Part/Disturbance/World names flow into tick-signature dot-names,
    Module entry-point names, and (via `entry_ident`) C++ symbols — a name
    like `"imu-9dof"` would only fail when the *generated* code compiles,
    far from the mistake. Refuse it at construction instead."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"{who}: name {name!r} is not a valid identifier. Names appear "
            f"in generated kernel names and C++ symbols — use letters, "
            f"digits, and underscores only, not starting with a digit.")
    return name


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateField:
    """One field of a Module's persistent State.

    * ``kind="manifold"`` — a packed ambient state vector; `manifold` carries
      the descriptor (a `StateSpec`: slot names, ambient/tangent layout,
      boxplus). `shape == (ambient_dim,)`.
    * ``kind="matrix"`` — a plain Euclidean tensor (the EKF covariance).

    `init` is the field's default value (a packed ndarray), used to allocate
    fresh State and to render typed-struct defaults.
    """
    name: str
    kind: str                  # "manifold" | "matrix"
    shape: tuple[int, ...]
    init: Any = None
    manifold: Any = None       # StateSpec for kind == "manifold"


@dataclass(frozen=True)
class StateLayout:
    """The ordered State fields a Module carries (empty for stateless)."""
    fields: tuple[StateField, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> StateField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"StateLayout: no field {name!r} (have {self.names}).")

    def __contains__(self, name: str) -> bool:
        return any(f.name == name for f in self.fields)


class Hosting(Enum):
    """How a runtime hosts the State.

    * ``HELD``     — the runtime owns mutable state members; entry points
                     read and write them in place (EKF, recurrence).
    * ``THREADED`` — state is passed in and returned by the caller; the
                     runtime holds nothing (Sim, and trivially stateless
                     blocks like LQR).
    """
    HELD = "held"
    THREADED = "threaded"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

class Role(Enum):
    """What a Port *is* — backends type parameters/returns from this.

    * ``CONTROL``     — the control vector `u`; `fields` name each Part
                        Input (with defaults) so typed backends emit an
                        Inputs struct.
    * ``MEASUREMENT`` — one sensor channel: produced by a Sim entry
                        (a reading) and/or consumed by a filter update (`z`).
    * ``NOISE``       — the stochastic draw vector; `fields` name each
                        channel (with its σ) so a driver can bind to it.
    * ``TIME``        — world-clock time `t` (defaults to 0 at call sites).
    * ``TIMESTEP``    — the integration step `dt` (always required).
    * ``STATE``       — a full ambient manifold state passed as data (the
                        LQR's input); `manifold` carries the StateSpec and
                        `init` the reference/operating point. May repeat:
                        the primary state is declared first, a reference
                        point after it (an LQR's `x`, `x_ref`).
    * ``OUTPUT``      — a named readout bundle (a recurrence's outputs);
                        `fields` name the components.
    * ``MATRIX``      — a plain matrix value (a covariance `Q`, a Jacobian).
    * ``PARAMETER``   — the tunable physical-parameter vector `p`; `fields`
                        name each promoted Parameter (with its declared
                        value as default), so a caller that doesn't fit
                        anything can pass the defaults and recover the
                        baked-constant model exactly.
    """
    CONTROL = "control"
    MEASUREMENT = "measurement"
    NOISE = "noise"
    TIME = "time"
    TIMESTEP = "timestep"
    STATE = "state"
    OUTPUT = "output"
    MATRIX = "matrix"
    PARAMETER = "parameter"


#: Roles a Port may have when it appears as an EntryPoint ARGUMENT.
#: OUTPUT ports are return-only (a recurrence's readout bundle). Every
#: backend's argument dispatch must cover exactly this set — enforced by
#: tests/test_role_dispatch.py, so growing the enum fails there instead
#: of at a runtime fallback.
ARG_ROLES = frozenset(Role) - {Role.OUTPUT}


@dataclass(frozen=True)
class PortField:
    """One named sub-component of a structured Port — a Part Input inside
    `u`, a noise channel inside `noise`, a readout inside `y`. `default`
    seeds typed-struct members; `sigma` is the channel's σ for NOISE."""
    name: str
    dim: int
    default: Any = 0.0
    sigma: float | None = None     # NOISE channels: the declared σ
    rate: float | None = None      # CONTROL inputs: declared intake rate (Hz)


@dataclass(frozen=True)
class Port:
    """A typed value channel of a Module — a runtime input to a method, a
    returned value, or both (a Sim reading that a filter later consumes)."""
    name: str
    role: Role
    shape: tuple[int, ...] = ()
    fields: tuple[PortField, ...] = ()
    manifold: Any = None       # StateSpec, for role == STATE
    init: Any = None           # reference vector, for role == STATE
    rate: float | None = None  # MEASUREMENT: declared sample rate (Hz)

    @property
    def size(self) -> int:
        return int(prod(self.shape)) if self.shape else 1


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateRef:
    """A kernel argument that is a State field (by name)."""
    name: str


@dataclass(frozen=True)
class PortRef:
    """A kernel argument that is a Port value supplied per call (by name)."""
    name: str


@dataclass(frozen=True)
class EntryPoint:
    """One typed method backed by exactly one `ca.Function`.

    `args` lays out the kernel's positional arguments — each a `StateRef`
    or a `PortRef`, nothing else. The kernel's outputs are consumed in
    order: the first `len(writes)` update those State fields; the remaining
    outputs are returned under the Ports named in `returns`.
    """
    method: str
    fn: str                                   # key into Module.functions
    args: tuple[Any, ...] = ()                # StateRef | PortRef
    writes: tuple[str, ...] = ()              # State field names
    returns: tuple[str, ...] = ()             # Port names


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Module:
    """A stateful unit = State + named Functions + typed methods.

    `functions` is the only thing a backend translates; `entry_points`
    describe the methods to expose over them — a backend lowers ALL of
    them, unconditionally.

    Immutable like every other IR dataclass: `functions` is wrapped in a
    read-only mapping view at construction.
    """
    name: str
    state: StateLayout
    ports: tuple[Port, ...]
    functions: Mapping[str, Any]              # {name: ca.Function}
    entry_points: tuple[EntryPoint, ...]
    hosting: Hosting = Hosting.THREADED

    def __post_init__(self) -> None:
        object.__setattr__(self, "functions",
                           MappingProxyType(dict(self.functions)))

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"Module {self.name!r}: no port {name!r}.")

    def ports_by_role(self, role: Role) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is role)

    def sole_port(self, role: Role) -> "Port | None":
        """The single Port of `role`, or None when there is none. Raises on
        duplicates rather than silently picking one (per-channel roles like
        MEASUREMENT go through `ports_by_role`) — except STATE, where
        repetition is part of the IR contract: the primary state is declared
        first and a reference point may follow (an LQR's `x`, `x_ref`).
        Backends key on the primary and treat the rest as data arguments."""
        ps = self.ports_by_role(role)
        if len(ps) > 1 and role is not Role.STATE:
            raise ValueError(
                f"Module {self.name!r}: {len(ps)} ports of role {role.name}; "
                f"sole_port expects at most one — use ports_by_role.")
        return ps[0] if ps else None

    def returns_role(self, role: Role) -> bool:
        """True if any entry point returns a Port of `role` (a structural
        query backends classify Module shape with — e.g. an LQR returns a
        CONTROL port, an oracle returns MEASUREMENTs)."""
        return any(self.port(name).role is role
                   for ep in self.entry_points for name in ep.returns)

    def entry(self, method: str) -> EntryPoint:
        for ep in self.entry_points:
            if ep.method == method:
                return ep
        raise KeyError(f"Module {self.name!r}: no entry point {method!r}.")

    @property
    def matrix_fields(self) -> tuple[StateField, ...]:
        """The Euclidean matrix State fields (an EKF's held covariance `P`)."""
        return tuple(f for f in self.state.fields if f.kind == "matrix")

    def _manifold_source(self, attr: str):
        """First non-None `attr` off a manifold State field, else off a
        STATE-role port — the shared scan behind `initial_x` and `spec`."""
        for f in self.state.fields:
            if f.kind == "manifold" and getattr(f, attr) is not None:
                return getattr(f, attr)
        for p in self.ports:
            if p.role is Role.STATE and getattr(p, attr) is not None:
                return getattr(p, attr)
        return None

    @property
    def initial_x(self):
        """The manifold state's packed init vector — a manifold State field's
        `init`, else a STATE-role port's `init`, else None. The reference
        point `spec` is built around; backends seed struct defaults from it."""
        return self._manifold_source("init")

    @property
    def spec(self):
        """The manifold StateSpec this Module's state/struct is built from —
        a manifold State field's, else a STATE-role port's, else None."""
        return self._manifold_source("manifold")

    def __repr__(self) -> str:
        return (f"<Module {self.name!r} state={list(self.state.names)} "
                f"methods={[e.method for e in self.entry_points]} "
                f"{self.hosting.value}>")
