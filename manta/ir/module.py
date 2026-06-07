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
from typing import Any


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

    @property
    def size(self) -> int:
        return int(prod(self.shape)) if self.shape else 1


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
                        `init` the reference/operating point.
    * ``OUTPUT``      — a named readout bundle (a recurrence's outputs);
                        `fields` name the components.
    * ``MATRIX``      — a plain matrix value (a covariance `Q`, a Jacobian).
    """
    CONTROL = "control"
    MEASUREMENT = "measurement"
    NOISE = "noise"
    TIME = "time"
    TIMESTEP = "timestep"
    STATE = "state"
    OUTPUT = "output"
    MATRIX = "matrix"


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


KernelArg = "StateRef | PortRef"


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

@dataclass
class Module:
    """A stateful unit = State + named Functions + typed methods.

    `functions` is the only thing a backend translates; `entry_points`
    describe the methods to expose over them — a backend lowers ALL of
    them, unconditionally.
    """
    name: str
    state: StateLayout
    ports: tuple[Port, ...]
    functions: dict[str, Any]                 # {name: ca.Function}
    entry_points: tuple[EntryPoint, ...]
    hosting: Hosting = Hosting.THREADED

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"Module {self.name!r}: no port {name!r}.")

    def ports_by_role(self, role: Role) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is role)

    def entry(self, method: str) -> EntryPoint:
        for ep in self.entry_points:
            if ep.method == method:
                return ep
        raise KeyError(f"Module {self.name!r}: no entry point {method!r}.")

    @property
    def spec(self):
        """The manifold StateSpec this Module's state/struct is built from —
        a manifold State field's, else a STATE-role port's, else None."""
        for f in self.state.fields:
            if f.kind == "manifold":
                return f.manifold
        for p in self.ports:
            if p.role is Role.STATE and p.manifold is not None:
                return p.manifold
        return None

    def __repr__(self) -> str:
        return (f"<Module {self.name!r} state={list(self.state.names)} "
                f"methods={[e.method for e in self.entry_points]} "
                f"{self.hosting.value}>")
