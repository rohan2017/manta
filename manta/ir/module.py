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

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from hashlib import sha256
from math import prod
from types import MappingProxyType
from typing import Any

import numpy as np


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


def _check_qualified_name(name: str, *, who: str) -> str:
    if not isinstance(name, str) or not name or any(
            not part.isidentifier() for part in name.split(".")):
        raise ValueError(f"{who}: invalid qualified name {name!r}")
    return name


def _finite_owned_array(value: Any, *, size: int, who: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{who}: expected real numeric data, got {raw.dtype}")
    arr = np.asarray(value, dtype=float)
    if arr.size != size:
        raise ValueError(f"{who}: expected {size} value(s), got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{who}: contains non-finite values")
    owned = arr.copy()
    owned.flags.writeable = False
    return owned


def _freeze(value: Any) -> Any:
    """Recursively own mutable metadata payloads."""
    if isinstance(value, np.ndarray):
        owned = value.copy()
        owned.flags.writeable = False
        return owned
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# eq=False: `init` is an ndarray, so a generated __eq__ raises ValueError
# (ambiguous truth) and frozen-dataclass __hash__ raises TypeError — identity
# semantics are what every consumer actually uses.
@dataclass(frozen=True, eq=False)
class StateField:
    """One field of a Module's persistent State.

    * ``kind="manifold"`` — a packed ambient state vector; `spec` carries
      the descriptor (a `StateSpec`: slot names, ambient/tangent layout,
      boxplus). `shape == (ambient_dim,)`.
    * ``kind="matrix"`` — a plain Euclidean tensor (the EKF covariance).

    `init` is the field's default value (a packed ndarray), used to allocate
    fresh State and to render typed-struct defaults.

    `spec` is the flat-layout `StateSpec` descriptor over per-slot manifolds.
    """
    name: str
    kind: str                  # "manifold" | "matrix"
    shape: tuple[int, ...]
    init: Any = None
    spec: Any = None           # StateSpec for kind == "manifold"

    def __post_init__(self) -> None:
        _check_qualified_name(self.name, who="StateField")
        if self.kind not in {"manifold", "matrix"}:
            raise ValueError(
                f"StateField {self.name!r}: unknown kind {self.kind!r}")
        if not self.shape or any(
                not isinstance(n, int) or isinstance(n, bool) or n <= 0
                for n in self.shape):
            raise ValueError(
                f"StateField {self.name!r}: invalid shape {self.shape!r}")
        if self.init is None:
            raise ValueError(f"StateField {self.name!r}: init is required")
        arr = _finite_owned_array(
            self.init, size=int(prod(self.shape)),
            who=f"StateField {self.name!r} init").reshape(self.shape)
        arr.flags.writeable = False
        object.__setattr__(self, "init", arr)
        if self.kind == "manifold":
            if self.spec is None or self.spec.ambient_dim != int(prod(self.shape)):
                raise ValueError(
                    f"StateField {self.name!r}: manifold spec does not match "
                    f"shape {self.shape}")
        elif self.spec is not None:
            raise ValueError(
                f"StateField {self.name!r}: matrix fields cannot carry a spec")


@dataclass(frozen=True)
class StateLayout:
    """The ordered State fields a Module carries (empty for stateless)."""
    fields: tuple[StateField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        names = [f.name for f in self.fields]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"StateLayout: duplicate field name(s) {duplicates}")

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


class ModuleKind(Enum):
    """Explicit runtime capability; semantic behavior is never inferred."""

    KERNEL = "kernel"
    SIMULATOR = "simulator"
    FILTER = "filter"
    RECURRENCE = "recurrence"
    REGULATOR = "regulator"


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
                        LQR's input); `spec` carries the StateSpec and
                        `init` the reference/operating point. May repeat:
                        the primary state is declared first, a reference
                        point after it (an LQR's `x`, `x_ref`).
    * ``OUTPUT``      — a named readout bundle (a recurrence's outputs);
                        `fields` name the components.
    * ``MATRIX``      — a plain matrix value (a covariance `Q`, a Jacobian,
                        an LQR's gain `K`). With an `init` it is a
                        coefficient the caller may replace but need not
                        supply; without one the caller must always pass it.
    * ``PARAMETER``   — the tunable physical-parameter vector `p`; `fields`
                        name each promoted Parameter (with its declared
                        value as default), so a caller that doesn't fit
                        anything can pass the defaults and recover the
                        baked-constant model exactly.
    * ``DIAGNOSTIC``  — a return-only numeric diagnostic. Unlike OUTPUT it
                        is not a structured recurrence readout, and unlike
                        MEASUREMENT it is not a sensor channel that a filter
                        may consume. Filters use it for innovation/NIS data.
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
    DIAGNOSTIC = "diagnostic"


#: Roles a Port may have when it appears as an EntryPoint ARGUMENT.
#: OUTPUT ports are return-only (a recurrence's readout bundle). Every
#: backend's argument dispatch must cover exactly this set — enforced by
#: tests/test_role_dispatch.py, so growing the enum fails there instead
#: of at a runtime fallback.
ARG_ROLES = frozenset(Role) - {Role.OUTPUT, Role.DIAGNOSTIC}


@dataclass(frozen=True)
class PortField:
    """One named sub-component of a structured Port — a Part Input inside
    `u`, a noise channel inside `noise`, a readout inside `y`. `default`
    seeds typed-struct members; `sigma` is the channel's σ for NOISE.

    Naming note: `dim` is this sub-component's width *inside* the port;
    the whole port's flattened width is `Port.size` (the product of
    `Port.shape`). They are deliberately different names because they
    measure different things."""
    name: str
    dim: int
    default: Any = 0.0
    sigma: float | None = None     # NOISE channels: the declared σ
    rate: float | None = None      # CONTROL inputs: declared intake rate (Hz)
    # Contract note emitted next to the field in generated headers (an
    # invariant a direct caller must honour). Documentation only: it is not
    # part of the Module fingerprint.
    doc: str = ""

    def __post_init__(self) -> None:
        _check_qualified_name(self.name, who="PortField")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim <= 0:
            raise ValueError(f"PortField {self.name!r}: dim must be a positive int")
        default = _finite_owned_array(
            self.default, size=(1 if np.asarray(self.default).size == 1 else self.dim),
            who=f"PortField {self.name!r} default")
        object.__setattr__(self, "default",
                           float(default.reshape(-1)[0])
                           if default.size == 1 else default)
        for attr in ("sigma", "rate"):
            value = getattr(self, attr)
            if value is not None:
                arr = _finite_owned_array(value, size=1,
                                          who=f"PortField {self.name!r} {attr}")
                number = float(arr.reshape(-1)[0])
                if number < 0.0 or (attr == "rate" and number == 0.0):
                    raise ValueError(
                        f"PortField {self.name!r}: {attr} must be "
                        f"{'positive' if attr == 'rate' else 'non-negative'}")
                object.__setattr__(self, attr, number)


# eq=False: `init`/`manifold` can hold ndarrays — see StateField.
@dataclass(frozen=True, eq=False)
class Port:
    """A typed value channel of a Module — a runtime input to a method, a
    returned value, or both (a Sim reading that a filter later consumes).

    For role ``STATE``, `spec` holds the value's `StateSpec`."""
    name: str
    role: Role
    shape: tuple[int, ...] = ()
    fields: tuple[PortField, ...] = ()
    spec: Any = None           # StateSpec, for role == STATE
    init: Any = None           # the port's default value: a reference
                               # vector (STATE) or a coefficient an
                               # unsupplied argument falls back to (MATRIX)
    rate: float | None = None  # MEASUREMENT: declared sample rate (Hz)

    def __post_init__(self) -> None:
        _check_qualified_name(self.name, who="Port")
        if not isinstance(self.role, Role):
            raise TypeError(f"Port {self.name!r}: role must be a Role")
        shape = tuple(self.shape)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
               for n in shape):
            raise ValueError(f"Port {self.name!r}: invalid shape {shape!r}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "fields", tuple(self.fields))
        field_names = [f.name for f in self.fields]
        duplicates = sorted({n for n in field_names if field_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Port {self.name!r}: duplicate fields {duplicates}")
        if self.fields and sum(f.dim for f in self.fields) != self.size:
            raise ValueError(
                f"Port {self.name!r}: field dimensions sum to "
                f"{sum(f.dim for f in self.fields)}, port size is {self.size}")
        if self.role is Role.STATE:
            if self.spec is None or self.spec.ambient_dim != self.size:
                raise ValueError(
                    f"Port {self.name!r}: STATE spec does not match size {self.size}")
        elif self.spec is not None:
            raise ValueError(f"Port {self.name!r}: only STATE ports carry a spec")
        if self.init is not None:
            arr = _finite_owned_array(
                self.init, size=self.size, who=f"Port {self.name!r} init")
            arr = arr.reshape(shape) if shape else arr.reshape(())
            arr.flags.writeable = False
            object.__setattr__(self, "init", arr)
        if self.rate is not None:
            arr = _finite_owned_array(self.rate, size=1,
                                      who=f"Port {self.name!r} rate")
            if float(arr.reshape(-1)[0]) <= 0.0:
                raise ValueError(f"Port {self.name!r}: rate must be positive")
            object.__setattr__(self, "rate", float(arr.reshape(-1)[0]))

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

# eq=False: contains Ports/StateFields with ndarray payloads — see
# StateField. Modules compare and hash by identity.
@dataclass(frozen=True, eq=False)
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
    kind: ModuleKind
    hosting: Hosting = Hosting.THREADED
    metadata: Mapping[str, Any] | None = None
    artifact_id: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        check_name(self.name, who="Module")
        if not isinstance(self.kind, ModuleKind):
            raise TypeError(f"Module {self.name!r}: kind must be a ModuleKind")
        if not isinstance(self.hosting, Hosting):
            raise TypeError(f"Module {self.name!r}: hosting must be Hosting")
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "entry_points", tuple(self.entry_points))
        object.__setattr__(self, "functions",
                           MappingProxyType(dict(self.functions)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata or {})))
        port_names = [p.name for p in self.ports]
        duplicate_ports = sorted({n for n in port_names if port_names.count(n) > 1})
        if duplicate_ports:
            raise ValueError(
                f"Module {self.name!r}: duplicate port name(s) {duplicate_ports}")
        methods = [ep.method for ep in self.entry_points]
        duplicate_methods = sorted({n for n in methods if methods.count(n) > 1})
        if duplicate_methods:
            raise ValueError(
                f"Module {self.name!r}: duplicate entry point(s) {duplicate_methods}")
        for ep in self.entry_points:
            self._validate_entry(ep)
        # `entry_ident` (dot → underscore) is not injective over dotted
        # names — `a_b.c` and `a.b_c` both flatten to `a_b_c`, which
        # would alias two entry points / kernel symbols / C++ methods
        # with no compile error. Reject the collision at construction,
        # where the offending names are still visible.
        seen: dict[str, str] = {}
        for ep in self.entry_points:
            flat = entry_ident(ep.method)
            if flat in seen and seen[flat] != ep.method:
                raise ValueError(
                    f"Module {self.name!r}: entry points "
                    f"{seen[flat]!r} and {ep.method!r} both flatten to "
                    f"{flat!r} (dot→underscore) — rename a craft/part so "
                    f"the generated symbols cannot alias.")
            seen[flat] = ep.method
        seen.clear()
        for p in self.ports:
            flat = entry_ident(p.name)
            if flat in seen and seen[flat] != p.name:
                raise ValueError(
                    f"Module {self.name!r}: ports {seen[flat]!r} and "
                    f"{p.name!r} both flatten to {flat!r} "
                    f"(dot→underscore) — rename a craft/part so the "
                    f"generated symbols cannot alias.")
            seen[flat] = p.name
        digest = sha256()
        digest.update(self.name.encode())
        digest.update(self.kind.value.encode())
        digest.update(self.hosting.value.encode())
        for field in self.state.fields:
            spec = (None if field.spec is None else tuple(
                (slot.name, repr(slot.manifold)) for slot in field.spec.slots))
            digest.update(repr((field.name, field.kind, field.shape, spec)).encode())
            digest.update(np.asarray(field.init).tobytes())
        for port in self.ports:
            spec = (None if port.spec is None else tuple(
                (slot.name, repr(slot.manifold)) for slot in port.spec.slots))
            field_contract = tuple(
                (f.name, f.dim, np.asarray(f.default).shape,
                 np.asarray(f.default, dtype=float).tobytes(), f.sigma, f.rate)
                for f in port.fields)
            digest.update(repr((port.name, port.role.value, port.shape,
                                field_contract, spec, port.rate)).encode())
            if port.init is not None:
                digest.update(np.asarray(port.init).tobytes())
        for name, fn in sorted(self.functions.items()):
            digest.update(name.encode())
            digest.update(fn.serialize().encode())
        for ep in self.entry_points:
            digest.update(repr(ep).encode())
        object.__setattr__(self, "artifact_id", digest.hexdigest())

    def _validate_entry(self, ep: EntryPoint) -> None:
        if not isinstance(ep, EntryPoint):
            raise TypeError(f"Module {self.name!r}: invalid entry point {ep!r}")
        _check_qualified_name(ep.method, who=f"Module {self.name!r} entry")
        if ep.fn not in self.functions:
            raise ValueError(
                f"Module {self.name!r}.{ep.method}: missing function {ep.fn!r}")
        if len(set(ep.writes)) != len(ep.writes):
            raise ValueError(f"Module {self.name!r}.{ep.method}: duplicate writes")
        if len(set(ep.returns)) != len(ep.returns):
            raise ValueError(f"Module {self.name!r}.{ep.method}: duplicate returns")
        for ref in ep.args:
            if isinstance(ref, StateRef):
                if ref.name not in self.state:
                    raise ValueError(
                        f"Module {self.name!r}.{ep.method}: unknown state argument "
                        f"{ref.name!r}")
            elif isinstance(ref, PortRef):
                try:
                    port = self.port(ref.name)
                except KeyError as exc:
                    raise ValueError(
                        f"Module {self.name!r}.{ep.method}: unknown port argument "
                        f"{ref.name!r}") from exc
                if port.role not in ARG_ROLES:
                    raise ValueError(
                        f"Module {self.name!r}.{ep.method}: role "
                        f"{port.role.name} cannot be an argument")
            else:
                raise TypeError(
                    f"Module {self.name!r}.{ep.method}: args must be StateRef "
                    f"or PortRef, got {type(ref).__name__}")
        for name in ep.writes:
            if name not in self.state:
                raise ValueError(
                    f"Module {self.name!r}.{ep.method}: unknown state write {name!r}")
        for name in ep.returns:
            try:
                self.port(name)
            except KeyError as exc:
                raise ValueError(
                    f"Module {self.name!r}.{ep.method}: unknown return port {name!r}") \
                    from exc
        fn = self.functions[ep.fn]
        if fn.n_in() != len(ep.args):
            raise ValueError(
                f"Module {self.name!r}.{ep.method}: function takes {fn.n_in()} "
                f"inputs, entry declares {len(ep.args)}")
        expected_out = len(ep.writes) + len(ep.returns)
        if fn.n_out() != expected_out:
            raise ValueError(
                f"Module {self.name!r}.{ep.method}: function returns {fn.n_out()} "
                f"outputs, entry declares {expected_out}")
        for index, ref in enumerate(ep.args):
            expected = (int(prod(self.state.field(ref.name).shape))
                        if isinstance(ref, StateRef) else self.port(ref.name).size)
            if fn.numel_in(index) != expected:
                raise ValueError(
                    f"Module {self.name!r}.{ep.method}: argument {ref.name!r} "
                    f"has {fn.numel_in(index)} values, expected {expected}")
        out_names = (*ep.writes, *ep.returns)
        for index, name in enumerate(out_names):
            expected = (int(prod(self.state.field(name).shape))
                        if index < len(ep.writes) else self.port(name).size)
            if fn.numel_out(index) != expected:
                raise ValueError(
                    f"Module {self.name!r}.{ep.method}: output {name!r} has "
                    f"{fn.numel_out(index)} values, expected {expected}")

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"Module {self.name!r}: no port {name!r}.")

    def ports_by_role(self, role: Role) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is role)

    def sole_port(self, role: Role) -> Port | None:
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

    def _state_source(self, attr: str):
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
        return self._state_source("init")

    @property
    def spec(self):
        """The manifold StateSpec this Module's state/struct is built from —
        a manifold State field's, else a STATE-role port's, else None."""
        return self._state_source("spec")

    def __repr__(self) -> str:
        return (f"<Module {self.name!r} state={list(self.state.names)} "
                f"methods={[e.method for e in self.entry_points]} "
                f"{self.hosting.value}>")
