"""Module — the single backend-neutral IR every transform lowers to.

A **Module** is the whole ontology in one object: a typed **State** plus a
set of named **Functions**, exposed through typed **EntryPoint** methods.
`Sim`, `EKF`, `LQR`, and every recurrence filter (PID, Madgwick, …) are all
just Modules with different state + entry points — so a backend implements
exactly one generic `lower_module(Module)` (plus a way to run a
`ca.Function`) and gets every feature for free, with no feature-specific
backend code. This generalizes the old `EvaluatorSpec` along three axes:

  * **State** can hold plain tensors (an EKF's covariance `P`), not only
    manifold slots;
  * an **EntryPoint** can take extra *runtime* inputs (a measurement `z`)
    via named **Port**s, not only `state/inputs/dt/t`;
  * an EntryPoint can write *multiple* State fields (`x` and `P`).

This module is pure data — it imports nothing heavy and knows nothing about
any backend. The per-backend `lower_module` reads it; the per-transform
builders (`manta.codegen.module_build`) produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any


@dataclass(frozen=True)
class StateField:
    """One field of a Module's persistent State.

    `kind` selects how a backend stores + (if needed) corrects the field:

      * ``"manifold"`` — a packed ambient state vector whose tangent
        corrections are manifold-aware. `manifold` carries the descriptor
        (a `StateSpec`, or any object with `boxplus_num`); `shape` is the
        ambient shape, e.g. `(ambient_dim,)`.
      * ``"matrix"`` / ``"vector"`` / ``"scalar"`` — a plain Euclidean
        tensor (covariance, integral term, …); ambient == tangent.

    `init` is the field's default value (ndarray / scalar / identity),
    used to allocate fresh State.
    """
    name: str
    kind: str
    shape: tuple[int, ...]
    init: Any = None
    manifold: Any = None      # descriptor for kind=="manifold" (e.g. StateSpec)

    @property
    def size(self) -> int:
        """Flat element count."""
        return int(prod(self.shape)) if self.shape else 1


@dataclass(frozen=True)
class StateLayout:
    """The ordered set of State fields a Module carries.

    Subsumes the manifold-only `StateSpec` (a StateSpec becomes one
    `kind="manifold"` field here). A Module with no persistent state (e.g.
    `LQR`) has an empty layout.
    """
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


@dataclass(frozen=True)
class PortField:
    """One named sub-component of a structured Port — e.g. one Part Input
    inside the `u` control vector, or one named readout inside a recurrence
    block's output. Carries the `default` so a typed backend (C++) can build
    the struct without reaching back to the model."""
    name: str
    dim: int
    default: Any = 0.0


@dataclass(frozen=True)
class Port:
    """A runtime input to a method — supplied per call, never stored in
    State (`u`, `z`, `dt`, `t`, `Q`, …). A scalar port has `shape == ()`.

    `fields` optionally names the port's flat sub-components in order (the
    individual Part Inputs in `u`, the named readouts in a recurrence's
    output), so a typed backend can emit a named struct + flat pack/unpack;
    an unstructured port (a bare `z`, `dt`, `Q`) leaves it empty.
    """
    name: str
    shape: tuple[int, ...] = ()
    fields: tuple[PortField, ...] = ()
    manifold: Any = None     # StateSpec/Manifold when the port is a manifold
                             # state vector (e.g. LQR's `x` input) — lets a
                             # typed backend build the same struct as State.

    @property
    def size(self) -> int:
        return int(prod(self.shape)) if self.shape else 1


@dataclass(frozen=True)
class EntryPoint:
    """One typed method backed by exactly one `ca.Function`.

    `kernel_args` is the ordered list of names — each a State field or a
    Port — laid out as the kernel's positional arguments. The kernel's
    outputs are consumed in order: the first `len(writes_state)` outputs
    update those State fields; the remaining outputs are returned to the
    caller under the names in `returns`.

    The remaining fields are lowering hints a typed backend uses and the
    numpy runtime ignores (it supplies every kernel arg or defaults dt/t):

      * `deploy`         — part of the deployable surface? `step` (the
                           driven-oracle Sim variant) sets False so a deploy
                           backend lowers `predict` instead.
      * `synthesize_zero`— kernel args the method does NOT expose, filled
                           with zero (a measurement is dt-independent → `dt`).
      * `port_from`      — `{port: entry}`: a port the method computes
                           internally by calling another entry's kernel
                           (the EKF's `Q` from `process_noise`), rather than
                           taking it as a parameter.
    """
    method: str
    fn: str                                  # key into Module.functions
    kernel_args: tuple[str, ...] = ()
    writes_state: tuple[str, ...] = ()
    returns: tuple[str, ...] = ()
    deploy: bool = True
    synthesize_zero: tuple[str, ...] = ()
    port_from: dict[str, str] = field(default_factory=dict)


@dataclass
class Module:
    """A stateful unit = State + named Functions + typed methods.

    `functions` is the only thing a backend translates; `entry_points`
    describe the methods to expose over them. `analysis` holds extra
    `ca.Function`s surfaced for IR-level analysis (e.g. the EKF's `F`/`H`
    for observability) but NOT lowered as runtime methods.
    """
    name: str
    state: StateLayout
    ports: tuple[Port, ...]
    functions: dict[str, Any]                # {name: ca.Function}
    entry_points: tuple[EntryPoint, ...]
    analysis: dict[str, Any] = field(default_factory=dict)
    held: bool = False    # State lives as runtime members (EKF/recurrence)
                          # vs. threaded by the caller / stateless (Sim/LQR).

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"Module {self.name!r}: no port {name!r}.")

    def entry(self, method: str) -> EntryPoint:
        for ep in self.entry_points:
            if ep.method == method:
                return ep
        raise KeyError(f"Module {self.name!r}: no entry point {method!r}.")

    def __repr__(self) -> str:
        return (f"<Module {self.name!r} state={list(self.state.names)} "
                f"methods={[e.method for e in self.entry_points]}>")
