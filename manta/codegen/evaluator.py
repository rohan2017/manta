"""EvaluatorSpec — the backend-agnostic shape of a pure-evaluator block.

An *evaluator block* is any lowerable IR whose entire runtime is "evaluate
baked `ca.Function`s" — no Kalman covariance, no native linear algebra.
`Sim` (forward dynamics), `LQR` (a baked control law), and every
`RecurrenceBlock` (PID, Madgwick, Mahony, the IMU integrator) are all this
shape. They differ only in their *entry points* — the typed, named methods
a backend exposes — so one generic lowering can serve all of them, keyed on
the single `evaluator` runtime kind. (The EKF is NOT an evaluator: it keeps
its own `ekf`/kalman kind.)

An `EvaluatorSpec` is what a backend consumes to emit one such block:

  * `spec`          — the `StateSpec` manifold layout (for pack/unpack).
  * `consts`        — the `static constexpr int` dims to expose, in order.
  * `state_fields` / `input_fields` / `output_fields` — the struct fields,
    each a `FieldSpec` whose `type_expr` / `init_expr` are already rendered
    for the target backend (the backend's *builder* fills these; the
    *emitter* just prints them — so this container stays language-neutral).
  * `entry_points`  — the ordered methods, each an `EntryPoint`.
  * `stateful`      — True iff the block holds a mutable member `State`
    (a recurrence's `step`); False iff `State` is threaded as a parameter
    (Sim/LQR).
  * `kernels`       — the `ca.Function`s to emit as flat-C kernels.
  * `funcs`         — carried through to the backend's result object
    (e.g. the C++ `EmitResult.funcs`; Sim keeps its `WorldFunctions`).

The per-backend *builder* (e.g. `cpp/evaluator_spec.py`) turns an IR block
into this; the per-backend *emitter* renders it. The structural skeleton
(entry-point names, param/return shapes) is backend-independent; only the
`type_expr`/`init_expr`/`fn_name` strings are backend-specific, and those
are produced by the builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParamKind(Enum):
    """A public method parameter."""
    STATE = "state"
    INPUTS = "inputs"
    DT = "dt"
    T = "t"


class ArgSource(Enum):
    """How to fill one positional slot of a flat-C kernel call."""
    STATE = "state"        # the packed state buffer
    INPUTS = "inputs"      # the packed inputs buffer
    DT = "dt"              # &dt — from the method's dt param, or a synthesized
                           # zero when the method exposes no dt (e.g. measure_*)
    T = "t"               # &t, from the method's t parameter


class ReturnKind(Enum):
    STATE = "state"                          # unpack flat → a fresh State
    INPUTS = "inputs"                        # scatter flat → the Inputs struct
    VEC = "vec"                              # an Eigen vector of `dim`
    MATRIX = "matrix"                        # an Eigen `rows`×`cols` matrix
    OUTPUTS_AND_STATE = "outputs_and_state"  # res→held State + res→Outputs


@dataclass(frozen=True)
class FieldSpec:
    """One struct field, pre-rendered for the target backend."""
    ident:     str
    type_expr: str            # e.g. "double", "Eigen::Vector3d"
    init_expr: str | None     # backend init string, or None for no initializer
    dim:       int            # element count (for flat pack/unpack arithmetic)


@dataclass(frozen=True)
class ReturnSpec:
    kind:      ReturnKind
    dim:       int = 0        # VEC
    rows:      int = 0        # MATRIX
    cols:      int = 0        # MATRIX


@dataclass(frozen=True)
class EntryPoint:
    """One typed method backed by (at most) one kernel."""
    method_name: str
    fn_name:     str | None                  # kernel C symbol; None = no kernel
    params:      tuple[ParamKind, ...]
    kernel_args: tuple[ArgSource, ...]
    ret:         ReturnSpec
    const:       bool = True                  # emits `const`; False mutates state


@dataclass
class EvaluatorSpec:
    """Everything a backend needs to lower one evaluator block."""
    spec:          Any                        # StateSpec
    consts:        tuple[tuple[str, int], ...]
    state_fields:  tuple[FieldSpec, ...]
    input_fields:  tuple[FieldSpec, ...]
    output_fields: tuple[FieldSpec, ...] | None
    entry_points:  tuple[EntryPoint, ...]
    stateful:      bool
    kernels:       tuple[Any, ...]            # ca.Function list to emit
    funcs:         Any                        # carried to the backend result
