"""IR → `EvaluatorSpec` builders for the C++ backend.

One builder per evaluator-block shape (Sim, LQR, recurrence). Each turns an
IR block into the backend-neutral `EvaluatorSpec` the generic
`evaluator_wrapper` emitter consumes. The entry-point *skeleton* (names,
params, return shapes) is backend-independent; this module fills the C++
`type_expr` / `init_expr` / kernel symbol strings, reusing the shared
`_structs` / `types` helpers so the emitted structs match the previous
per-kind wrappers.

`build_evaluator_spec(block)` dispatches on the concrete IR type — the one
place the C++ backend keys on block class, mirroring how the numpy backend
maps a block to its runtime façade.
"""

from __future__ import annotations

from . import _structs as S
from ._casadi import densify as _densify
from .extract import extract
from .types import cpp_type_for
from ..evaluator import (
    ArgSource, EntryPoint, EvaluatorSpec, FieldSpec, ParamKind,
    ReturnKind, ReturnSpec,
)

_P = ParamKind
_A = ArgSource


# ---------------------------------------------------------------------------
# Field builders (shared)
# ---------------------------------------------------------------------------

def _state_fields(spec, init_expr) -> tuple[FieldSpec, ...]:
    """State struct fields for `spec`; `init_expr(slot) -> str` renders the
    per-slot default."""
    return tuple(
        FieldSpec(ident=S.cpp_ident(s.name), type_expr=S.eigen_type_for(s),
                  init_expr=init_expr(s), dim=s.ambient_dim)
        for s in spec.slots)


def _scalar_input_fields(input_names, world) -> tuple[FieldSpec, ...]:
    """Sim/LQR inputs: one scalar `double` per input name (part default)."""
    return tuple(
        FieldSpec(ident=S.cpp_ident(n), type_expr="double",
                  init_expr=repr(S.input_default(n, world)), dim=1)
        for n in input_names)


def _port_fields(ports, *, with_init: bool) -> tuple[FieldSpec, ...]:
    """Recurrence inputs/outputs: one (possibly vector) field per port."""
    out = []
    for p in ports:
        tp = S.eigen_vec_type(p.dim)
        init = None
        if with_init:
            init = f"{tp}(0.0)" if p.dim == 1 else f"{tp}::Zero()"
        out.append(FieldSpec(ident=S.cpp_ident(p.name), type_expr=tp,
                             init_expr=init, dim=p.dim))
    return tuple(out)


def _x0_init(x0):
    """Return an `init_expr(slot)` that renders a recurrence slot's default
    from the packed `x0` vector."""
    def init(slot) -> str:
        cpp = cpp_type_for(slot.manifold)
        seg = x0[slot.ambient_offset:slot.ambient_offset + slot.ambient_dim]
        if slot.ambient_dim == 1:
            return cpp.literal(float(seg[0]))
        return cpp.literal([float(v) for v in seg])
    return init


# ---------------------------------------------------------------------------
# Per-shape builders
# ---------------------------------------------------------------------------

def evaluator_spec_for_sim(sim) -> EvaluatorSpec:
    return sim_spec_from_funcs(extract(sim), sim.world)


def sim_spec_from_funcs(funcs, world) -> EvaluatorSpec:
    """Build a Sim `EvaluatorSpec` from an already-extracted `WorldFunctions`.
    Used both by `evaluator_spec_for_sim` and by the `wrapper.py` shim."""
    spec = funcs.spec
    tan = spec.tangent_dim

    entries = [
        EntryPoint("initial_state", None, (), (), ReturnSpec(ReturnKind.STATE)),
        EntryPoint("predict", funcs.predict_fn.name(),
                   (_P.STATE, _P.INPUTS, _P.DT, _P.T),
                   (_A.STATE, _A.INPUTS, _A.DT, _A.T),
                   ReturnSpec(ReturnKind.STATE)),
        EntryPoint("predict_jacobian", funcs.predict_jacobian_fn.name(),
                   (_P.STATE, _P.INPUTS, _P.DT, _P.T),
                   (_A.STATE, _A.INPUTS, _A.DT, _A.T),
                   ReturnSpec(ReturnKind.MATRIX, rows=tan, cols=tan)),
    ]
    kernels = [funcs.predict_fn, funcs.predict_jacobian_fn]
    for o in funcs.outputs:
        ident = S.cpp_ident(o.full_name)
        # h/H take the full (x,u,dt,t) kernel signature but the public method
        # omits dt (h is dt-independent in our parts) — the emitter feeds a
        # zero dt since the kernel wants DT and the method has no DT param.
        entries.append(EntryPoint(
            f"measure_{ident}", o.h_fn.name(),
            (_P.STATE, _P.INPUTS, _P.T), (_A.STATE, _A.INPUTS, _A.DT, _A.T),
            ReturnSpec(ReturnKind.VEC, dim=o.out_dim)))
        entries.append(EntryPoint(
            f"measure_{ident}_jacobian", o.H_fn.name(),
            (_P.STATE, _P.INPUTS, _P.T), (_A.STATE, _A.INPUTS, _A.DT, _A.T),
            ReturnSpec(ReturnKind.MATRIX, rows=o.out_dim, cols=tan)))
        kernels += [o.h_fn, o.H_fn]

    return EvaluatorSpec(
        spec=spec,
        consts=(("ambient_dim", spec.ambient_dim), ("tangent_dim", tan),
                ("n_inputs", funcs.n_inputs)),
        state_fields=_state_fields(spec, lambda s: S.state_field_init(s, world)),
        input_fields=_scalar_input_fields(funcs.input_names, world),
        output_fields=None,
        entry_points=tuple(entries),
        stateful=False,
        kernels=tuple(kernels),
        funcs=funcs,
    )


def evaluator_spec_for_lqr(lqr) -> EvaluatorSpec:
    spec = lqr.spec
    world = lqr.world
    control_fn = _densify(lqr.control_fn, f"{world.name}_lqr_control")
    return EvaluatorSpec(
        spec=spec,
        consts=(("ambient_dim", spec.ambient_dim),
                ("n_inputs", len(lqr.input_names))),
        state_fields=_state_fields(spec, lambda s: S.state_field_init(s, world)),
        input_fields=_scalar_input_fields(list(lqr.input_names), world),
        output_fields=None,
        entry_points=(EntryPoint(
            "control", control_fn.name(), (_P.STATE,), (_A.STATE,),
            ReturnSpec(ReturnKind.INPUTS)),),
        stateful=False,
        kernels=(control_fn,),
        funcs=lqr,
    )


def evaluator_spec_for_recurrence(block) -> EvaluatorSpec:
    spec = block.spec
    return EvaluatorSpec(
        spec=spec,
        consts=(("ambient_dim", spec.ambient_dim),
                ("input_dim", block.input_dim),
                ("output_dim", block.output_dim)),
        state_fields=_state_fields(spec, _x0_init(block.x0)),
        input_fields=_port_fields(block.inputs, with_init=True),
        output_fields=_port_fields(block.outputs, with_init=False),
        entry_points=(EntryPoint(
            "step", block.update_fn.name(),
            (_P.INPUTS, _P.DT, _P.T), (_A.STATE, _A.INPUTS, _A.DT, _A.T),
            ReturnSpec(ReturnKind.OUTPUTS_AND_STATE),
            const=False),),
        stateful=True,
        kernels=(block.update_fn,),
        funcs=block,
    )


def build_evaluator_spec(block) -> EvaluatorSpec:
    """Dispatch on the concrete IR type to the matching builder."""
    from ...sim import Sim
    from ...control.lqr import LQR
    from ...recurrence import RecurrenceBlock
    if isinstance(block, Sim):
        return evaluator_spec_for_sim(block)
    if isinstance(block, LQR):
        return evaluator_spec_for_lqr(block)
    if isinstance(block, RecurrenceBlock):
        return evaluator_spec_for_recurrence(block)
    raise TypeError(
        f"build_evaluator_spec: {type(block).__name__} is not an evaluator "
        f"block (expected Sim, LQR, or a RecurrenceBlock).")
