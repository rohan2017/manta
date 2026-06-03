"""Block → `Module` builders — the one place a transform's IR is reshaped
into the backend-neutral `Module` (`manta.ir.module`).

Every analysis transform (`Sim`, `LQR`, the recurrence filters; the `EKF`
follows once its state is the `(x, P)` two-field layout) maps onto a Module:
a typed State + named `ca.Function`s + typed entry points. A backend then
consumes only the Module — it never sees `Sim`/`LQR`/`RecurrenceBlock`. This
generalizes the old per-shape `EvaluatorSpec` builders
(`codegen/cpp/evaluator_spec.py`); the Sim branch reuses the existing flat
`extract()` (its location under `cpp/` is incidental — it produces
backend-neutral `ca.Function`s).

`to_module(block)` dispatches on the concrete IR type.
"""

from __future__ import annotations

from ..ir.module import EntryPoint, Module, Port, StateField, StateLayout


# ---------------------------------------------------------------------------
# Sim — world x (one manifold field); step + measurement entries
# ---------------------------------------------------------------------------

def module_for_sim(sim) -> Module:
    from .cpp.extract import extract
    from ..linearized_world import flatten_nested

    funcs = extract(sim)
    spec = funcs.spec

    init_flat = flatten_nested(sim.world._initial_state_dict())
    x0 = spec.pack({k: v for k, v in init_flat.items() if k in spec})
    x_field = StateField("x", "manifold", (spec.ambient_dim,),
                         init=x0, manifold=spec)

    ports = (Port("u", (funcs.n_inputs,)), Port("dt", ()), Port("t", ()))
    functions = {"step": funcs.predict_fn}
    entries = [EntryPoint("step", "step", ("x", "u", "dt", "t"),
                          writes_state=("x",))]
    analysis = {"F": funcs.predict_jacobian_fn}

    # Measurement entry points: h(x,u,t) -> reading. dt is a kernel arg the
    # method feeds zero (a measurement is dt-independent), so no dt port on
    # the method — the runtime supplies the kernel's dt slot as 0.
    for o in funcs.outputs:
        hk, Hk = f"h_{o.flat_name}", f"H_{o.flat_name}"
        functions[hk] = o.h_fn
        entries.append(EntryPoint(f"measure_{o.flat_name}", hk,
                                  ("x", "u", "dt", "t"),
                                  returns=(o.full_name,)))
        analysis[Hk] = o.H_fn

    return Module(sim.world.name, StateLayout((x_field,)), ports,
                  functions, tuple(entries), analysis)


# ---------------------------------------------------------------------------
# LQR — no state; one stateless control(x_full) -> u entry
# ---------------------------------------------------------------------------

def module_for_lqr(lqr) -> Module:
    spec = lqr.spec                       # full ambient layout (the law gathers)
    ports = (Port("x", (spec.ambient_dim,)),)
    return Module(
        f"{lqr.world.name}_lqr", StateLayout(()), ports,
        {"control": lqr.control_fn},
        (EntryPoint("control", "control", ("x",), returns=("u",)),))


# ---------------------------------------------------------------------------
# Recurrence (PID / Madgwick / Mahony / IMU) — one manifold state field
# ---------------------------------------------------------------------------

def module_for_recurrence(block) -> Module:
    spec = block.spec
    x_field = StateField("x", "manifold", (spec.ambient_dim,),
                         init=block.x0.copy(), manifold=spec)
    ports = (Port("u", (block.input_dim,)), Port("dt", ()), Port("t", ()))
    return Module(
        block.name, StateLayout((x_field,)), ports,
        {"step": block.update_fn},
        (EntryPoint("step", "step", ("x", "u", "dt", "t"),
                    writes_state=("x",), returns=("y",)),))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def to_module(block) -> Module:
    """Convert a lowerable IR block into its `Module`."""
    from ..sim import Sim
    from ..control.lqr import LQR
    from ..recurrence import RecurrenceBlock
    if isinstance(block, Sim):
        return module_for_sim(block)
    if isinstance(block, LQR):
        return module_for_lqr(block)
    if isinstance(block, RecurrenceBlock):
        return module_for_recurrence(block)
    raise TypeError(
        f"to_module: {type(block).__name__} has no Module builder "
        f"(expected Sim, LQR, or a RecurrenceBlock).")
