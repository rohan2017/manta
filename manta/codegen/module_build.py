"""Block → `Module` builders — the one place a transform's IR is reshaped
into the backend-neutral `Module` (`manta.ir.module`).

Every analysis transform (`Sim`, `EKF`, `LQR`, the recurrence filters) maps
onto a Module: a typed State + named `ca.Function`s + typed entry points. A
backend then consumes only the Module — it never sees `Sim`/`EKF`/`LQR`/
`RecurrenceBlock`. This is the single source both backends lower; it
generalizes (and replaced) the old per-shape `EvaluatorSpec` builders. The
Sim branch builds its kernels directly from `Linearization` over the world
tick (predict / step / F / per-output h, H).

`to_module(block)` dispatches on the concrete IR type.
"""

from __future__ import annotations

from ..ir.module import (
    EntryPoint, Module, Port, PortField, StateField, StateLayout,
)


def _input_port(name, input_names, defaults) -> Port:
    """Build the `u` control Port carrying one named scalar field per Part
    Input (so a typed backend emits an Inputs struct)."""
    return Port(name, (len(input_names),), fields=tuple(
        PortField(n, 1, float(defaults[n])) for n in input_names))


# ---------------------------------------------------------------------------
# Sim — world x (one manifold field); step + measurement entries
# ---------------------------------------------------------------------------

def module_for_sim(sim) -> Module:
    import casadi as ca
    from ..estimation.state_spec import StateSpec
    from ..linearization import Linearization
    from ..linearized_system import flatten_nested
    from ..tick import walk_tick_signature

    world = sim.world
    spec = StateSpec.from_world(world)
    cf = sim.tick.casadi_function
    sig = walk_tick_signature(cf, world, spec)
    lin = Linearization(cf, spec, frozen={}, input_names=sig.input_names,
                        noise_specs=sig.noise, outputs=sig.sensor_names,
                        build_functions=True)

    init_flat = flatten_nested(world._initial_state_dict())
    x0 = spec.pack({k: v for k, v in init_flat.items() if k in spec})
    x_field = StateField("x", "manifold", (spec.ambient_dim,),
                         init=x0, manifold=spec)

    ports = [_input_port("u", sig.input_names, sig.input_defaults),
             Port("noise", (lin.n_noise,)), Port("dt", ()), Port("t", ())]

    # The Sim `step` is the full forward tick — it advances the state AND
    # emits every sensor reading, from ONE noise draw (so a driven runtime's
    # state and readings share the same realization). Noise is a Port (the
    # tick's honest signature); the noiseless oracle just passes zeros. The
    # readings are RETURNED, not stored in State — the runtime publishes them
    # to ports / `outputs()`, never into the state dict.
    sensor_fulls = list(lin.outputs)
    step_outs = [lin.x_new_noisy] + [
        ca.reshape(lin.outputs[f].h_noisy_sym, lin.outputs[f].dim, 1)
        for f in sensor_fulls]
    step_names = ["x_new"] + [f.replace(".", "_") for f in sensor_fulls]
    step_fn = ca.Function(
        "step",
        [lin.x_sym, lin.u_sym, lin.n_sym, lin.dt_sym, lin.t_sym],
        step_outs, ["x", "u", "noise", "dt", "t"], step_names)
    # `predict` is the NOISELESS deploy-shaped forward map (x,u,dt,t)→x —
    # what an on-device backend (C++) runs against real sensors. `step`
    # (above) is the driven-oracle variant the numpy sim uses. Same tick;
    # `predict` == `step`'s x output at noise=0.
    functions = {"step": step_fn, "predict": lin.predict_fn, "F": lin.F_fn}
    entries = [
        # `step` is the numpy driven-oracle variant (deploy=False); a deploy
        # backend lowers `predict` + `measure_*` instead.
        EntryPoint("step", "step", ("x", "u", "noise", "dt", "t"),
                   writes_state=("x",), returns=tuple(sensor_fulls),
                   deploy=False),
        EntryPoint("predict", "predict", ("x", "u", "dt", "t"),
                   writes_state=("x",)),
        EntryPoint("predict_jacobian", "F", ("x", "u", "dt", "t"),
                   returns=("F",)),
    ]
    analysis = {"F": lin.F_fn}

    # Measurement entries: noiseless h(x,u,t) -> reading + its Jacobian H,
    # WITHOUT stepping. dt is dt-independent (synthesized zero, not exposed).
    for full, o in lin.outputs.items():
        ident = full.replace(".", "_")
        hk, Hk = f"h_{ident}", f"H_{ident}"
        functions[hk] = o.h_fn
        functions[Hk] = o.H_fn
        entries.append(EntryPoint(f"measure_{ident}", hk,
                                  ("x", "u", "dt", "t"), returns=(full,),
                                  synthesize_zero=("dt",)))
        entries.append(EntryPoint(f"measure_{ident}_jacobian", Hk,
                                  ("x", "u", "dt", "t"), returns=(Hk,),
                                  synthesize_zero=("dt",)))
        analysis[Hk] = o.H_fn

    return Module(world.name, StateLayout((x_field,)), tuple(ports),
                  functions, tuple(entries), analysis)


# ---------------------------------------------------------------------------
# LQR — no state; one stateless control(x_full) -> u entry
# ---------------------------------------------------------------------------

def module_for_lqr(lqr) -> Module:
    spec = lqr.spec                       # full ambient layout (the law gathers)
    # `x` is a manifold state vector (carries the spec so a typed backend
    # builds the same State struct); `u` is the returned control, named.
    defaults = {n: 0.0 for n in lqr.input_names}
    ports = (Port("x", (spec.ambient_dim,), manifold=spec),
             _input_port("u", lqr.input_names, defaults))
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
    u_port = Port("u", (block.input_dim,), fields=tuple(
        PortField(p.name, p.dim, 0.0) for p in block.inputs))
    y_port = Port("y", (block.output_dim,), fields=tuple(
        PortField(p.name, p.dim, 0.0) for p in block.outputs))
    ports = (u_port, y_port, Port("dt", ()), Port("t", ()))
    return Module(
        block.name, StateLayout((x_field,)), ports,
        {"step": block.update_fn},
        (EntryPoint("step", "step", ("x", "u", "dt", "t"),
                    writes_state=("x",), returns=("y",)),),
        held=True)


# ---------------------------------------------------------------------------
# EKF — the proof the filter "is not special": a two-field (x, P) State, a
# predict entry + one update entry per sensor, all baked ca.Functions.
# ---------------------------------------------------------------------------

def module_for_ekf(ekf) -> Module:
    import numpy as np
    from ..linearized_system import flatten_nested

    spec = ekf.spec
    tan = spec.tangent_dim

    # Initial nominal state: pack the world's seed into the EKF (sub)spec
    # (same source NumpyEKF uses). P seeds to the same 1e-2·I default.
    init_flat = flatten_nested(ekf.world._initial_state_dict())
    x0 = spec.pack({k: v for k, v in init_flat.items() if k in spec})
    fields = (
        StateField("x", "manifold", (spec.ambient_dim,), init=x0, manifold=spec),
        StateField("P", "matrix", (tan, tan), init=np.eye(tan) * 1e-2),
    )

    defaults = {n: float(d) for n, d in
                zip(ekf._input_names, ekf._u_defaults)}
    ports = [_input_port("u", ekf._input_names, defaults),
             Port("dt", ()), Port("t", ()), Port("Q", (tan, tan))]

    # predict: x' = f, P' = F P Fᵀ + Q   (Q supplied as a port; the model's
    # default Q comes from the `process_noise` entry below).
    functions = {"predict": ekf._predict_fn}
    # `Q` is computed internally — from the `process_noise` kernel when the
    # model declares noise, else zero — never a method parameter.
    if ekf._process_noise_fn is not None:
        functions["process_noise"] = ekf._process_noise_fn
        predict = EntryPoint("predict", "predict",
                             ("x", "P", "Q", "u", "dt", "t"),
                             writes_state=("x", "P"),
                             port_from={"Q": "process_noise"})
        # Exposed for the numpy runtime (nm.process_noise); a deploy backend
        # folds it into predict via port_from, so deploy=False here.
        extra_process_noise = EntryPoint(
            "process_noise", "process_noise", ("x", "u", "dt", "t"),
            returns=("Q",), deploy=False)
    else:
        predict = EntryPoint("predict", "predict",
                             ("x", "P", "Q", "u", "dt", "t"),
                             writes_state=("x", "P"), synthesize_zero=("Q",))
        extra_process_noise = None
    entries = [predict]
    if extra_process_noise is not None:
        entries.append(extra_process_noise)
    analysis = {k: v for k, v in (("F", ekf._F_fn),) if v is not None}

    # One Joseph-update entry per sensor: (x, P, z, u, t) -> (x, P).
    for (_pid, _out), o in ekf._sensors.items():
        full = o["full"]
        ident = full.replace(".", "_")
        zname = f"z_{ident}"
        fn_key = f"update_{ident}"
        functions[fn_key] = o["update_fn"]
        ports.append(Port(zname, (o["dim"],)))
        entries.append(EntryPoint(f"update_{ident}", fn_key,
                                  ("x", "P", zname, "u", "t"),
                                  writes_state=("x", "P")))
        if o.get("H_fn") is not None:
            analysis[f"H_{ident}"] = o["H_fn"]

    return Module(f"{ekf.world.name}_ekf", StateLayout(fields), tuple(ports),
                  functions, tuple(entries), analysis, held=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def to_module(block) -> Module:
    """Convert a lowerable IR block into its `Module`."""
    from ..sim import Sim
    from ..control.lqr import LQR
    from ..recurrence import RecurrenceBlock
    from ..estimation.ekf import EKF
    if isinstance(block, Sim):
        return module_for_sim(block)
    if isinstance(block, LQR):
        return module_for_lqr(block)
    if isinstance(block, RecurrenceBlock):
        return module_for_recurrence(block)
    if isinstance(block, EKF):
        return module_for_ekf(block)
    raise TypeError(
        f"to_module: {type(block).__name__} has no Module builder "
        f"(expected Sim, LQR, EKF, or a RecurrenceBlock).")
