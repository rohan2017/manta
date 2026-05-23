"""Per-function CasADi extraction from a Craft.

Given a Craft, we want exportable ca.Function objects:

  * predict(x_flat, u_flat, dt) → x_flat_new
  * predict_jacobian(x_flat, u_flat, dt) → F (tangent_dim × tangent_dim)
  * For every Part Output named "<part>.<name>":
      h_<part>_<name>(x_flat, u_flat, dt) → output value (flat)
      H_<part>_<name>(x_flat, u_flat, dt) → ∂h/∂δ at δ=0 (out_dim × tangent_dim)

Why flat? CasADi's codegen is row-major flat C; the C++ wrapper handles
typed pack/unpack. Why include dt in the measurement Jacobian arg list?
For uniformity — most Outputs don't actually use dt, but a few might (an
integrating sensor, a difference operator). CasADi prunes unused args
from the generated code automatically.

Mirrors EKF._tick_on_flat / EKF.__init__ but exports more functions and
doesn't carry a runtime _x / _P. The shared bits (StateSpec walking,
input enumeration) live here so EKF can grow to use this module later.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ..estimation.state_spec import StateSpec


@dataclass(frozen=True)
class OutputSpec:
    """One Part Output, ready to codegen."""
    part_name: str
    output_name: str
    out_dim: int       # ambient dim of the value (vec3 → 3, scalar → 1)
    h_fn:  ca.Function  # h(x, u, dt) → value
    H_fn:  ca.Function  # H(x, u, dt) → ∂h/∂δ at δ=0  (out_dim × tangent_dim)

    @property
    def flat_name(self) -> str:
        """C-identifier-safe name used by codegen (`gps_position`, `imu_gyro`)."""
        return f"{self.part_name}_{self.output_name}"

    @property
    def full_name(self) -> str:
        """Dotted name as used in state dicts (`gps.position`)."""
        return f"{self.part_name}.{self.output_name}"


@dataclass
class CraftFunctions:
    """The complete set of exportable ca.Functions for one Craft."""
    craft_name: str
    spec: StateSpec
    input_names: list[str]                  # ordered as the u vector
    predict_fn:          ca.Function        # x, u, dt → x_new
    predict_jacobian_fn: ca.Function        # x, u, dt → F
    outputs:             list[OutputSpec]

    @property
    def ambient_dim(self) -> int:
        return self.spec.ambient_dim

    @property
    def tangent_dim(self) -> int:
        return self.spec.tangent_dim

    @property
    def n_inputs(self) -> int:
        return len(self.input_names)


def extract(craft,
            *,
            gravity_field=None,
            fluid_field=None,
            mag_field=None,
            collision_field=None,
            ) -> CraftFunctions:
    """Extract per-function ca.Functions from a Craft.

    The compiled tick is built once; all extracted functions reference
    the same set of MX symbols so they can be evaluated/codegen'd
    consistently.

    Args:
        craft           — the manta_next Craft to extract from.
        gravity_field   — GravityField baked into the predict kernel.
                          None ⇒ no gravity.
        fluid_field / mag_field / collision_field — as for compile_tick.

    Returns:
        A `CraftFunctions` bundle.
    """
    spec = StateSpec.from_craft(craft)
    n_ambient = spec.ambient_dim
    n_tangent = spec.tangent_dim

    compiled_tick = craft.compile_tick(
        gravity_field=gravity_field,
        fluid_field=fluid_field,
        mag_field=mag_field,
        collision_field=collision_field,
    )
    cf = compiled_tick.casadi_function

    # Enumerate part Inputs from the tick's input signature.
    input_names = _discover_input_names(cf, craft, spec)
    n_u = len(input_names)

    # Top-level MX symbols.
    x_sym  = ca.MX.sym("x",  n_ambient, 1)
    u_sym  = ca.MX.sym("u",  n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
    dt_sym = ca.MX.sym("dt", 1, 1)
    t_sym  = ca.MX.sym("t",  1, 1)

    # Group tick outputs by name (state slots + sensor outputs).
    tick_outputs_by_name = _evaluate_tick_flat(
        cf, x_sym, u_sym, dt_sym, t_sym, spec, input_names, craft=craft)

    # --- predict_fn -------------------------------------------------------
    state_chunks = []
    for slot in spec.slots:
        chunk = tick_outputs_by_name[slot.name]
        if chunk.shape != (slot.dim, 1):
            chunk = ca.reshape(chunk, slot.dim, 1)
        state_chunks.append(chunk)
    x_new = ca.vertcat(*state_chunks)

    predict_fn = ca.Function(
        f"{craft.name}_predict",
        [x_sym, u_sym, dt_sym, t_sym], [x_new],
        ["x", "u", "dt", "t"], ["x_new"],
    )

    # --- predict_jacobian_fn ---------------------------------------------
    delta = ca.MX.sym("delta", n_tangent, 1)
    x_pert = spec.boxplus_sym(x_sym, delta)
    pert_outputs_by_name = _evaluate_tick_flat(
        cf, x_pert, u_sym, dt_sym, t_sym, spec, input_names, craft=craft)
    pert_chunks = []
    for slot in spec.slots:
        chunk = pert_outputs_by_name[slot.name]
        if chunk.shape != (slot.dim, 1):
            chunk = ca.reshape(chunk, slot.dim, 1)
        pert_chunks.append(chunk)
    x_pert_new = ca.vertcat(*pert_chunks)
    delta_out = spec.boxminus_sym(x_pert_new, x_new)
    F_sym = ca.substitute(
        ca.jacobian(delta_out, delta),
        delta,
        ca.MX.zeros(n_tangent, 1),
    )
    # densify so codegen writes a dense column-major matrix to res[0]
    # instead of CasADi's packed sparse storage.
    F_sym = ca.densify(F_sym)
    predict_jacobian_fn = ca.Function(
        f"{craft.name}_predict_jacobian",
        [x_sym, u_sym, dt_sym, t_sym], [F_sym],
        ["x", "u", "dt", "t"], ["F"],
    )

    # --- per-Output (h, H) -----------------------------------------------
    outputs: list[OutputSpec] = []
    for part in craft.parts:
        for out_name in part.output_declarations():
            full = f"{part.name}.{out_name}"
            if full not in tick_outputs_by_name:
                raise RuntimeError(
                    f"extract: output {full!r} declared on part but missing "
                    f"from compiled tick outputs (compile_tick bug?)")
            h_mx = tick_outputs_by_name[full]
            out_dim = int(h_mx.numel())
            h_mx_flat = ca.reshape(h_mx, out_dim, 1)

            # H = ∂h(boxplus(x, δ))/∂δ at δ=0.
            h_pert_mx = pert_outputs_by_name[full]
            h_pert_flat = ca.reshape(h_pert_mx, out_dim, 1)
            H_sym = ca.substitute(
                ca.jacobian(h_pert_flat, delta),
                delta,
                ca.MX.zeros(n_tangent, 1),
            )
            H_sym = ca.densify(H_sym)
            cname = f"{craft.name}_h_{part.name}_{out_name}"
            h_fn = ca.Function(
                cname, [x_sym, u_sym, dt_sym, t_sym], [h_mx_flat],
                ["x", "u", "dt", "t"], ["h"],
            )
            H_fn = ca.Function(
                cname + "_jacobian",
                [x_sym, u_sym, dt_sym, t_sym], [H_sym],
                ["x", "u", "dt", "t"], ["H"],
            )
            outputs.append(OutputSpec(
                part_name=part.name,
                output_name=out_name,
                out_dim=out_dim,
                h_fn=h_fn,
                H_fn=H_fn,
            ))

    return CraftFunctions(
        craft_name=craft.name,
        spec=spec,
        input_names=input_names,
        predict_fn=predict_fn,
        predict_jacobian_fn=predict_jacobian_fn,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_input_names(cf: ca.Function,
                          craft,
                          spec: StateSpec) -> list[str]:
    """Return the ordered list of part-Input names found in the tick
    Function's input signature (skipping state slots, dt, and Noise
    channels — Noise inputs are routed separately, see
    `_discover_noise_names`)."""
    out: list[str] = []
    for i in range(cf.n_in()):
        name = cf.name_in(i)
        if name in ("dt", "t") or name in spec:
            continue
        if "." in name:
            part_name, sub = name.split(".", 1)
            part = next((p for p in craft.parts if p.name == part_name), None)
            if part and sub in part.input_declarations():
                out.append(name)
                continue
            if part and sub in part.noise_declarations():
                # White driver (RW drivers carry the `_driver` suffix
                # below and are not in noise_declarations directly).
                continue
            if (part and sub.endswith("_driver")
                    and sub[: -len("_driver")] in part.noise_declarations()
                    and part.noise_declarations()[sub[: -len("_driver")]].kind
                        == "rw"):
                continue   # RW driver input
        raise RuntimeError(
            f"extract: tick input {name!r} not in StateSpec and not a "
            f"recognized part Input or Noise.")
    return out


def _discover_noise_names(cf: ca.Function, craft) -> list[str]:
    """Return the ordered list of Noise channels (white drivers + RW
    drivers) exposed by the tick."""
    out: list[str] = []
    for i in range(cf.n_in()):
        name = cf.name_in(i)
        if "." not in name:
            continue
        part_name, sub = name.split(".", 1)
        part = next((p for p in craft.parts if p.name == part_name), None)
        if part is None:
            continue
        if sub in part.noise_declarations() \
                and part.noise_declarations()[sub].kind == "white":
            out.append(name)
        elif (sub.endswith("_driver")
              and sub[: -len("_driver")] in part.noise_declarations()
              and part.noise_declarations()[sub[: -len("_driver")]].kind
                  == "rw"):
            out.append(name)
    return out


def _evaluate_tick_flat(cf: ca.Function,
                        x_sym,
                        u_sym,
                        dt_sym,
                        t_sym,
                        spec: StateSpec,
                        input_names: list[str],
                        craft=None) -> dict:
    """Apply the compiled tick CasADi Function to a flat ambient-state
    symbolic vector + flat input vector + dt + t, returning a dict mapping
    every output name (state slots + sensor outputs) → its symbolic MX."""
    in_names  = [cf.name_in(i)  for i in range(cf.n_in())]
    out_names = [cf.name_out(i) for i in range(cf.n_out())]
    u_index = {name: i for i, name in enumerate(input_names)}

    sliced = []
    for name in in_names:
        if name == "dt":
            sliced.append(dt_sym)
        elif name == "t":
            sliced.append(t_sym)
        elif name in spec:
            slot = spec.slot(name)
            sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
        elif name in u_index:
            sliced.append(u_sym[u_index[name]])
        elif "." in name and craft is not None:
            # Noise channel — feed zero of the right shape (predict /
            # H jacobian evaluate the clean tick). White: name matches
            # a noise declaration directly. RW driver: name ends in
            # `_driver`, prefix matches an RW noise declaration.
            part_name, sub = name.split(".", 1)
            part = next((p for p in craft.parts if p.name == part_name), None)
            if part and sub in part.noise_declarations() \
                    and part.noise_declarations()[sub].kind == "white":
                ndecl = part.noise_declarations()[sub]
                dim = 1 if ndecl.shape == "scalar" else 3
                sliced.append(ca.MX.zeros(dim, 1))
                continue
            if (part and sub.endswith("_driver")):
                bias = sub[: -len("_driver")]
                if (bias in part.noise_declarations()
                        and part.noise_declarations()[bias].kind == "rw"):
                    ndecl = part.noise_declarations()[bias]
                    dim = 1 if ndecl.shape == "scalar" else 3
                    sliced.append(ca.MX.zeros(dim, 1))
                    continue
            raise RuntimeError(f"extract: tick input {name!r} not handled.")
        else:
            raise RuntimeError(f"extract: tick input {name!r} not handled.")

    result = cf(*sliced)
    if len(out_names) == 1:
        return {out_names[0]: result}
    return {n: result[i] for i, n in enumerate(out_names)}
