"""Per-function CasADi extraction from a CompiledWorld.

Given a compiled World (an IR carrying a single CasADi tick function
over every craft + every state-bearing disturbance), `extract` builds
the function bundle the codegen backends consume:

  * predict_fn(x_flat, u_flat, dt, t) → x_flat_new
  * predict_jacobian_fn(x_flat, u_flat, dt, t) → F  (tangent_dim²)
  * For every Part Output named "<craft>.<part>.<name>":
      h_<craft>_<part>_<name>(x_flat, u_flat, dt, t) → flat value
      H_<craft>_<part>_<name>(x_flat, u_flat, dt, t) → ∂h/∂δ at δ=0

Flat because CasADi's codegen targets row-major flat C; the C++
wrapper handles typed pack/unpack via the StateSpec slot layout.

The world tick's named inputs / outputs are routed by walking the
casadi.Function signature:

  Owners — head of `<owner>.<sub>` — are looked up in the world. They
  can be a craft (which exposes parts with Input / Output / Noise
  declarations) or a state-bearing disturbance (which exposes Noise
  declarations driving its bias state).

  Sub-names — the rest — are matched to State (already in spec), Input
  (routed to u), white-Noise (zero-fed for the clean predict), or RW
  drivers (`<bias>_driver`, also zero-fed for the clean predict).
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ..estimation.state_spec import StateSpec


@dataclass(frozen=True)
class OutputSpec:
    """One Part Output, ready to codegen."""
    craft_name:   str
    part_name:    str
    output_name:  str
    out_dim:      int
    h_fn:         ca.Function
    H_fn:         ca.Function

    @property
    def flat_name(self) -> str:
        """C-identifier-safe name used by codegen
        (`drone_gps_position`, `alpha_imu_gyro`)."""
        return f"{self.craft_name}_{self.part_name}_{self.output_name}"

    @property
    def full_name(self) -> str:
        """Dotted name as used in state dicts (`drone.gps.position`)."""
        return f"{self.craft_name}.{self.part_name}.{self.output_name}"


@dataclass
class WorldFunctions:
    """The complete set of exportable ca.Functions for one CompiledWorld."""
    world_name:          str
    spec:                StateSpec
    input_names:         list[str]      # ordered as the u vector
    predict_fn:          ca.Function    # x, u, dt, t → x_new
    predict_jacobian_fn: ca.Function    # x, u, dt, t → F
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


def extract(cw) -> WorldFunctions:
    """Extract per-function ca.Functions from a CompiledWorld.

    Args:
        cw  — the manta CompiledWorld to extract from.

    Returns:
        A `WorldFunctions` bundle.
    """
    from ..world import CompiledWorld
    if not isinstance(cw, CompiledWorld):
        raise TypeError(
            f"extract: expected CompiledWorld, got {type(cw).__name__}")

    world = cw.world
    spec  = StateSpec.from_world(world)
    cf    = cw.tick.casadi_function

    n_ambient = spec.ambient_dim
    n_tangent = spec.tangent_dim

    # Index field disturbances by name for noise-routing.
    dist_by_name = _build_dist_lookup(world)

    # Walk tick inputs → discover Inputs (u) + Noise (zero-fed).
    input_names = _discover_input_names(cf, world, spec, dist_by_name)
    n_u         = len(input_names)

    # Top-level MX symbols.
    x_sym  = ca.MX.sym("x",  n_ambient, 1)
    u_sym  = ca.MX.sym("u",  n_u, 1) if n_u > 0 else ca.MX.zeros(0, 1)
    dt_sym = ca.MX.sym("dt", 1, 1)
    t_sym  = ca.MX.sym("t",  1, 1)

    # Evaluate the tick on flat symbols; gather outputs by name.
    tick_outputs_by_name = _evaluate_tick_flat(
        cf, x_sym, u_sym, dt_sym, t_sym, spec, input_names,
        world, dist_by_name)

    # --- predict_fn -------------------------------------------------------
    state_chunks = []
    for slot in spec.slots:
        chunk = tick_outputs_by_name[slot.name]
        if chunk.shape != (slot.dim, 1):
            chunk = ca.reshape(chunk, slot.dim, 1)
        state_chunks.append(chunk)
    x_new = ca.vertcat(*state_chunks)

    predict_fn = ca.Function(
        f"{world.name}_predict",
        [x_sym, u_sym, dt_sym, t_sym], [x_new],
        ["x", "u", "dt", "t"], ["x_new"],
    )

    # --- predict_jacobian_fn ---------------------------------------------
    delta = ca.MX.sym("delta", n_tangent, 1)
    x_pert = spec.boxplus_sym(x_sym, delta)
    pert_outputs_by_name = _evaluate_tick_flat(
        cf, x_pert, u_sym, dt_sym, t_sym, spec, input_names,
        world, dist_by_name)
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
    F_sym = ca.densify(F_sym)
    predict_jacobian_fn = ca.Function(
        f"{world.name}_predict_jacobian",
        [x_sym, u_sym, dt_sym, t_sym], [F_sym],
        ["x", "u", "dt", "t"], ["F"],
    )

    # --- per-Output (h, H) -----------------------------------------------
    outputs: list[OutputSpec] = []
    for craft in world.crafts:
        for part in craft.parts:
            for out_name in part.output_declarations():
                full = f"{craft.name}.{part.name}.{out_name}"
                if full not in tick_outputs_by_name:
                    raise RuntimeError(
                        f"extract: output {full!r} declared on part but "
                        f"missing from world tick outputs (compile bug?)")
                h_mx = tick_outputs_by_name[full]
                out_dim = int(h_mx.numel())
                h_mx_flat = ca.reshape(h_mx, out_dim, 1)

                h_pert_mx = pert_outputs_by_name[full]
                h_pert_flat = ca.reshape(h_pert_mx, out_dim, 1)
                H_sym = ca.substitute(
                    ca.jacobian(h_pert_flat, delta),
                    delta,
                    ca.MX.zeros(n_tangent, 1),
                )
                H_sym = ca.densify(H_sym)
                cname = f"{world.name}_h_{craft.name}_{part.name}_{out_name}"
                h_fn = ca.Function(
                    cname,
                    [x_sym, u_sym, dt_sym, t_sym], [h_mx_flat],
                    ["x", "u", "dt", "t"], ["h"],
                )
                H_fn = ca.Function(
                    cname + "_jacobian",
                    [x_sym, u_sym, dt_sym, t_sym], [H_sym],
                    ["x", "u", "dt", "t"], ["H"],
                )
                outputs.append(OutputSpec(
                    craft_name=craft.name,
                    part_name=part.name,
                    output_name=out_name,
                    out_dim=out_dim,
                    h_fn=h_fn,
                    H_fn=H_fn,
                ))

    return WorldFunctions(
        world_name=world.name,
        spec=spec,
        input_names=input_names,
        predict_fn=predict_fn,
        predict_jacobian_fn=predict_jacobian_fn,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Input / Noise routing helpers
# ---------------------------------------------------------------------------

def _build_dist_lookup(world) -> dict:
    """Index every state-bearing disturbance by name for noise routing."""
    from ..fields.base import Disturbance
    out = {}
    for field in world.fields:
        for d in field._disturbances:
            if isinstance(d, Disturbance):
                out[d.name] = d
    return out


def _resolve_owner(world, dist_by_name: dict, head: str):
    """Return (owner, owner_label) for `head` — a craft, a disturbance,
    or (None, None)."""
    craft = next((c for c in world.crafts if c.name == head), None)
    if craft is not None:
        return craft, f"craft '{craft.name}'"
    dist = dist_by_name.get(head)
    if dist is not None:
        return dist, f"disturbance '{dist.name}'"
    return None, None


def _discover_input_names(cf: ca.Function,
                          world,
                          spec: StateSpec,
                          dist_by_name: dict) -> list[str]:
    """Walk the tick's input signature; return the ordered list of
    part-Input names. Noise channels are recognized but routed
    separately at evaluation time (zero-fed for the clean predict)."""
    out: list[str] = []
    for i in range(cf.n_in()):
        name = cf.name_in(i)
        if name in ("dt", "t") or name in spec:
            continue
        head, _, rest = name.partition(".")
        owner, owner_label = _resolve_owner(world, dist_by_name, head)
        if owner is None:
            raise RuntimeError(
                f"extract: tick input {name!r} doesn't match any craft "
                f"or disturbance in this world.")
        if isinstance(owner, type(world.crafts[0])) and "." in rest:
            # Per-craft Input or Noise → drill into the right part.
            part_name, sub = rest.split(".", 1)
            part = next((p for p in owner.parts if p.name == part_name), None)
            if part is None:
                raise RuntimeError(
                    f"extract: tick input {name!r}: unknown part "
                    f"{part_name!r} on craft {owner.name!r}.")
            if sub in part.input_declarations():
                out.append(name)
                continue
            if _is_known_noise_input(part, sub):
                continue
            raise RuntimeError(
                f"extract: tick input {name!r} on {owner_label} is not "
                f"an Input or recognized Noise.")
        # Per-disturbance Noise (state slots already handled by `in spec`).
        if _is_known_noise_input(owner, rest):
            continue
        raise RuntimeError(
            f"extract: tick input {name!r} on {owner_label} is not a "
            f"recognized Noise channel.")
    return out


def _is_known_noise_input(owner, sub: str) -> bool:
    """True if `<owner>.<sub>` matches a white Noise channel or an RW
    Noise driver (`<bias>_driver`)."""
    ndecls = owner.noise_declarations()
    if sub in ndecls and ndecls[sub].kind == "white":
        return True
    if sub.endswith("_driver"):
        bias = sub[: -len("_driver")]
        if bias in ndecls and ndecls[bias].kind == "rw":
            return True
    return False


def _noise_zero(owner, sub: str):
    """Return the zero MX of the right shape for a Noise input.
    Raises if the input doesn't match a known channel."""
    ndecls = owner.noise_declarations()
    if sub in ndecls and ndecls[sub].kind == "white":
        ndecl = ndecls[sub]
        dim = 1 if ndecl.shape == "scalar" else 3
        return ca.MX.zeros(dim, 1)
    if sub.endswith("_driver"):
        bias = sub[: -len("_driver")]
        if bias in ndecls and ndecls[bias].kind == "rw":
            ndecl = ndecls[bias]
            dim = 1 if ndecl.shape == "scalar" else 3
            return ca.MX.zeros(dim, 1)
    raise RuntimeError(f"extract: unhandled noise sub-name {sub!r}")


def _evaluate_tick_flat(cf: ca.Function,
                        x_sym,
                        u_sym,
                        dt_sym,
                        t_sym,
                        spec: StateSpec,
                        input_names: list[str],
                        world,
                        dist_by_name: dict) -> dict:
    """Apply the world tick on flat symbolic inputs; return a dict mapping
    every named output to its symbolic MX expression."""
    in_names  = [cf.name_in(i)  for i in range(cf.n_in())]
    out_names = [cf.name_out(i) for i in range(cf.n_out())]
    u_index   = {name: i for i, name in enumerate(input_names)}

    sliced = []
    for name in in_names:
        if name == "dt":
            sliced.append(dt_sym); continue
        if name == "t":
            sliced.append(t_sym); continue
        if name in spec:
            slot = spec.slot(name)
            sliced.append(x_sym[slot.offset : slot.offset + slot.dim])
            continue
        if name in u_index:
            sliced.append(u_sym[u_index[name]])
            continue
        # Noise — feed zero of the right shape.
        head, _, rest = name.partition(".")
        owner, _ = _resolve_owner(world, dist_by_name, head)
        if owner is None:
            raise RuntimeError(f"extract: tick input {name!r} not handled.")
        # Per-craft noise lives at `<craft>.<part>.<sub>`.
        from ..craft import Craft as _Craft
        if isinstance(owner, _Craft) and "." in rest:
            part_name, sub = rest.split(".", 1)
            part = next((p for p in owner.parts if p.name == part_name), None)
            if part is None:
                raise RuntimeError(
                    f"extract: tick input {name!r}: unknown part.")
            sliced.append(_noise_zero(part, sub))
            continue
        # Per-disturbance noise lives at `<dist>.<sub>`.
        sliced.append(_noise_zero(owner, rest))

    result = cf(*sliced)
    if len(out_names) == 1:
        return {out_names[0]: result}
    return {n: result[i] for i, n in enumerate(out_names)}
