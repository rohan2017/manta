"""Per-function CasADi extraction from a Sim.

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

from ...estimation.state_spec import StateSpec
from ...linearization import Linearization


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
    """The complete set of exportable ca.Functions for one Sim."""
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
    """Extract per-function ca.Functions from a Sim.

    The boxplus→jacobian→boxminus machinery is the shared
    `manta.linearization.Linearization` transform (the same one the EKF
    uses). This function does the C++-specific work around it: discover
    the tick's Input / Noise routing, then wrap Linearization's symbolic
    expressions into densified, world-name-prefixed `ca.Function`s with
    the names CasADi's CodeGenerator emits into C.

    Args:
        cw  — the manta Sim to extract from.

    Returns:
        A `WorldFunctions` bundle.
    """
    from ...world import Sim
    if not isinstance(cw, Sim):
        raise TypeError(
            f"extract: expected Sim, got {type(cw).__name__}")

    world = cw.world
    spec  = StateSpec.from_world(world)
    cf    = cw.tick.casadi_function

    # Index field disturbances by name for noise-routing.
    dist_by_name = _build_dist_lookup(world)

    # Walk the tick signature → Inputs (u) and Noise channels. The C++
    # predict zeroes noise (Σ/L are unused here), so a placeholder sigma
    # is fine — Linearization only needs full+dim to route + zero it.
    input_names = _discover_input_names(cf, world, spec, dist_by_name)
    noise_specs = _discover_noise_specs(cf, world, spec, dist_by_name)

    # Every Part Output is a candidate measurement.
    all_outputs = [f"{c.name}.{p.name}.{o}"
                   for c in world.crafts
                   for p in c.parts
                   for o in p.output_declarations()]

    lin = Linearization(
        cf, spec, frozen={}, input_names=input_names,
        noise_specs=noise_specs, outputs=all_outputs,
        build_functions=False)

    args = [lin.x_sym, lin.u_sym, lin.dt_sym, lin.t_sym]
    argn = ["x", "u", "dt", "t"]

    # predict (x_new, noise-zeroed) — dense by construction. F densified
    # for the flat row-major C interface.
    predict_fn = ca.Function(
        f"{world.name}_predict", args, [lin.x_new], argn, ["x_new"])
    predict_jacobian_fn = ca.Function(
        f"{world.name}_predict_jacobian", args, [ca.densify(lin.F_sym)],
        argn, ["F"])

    # --- per-Output (h, H) -----------------------------------------------
    outputs: list[OutputSpec] = []
    for craft in world.crafts:
        for part in craft.parts:
            for out_name in part.output_declarations():
                full = f"{craft.name}.{part.name}.{out_name}"
                o = lin.outputs[full]
                cname = f"{world.name}_h_{craft.name}_{part.name}_{out_name}"
                h_fn = ca.Function(
                    cname, args, [o.h_sym], argn, ["h"])
                H_fn = ca.Function(
                    cname + "_jacobian", args, [ca.densify(o.H_sym)],
                    argn, ["H"])
                outputs.append(OutputSpec(
                    craft_name=craft.name,
                    part_name=part.name,
                    output_name=out_name,
                    out_dim=o.dim,
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
    from ...fields.base import Disturbance
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


def _match_noise_driver(owner, sub: str):
    """Look up the noise channel whose driver-input name equals `sub`.
    Returns (nname, ndecl) on a hit, else None. Replaces the prior
    `isinstance(WhiteNoise) / sub.endswith('_driver') + isinstance(RW)`
    pair — each Noise subclass owns its own driver-name convention."""
    for nname, ndecl in owner.noise_declarations().items():
        if ndecl.driver_input_name(nname) == sub:
            return nname, ndecl
    return None


def _is_known_noise_input(owner, sub: str) -> bool:
    return _match_noise_driver(owner, sub) is not None


def _discover_noise_specs(cf: ca.Function,
                          world,
                          spec: StateSpec,
                          dist_by_name: dict) -> list[dict]:
    """Walk the tick signature; return a `{full, dim, sigma}` entry for
    every Noise channel. `sigma` is a placeholder (0.0): the C++ predict
    zeroes noise, so Σ/L are unused — Linearization needs only `full`
    and `dim` to route (and zero) the noise inputs. Inputs and state
    slots are skipped; `_discover_input_names` already validates them."""
    from ...craft import Craft as _Craft
    out: list[dict] = []
    for i in range(cf.n_in()):
        name = cf.name_in(i)
        if name in ("dt", "t") or name in spec:
            continue
        head, _, rest = name.partition(".")
        owner, _ = _resolve_owner(world, dist_by_name, head)
        if owner is None:
            continue
        if isinstance(owner, _Craft) and "." in rest:
            part_name, sub = rest.split(".", 1)
            part = next((p for p in owner.parts if p.name == part_name), None)
            target, key = part, sub
        else:
            target, key = owner, rest
        if target is None:
            continue
        hit = _match_noise_driver(target, key)
        if hit is not None:
            _, ndecl = hit
            out.append({"full": name,
                        "dim": ndecl.signal_manifold.ambient_dim,
                        "sigma": 0.0})
    return out
