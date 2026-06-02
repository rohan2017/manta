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

The tick's named I/O is classified by the shared
`manta.tick_signature.walk_tick_signature` (Inputs → u, Noise channels,
candidate sensors); the Jacobian machinery is the shared
`manta.linearization.Linearization`. This module only wraps
Linearization's symbolic exprs into densified, world-name-prefixed
`ca.Function`s with the names CasADi's CodeGenerator emits into C.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from ...estimation.state_spec import StateSpec
from ...linearization import Linearization
from ...tick import walk_tick_signature


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
    from ...sim import Sim
    if not isinstance(cw, Sim):
        raise TypeError(
            f"extract: expected Sim, got {type(cw).__name__}")

    world = cw.world
    spec  = StateSpec.from_world(world)
    cf    = cw.tick.casadi_function

    # Classify the tick's I/O (Inputs, Noise, candidate sensors) — the
    # shared tick-signature walker. The C++ predict zeroes noise (Σ/L are
    # unused here), so the channels' sigma is ignored; Linearization needs
    # only full+dim to route + zero them.
    sig = walk_tick_signature(cf, world, spec)

    lin = Linearization(
        cf, spec, frozen={}, input_names=sig.input_names,
        noise_specs=sig.noise, outputs=sig.sensor_names,
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
    for s in sig.sensors:
        o = lin.outputs[s.full]
        cname = f"{world.name}_h_{s.craft.name}_{s.part.name}_{s.output_name}"
        h_fn = ca.Function(cname, args, [o.h_sym], argn, ["h"])
        H_fn = ca.Function(
            cname + "_jacobian", args, [ca.densify(o.H_sym)], argn, ["H"])
        outputs.append(OutputSpec(
            craft_name=s.craft.name,
            part_name=s.part.name,
            output_name=s.output_name,
            out_dim=o.dim,
            h_fn=h_fn,
            H_fn=H_fn,
        ))

    return WorldFunctions(
        world_name=world.name,
        spec=spec,
        input_names=sig.input_names,
        predict_fn=predict_fn,
        predict_jacobian_fn=predict_jacobian_fn,
        outputs=outputs,
    )
