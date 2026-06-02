"""Per-function CasADi extraction from an `EKF` IR.

The EKF IR already carries the `ca.Function`s it needs (built by the shared
`Linearization`): predict, tangent `F`, noise gain `L`, and per-sensor
`h`/`H`. This module gathers them into the bundle the C++ EKF wrapper
consumes, doing three pieces of codegen-specific work:

  * **densify** every Jacobian output (`F`, `L`, `H`) and rename to a
    C-identifier-safe symbol — CasADi's CodeGenerator targets dense
    row/col-major C, and the Eigen wrapper reads `.data()`.
  * emit a **`boxplus` kernel** (`x_ambient, δ_tangent → x_ambient`) from
    `spec.boxplus_sym`, so the Kalman correction `x ⊞ Kδ` reuses the exact
    manifold math — no SO(3)/quaternion code is written in C++.
  * emit the **`L_h` noise-gain kernel** per sensor so the wrapper forms
    `R = L_h · Σ · L_hᵀ` at runtime — matching the numpy EKF exactly (R is
    state-dependent for sensors like the accelerometer, whose noise
    sensitivity rotates with attitude). `Σ`, the block partition, and a
    default `P0` are baked as constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ...estimation.state_spec import StateSpec


@dataclass(frozen=True)
class EkfSensorSpec:
    """One registered sensor, ready to codegen + fuse."""
    craft_name:  str
    part_name:   str
    output_name: str
    out_dim:     int
    h_fn:        ca.Function
    H_fn:        ca.Function
    L_h_fn:      ca.Function | None  # x,u,dt,t → ∂h/∂noise (dim × n_noise);
                                     # R = L_h·Σ·L_hᵀ formed at runtime

    @property
    def flat_name(self) -> str:
        return f"{self.craft_name}_{self.part_name}_{self.output_name}"

    @property
    def full_name(self) -> str:
        return f"{self.craft_name}.{self.part_name}.{self.output_name}"


@dataclass
class EkfFunctions:
    """Everything the C++ EKF wrapper needs from one `EKF` IR."""
    world_name:   str
    spec:         StateSpec
    input_names:  list[str]
    predict_fn:   ca.Function       # x, u, dt, t → x_new (ambient)
    F_fn:         ca.Function       # x, u, dt, t → F (tangent²)
    L_fn:         ca.Function | None  # x, u, dt, t → L (tangent × n_noise)
    boxplus_fn:   ca.Function       # x_ambient, δ_tangent → x_ambient
    Sigma:        np.ndarray        # process-noise covariance (n_noise²)
    P0:           np.ndarray        # default initial covariance (tangent²)
    blocks:       list[list[int]]   # independent-subsystem tangent partition
    sensors:      list[EkfSensorSpec]

    @property
    def ambient_dim(self) -> int:
        return self.spec.ambient_dim

    @property
    def tangent_dim(self) -> int:
        return self.spec.tangent_dim

    @property
    def n_inputs(self) -> int:
        return len(self.input_names)


def _densify(fn: ca.Function, name: str) -> ca.Function:
    """Rewrap a `ca.Function` so its outputs are dense (row/col-major fill
    for the flat-C / Eigen interface) and rename to a codegen-safe symbol.
    Dense outputs (state/measurement vectors) pass through unchanged."""
    ins = [ca.MX.sym(fn.name_in(i), fn.size1_in(i), fn.size2_in(i))
           for i in range(fn.n_in())]
    outs = fn(*ins)
    if fn.n_out() == 1:
        outs = [outs]
    outs = [ca.densify(o) for o in outs]
    return ca.Function(name, ins, outs,
                       [fn.name_in(i) for i in range(fn.n_in())],
                       [fn.name_out(i) for i in range(fn.n_out())])


def extract_ekf(ekf) -> EkfFunctions:
    """Extract the codegen function bundle from an `EKF` IR.

    Args:
        ekf — the `EKF` (returned by `EKF(world)`).
    """
    from ...estimation.ekf import EKF
    if not isinstance(ekf, EKF):
        raise TypeError(f"extract_ekf: expected EKF, got {type(ekf).__name__}")

    world = ekf.world
    spec  = ekf.spec
    wn    = world.name

    predict_fn = _densify(ekf._f_fn, f"{wn}_ekf_predict")
    F_fn       = _densify(ekf._F_fn, f"{wn}_ekf_F")
    L_fn       = (_densify(ekf._L_fn, f"{wn}_ekf_L")
                  if ekf._L_fn is not None else None)

    # boxplus kernel: x ⊞ δ, reusing the spec's manifold-correct symbolic op.
    x_bp = ca.MX.sym("x", spec.ambient_dim)
    d_bp = ca.MX.sym("delta", spec.tangent_dim)
    boxplus_fn = ca.Function(
        f"{wn}_ekf_boxplus", [x_bp, d_bp], [spec.boxplus_sym(x_bp, d_bp)],
        ["x", "delta"], ["x_new"])

    Sigma = (np.asarray(ekf._Sigma, dtype=float)
             if ekf._Sigma is not None else np.zeros((0, 0)))

    sensors: list[EkfSensorSpec] = []
    for spec_o in ekf._sensors.values():
        craft, part, out_name = spec_o["craft"].name, spec_o["part"].name, \
            spec_o["full"].rsplit(".", 1)[-1]
        safe = spec_o["full"].replace(".", "_")
        L_h_fn = (_densify(spec_o["L_h_fn"], f"{wn}_ekf_Lh_{safe}")
                  if spec_o["L_h_fn"] is not None and Sigma.size else None)
        sensors.append(EkfSensorSpec(
            craft_name=craft, part_name=part, output_name=out_name,
            out_dim=spec_o["dim"],
            h_fn=_densify(spec_o["h_fn"], f"{wn}_ekf_h_{safe}"),
            H_fn=_densify(spec_o["H_fn"], f"{wn}_ekf_H_{safe}"),
            L_h_fn=L_h_fn))

    blocks = [list(map(int, np.asarray(b).ravel())) for b in ekf._blocks]
    P0 = np.eye(spec.tangent_dim) * 1e-2

    return EkfFunctions(
        world_name=wn, spec=spec, input_names=list(ekf._input_names),
        predict_fn=predict_fn, F_fn=F_fn, L_fn=L_fn, boxplus_fn=boxplus_fn,
        Sigma=Sigma, P0=P0, blocks=blocks, sensors=sensors)
