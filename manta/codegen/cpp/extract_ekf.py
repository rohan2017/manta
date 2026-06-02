"""Per-function CasADi extraction from an `EKF` IR.

The `EKF` IR carries the *whole* Kalman recursion as fused `ca.Function`s
(built by `Linearization.kalman_functions`): a predict step
`(x,P,Q,u,dt,t) → (x',P')`, a process-noise kernel `(x,u,dt,t) → Q`, and a
per-sensor Joseph update `(x,P,z,u,t) → (x',P')`. This module just densifies
them (CasADi's CodeGenerator targets dense col-major C; the Eigen wrapper
reads `.data()`) and renames to C-identifier-safe symbols. **No linear
algebra is extracted or re-derived** — the recursion already lives in the
kernels, so the C++ wrapper is pure pack/call/unpack.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ._casadi import densify as _densify
from ...estimation.state_spec import StateSpec


@dataclass(frozen=True)
class EkfSensorSpec:
    """One registered sensor + its baked Joseph-update kernel."""
    craft_name:  str
    part_name:   str
    output_name: str
    out_dim:     int
    update_fn:   ca.Function       # x, P, z, u, t → (x_new, P_new)

    @property
    def flat_name(self) -> str:
        return f"{self.craft_name}_{self.part_name}_{self.output_name}"

    @property
    def full_name(self) -> str:
        return f"{self.craft_name}.{self.part_name}.{self.output_name}"


@dataclass
class EkfFunctions:
    """Everything the C++ EKF wrapper needs from one `EKF` IR."""
    world_name:       str
    spec:             StateSpec
    input_names:      list[str]
    predict_fn:       ca.Function       # x, P, Q, u, dt, t → (x_new, P_new)
    process_noise_fn: ca.Function | None  # x, u, dt, t → Q  (None ⇒ no noise)
    P0:               np.ndarray        # default initial covariance (tangent²)
    sensors:          list[EkfSensorSpec]

    @property
    def ambient_dim(self) -> int:
        return self.spec.ambient_dim

    @property
    def tangent_dim(self) -> int:
        return self.spec.tangent_dim

    @property
    def n_inputs(self) -> int:
        return len(self.input_names)


def extract_ekf(ekf) -> EkfFunctions:
    """Densify the EKF IR's Kalman kernels into the C++ wrapper bundle."""
    from ...estimation.ekf import EKF
    if not isinstance(ekf, EKF):
        raise TypeError(f"extract_ekf: expected EKF, got {type(ekf).__name__}")

    spec = ekf.spec
    wn   = ekf.world.name

    predict_fn = _densify(ekf._predict_fn, f"{wn}_ekf_predict")
    process_noise_fn = (
        _densify(ekf._process_noise_fn, f"{wn}_ekf_process_noise")
        if ekf._process_noise_fn is not None else None)

    sensors: list[EkfSensorSpec] = []
    for spec_o in ekf._sensors.values():
        craft, part, out_name = spec_o["craft"].name, spec_o["part"].name, \
            spec_o["full"].rsplit(".", 1)[-1]
        safe = spec_o["full"].replace(".", "_")
        sensors.append(EkfSensorSpec(
            craft_name=craft, part_name=part, output_name=out_name,
            out_dim=spec_o["dim"],
            update_fn=_densify(spec_o["update_fn"], f"{wn}_ekf_update_{safe}")))

    P0 = np.eye(spec.tangent_dim) * 1e-2
    return EkfFunctions(
        world_name=wn, spec=spec, input_names=list(ekf._input_names),
        predict_fn=predict_fn, process_noise_fn=process_noise_fn,
        P0=P0, sensors=sensors)
