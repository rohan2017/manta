"""State estimation — Error-State Kalman Filters + state layout.

Public surface:
  * `EKF`               — manifold-aware Error-State EKF over a World.
  * `UKF`               — the unscented twin (sigma-point predict/update);
                          same constructor, runtime surface, and Module.
  * `INS`               — IMU-strapdown error-state filter with the same
                          runtime and analysis surface.
  * `measurement_slot`  — h_sym builder for "observe a single state slot".
  * `measurement_component` — h_sym builder for "observe one component
                              of a slot" (e.g. z position only).
Plus the recurrence attitude filters (`Madgwick`, `Mahony`,
`IMUIntegrator`, `IMUPreintegrator`) and the analysis tools (`observability`,
`sigma_horizon`, `nees`).

The state-layout types (`StateSpec`, `StateSlot`, `SlotSet`, …) live in
`manta.ir.state_spec` — they are IR, not estimation.
"""

from .consistency import NEESReport, nees
from .ekf import EKF, measurement_component, measurement_slot
from .imu_integrator import IMUIntegrator
from .imu_preintegrator import IMUPreintegrator
from .madgwick import Madgwick
from .mahony import Mahony
from .observability import (
    ObservabilityReport,
    SigmaHorizonReport,
    observability,
    observability_trajectory,
    sigma_horizon,
)
from .ukf import UKF

__all__ = [
    "EKF",
    "INS",
    "UKF",
    "IMUIntegrator",
    "IMUPreintegrator",
    "Madgwick",
    "Mahony",
    "NEESReport",
    "ObservabilityReport",
    "SigmaHorizonReport",
    "measurement_component",
    "measurement_slot",
    "nees",
    "observability",
    "observability_trajectory",
    "sigma_horizon",
]


def __getattr__(name):
    # INS imports physical Part/Field types. Keep that heavier edge lazy so
    # codegen's import of estimation._kalman cannot close
    # codegen -> estimation -> fields -> parts -> fields during startup.
    if name == "INS":
        from .ins import INS
        return INS
    raise AttributeError(name)
