"""State estimation — Error-State Kalman Filter + state layout.

Public surface:
  * `EKF`               — manifold-aware Error-State EKF over a World.
  * `measurement_slot`  — h_sym builder for "observe a single state slot".
  * `measurement_component` — h_sym builder for "observe one component
                              of a slot" (e.g. z position only).
Plus the recurrence attitude filters (`Madgwick`, `Mahony`,
`IMUIntegrator`) and the analysis tools (`observability`,
`sigma_horizon`, `nees`).

The state-layout types (`StateSpec`, `StateSlot`, `SlotSet`, …) live in
`manta.ir.state_spec` — they are IR, not estimation.
"""

from .ekf import EKF, measurement_slot, measurement_component
from .imu_integrator import IMUIntegrator
from .madgwick import Madgwick
from .mahony import Mahony
from .observability import (
    ObservabilityReport, SigmaHorizonReport, observability,
    observability_trajectory, sigma_horizon,
)
from .consistency import NEESReport, nees

__all__ = [
    "EKF", "measurement_slot", "measurement_component",
    "Madgwick", "Mahony", "IMUIntegrator",
    "observability", "observability_trajectory", "ObservabilityReport",
    "sigma_horizon", "SigmaHorizonReport",
    "nees", "NEESReport",
]
