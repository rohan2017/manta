"""State estimation — Error-State Kalman Filter + state layout.

Public surface:
  * `EKF`               — manifold-aware Error-State EKF on a Craft.
  * `measurement_slot`  — h_sym builder for "observe a single state slot".
  * `measurement_component` — h_sym builder for "observe one component
                              of a slot" (e.g. z position only).
  * `StateSpec`         — flat-vector layout of the Craft's state, with
                           boxplus/boxminus dispatched per slot.

The single-craft EKF is what's shipped today. Multi-craft StateSpec and
a UKF variant land in follow-up milestones.
"""

from .ekf import EKF, measurement_slot, measurement_component
from .state_spec import StateSlot, StateSpec

__all__ = [
    "EKF", "measurement_slot", "measurement_component",
    "StateSlot", "StateSpec",
]
