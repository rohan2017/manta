"""manta.control — controller synthesis transforms over the model.

Siblings of `Sim(world)` / `EKF(world)`: each consumes the model and the
shared `manta.linearization.Linearization` seam, and lowers to a backend
runtime via `Target*`.

  * `LQR(world, ...)` — infinite-horizon discrete LQR about an operating
    point; a baked state-feedback control law.
  * `MPC(world, ...)` — receding-horizon point-to-point MPC; the
    fixed-work RTI tick over the compiled world step, warm plan as
    HELD Module state.
  * `PID(kp, ki, kd, ...)` — a scalar PID controller; a freestanding
    `recurrence` block (no world needed), lowered by the same backends.
"""

from .lqr import LQR, LQRSolution
from .mpc import MPC, MpcPlan
from .pid import PID

__all__ = ["LQR", "LQRSolution", "MPC", "MpcPlan", "PID"]
