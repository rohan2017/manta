"""manta.control — controller synthesis transforms over the model.

Siblings of `Sim(world)` / `EKF(world)`: each consumes the model and the
shared `manta.linearization.Linearization` seam, and lowers to a backend
runtime via `Target*`.

  * `LQR(world, ...)` — infinite-horizon discrete LQR about an operating
    point; a baked state-feedback control law.
  * `PID(kp, ki, kd, ...)` — a scalar PID controller; a freestanding
    `recurrence` block (no world needed), lowered by the same backends.
"""

from .lqr import LQR, LQRSolution
from .pid import PID

__all__ = ["LQR", "LQRSolution", "PID"]
