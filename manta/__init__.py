"""manta — Python-first, CasADi-backed rigid-body sim + EKF.

Three layers:

  1. **Model** — declarative. Build a `World` containing `Craft`s,
     `Planet`s, `Coupling`s, and shared `Field`s with `Disturbance`s.
     Parts declare `Parameter` / `State` / `Input` / `Output` /
     `Noise` channels at class scope.

  2. **IR** — compile to symbolic. `Sim(world)` returns a
     `Sim` (one CasADi tick function over every craft +
     every coupling). `EKF(world)` returns an EKF carrying the
     symbolic predict + auto-built per-sensor measurement bundles.
     Neither is directly callable.

  3. **Target** — lower IR to a backend. `TargetNumpy(cw)` returns
     a `NumpyWorld` you `.step()`; `TargetNumpy(ekf)` returns a
     `NumpyEKF` you `.predict()` / `.update()`. Future
     `TargetCpp(...)` will emit C++ for embedded use.

Standard usage::

    from manta import World, Craft, EKF, TargetNumpy
    from manta.fields import GravityField
    from manta.parts import IMU, Mass, Thruster

    drone = Craft("drone")
    drone.add(Mass("body", mass=1.5))
    drone.add(Thruster("t", force=(0,0,1)))
    drone.add(IMU("imu", gyro_noise_sigma=0.005, gyro_bias_sigma=1e-4))

    w = World().add_field(GravityField(g=(0,0,-9.81)))
    w.add_craft(drone, position=(0,0,5))

    sim = TargetNumpy(Sim(w))
    ekf = TargetNumpy(EKF(w))

    state = sim.initial_state()
    for _ in range(N):
        state = sim.step(state, dt=dt, t=t)
        ekf.predict(dt=dt, t=t)
        ekf.update(drone.parts[-1], gyro=state["drone"]["imu.gyro"])
        t += dt

The low-level IR (`manta.ir`) is still exported for advanced use
(building bare CasADi graphs directly), but the typical user works
through the World/Target API above.
"""

from . import ir
from .craft import Craft
from .world import World
from .sim import Sim
from .planets import Planet
from .couplings import Coupling
from .estimation.ekf import EKF
from .estimation.state_spec import ALL, POSE, TWIST, SlotSet
from .control import LQR, PID
from .recurrence import RecurrenceBlock
from .codegen import NoiseDriver, TargetCpp, TargetNumpy
from .signal import Signal, wire

__all__ = [
    "ir", "Craft", "World", "Coupling", "Sim", "Planet", "EKF", "LQR", "PID",
    "RecurrenceBlock",
    "TargetNumpy", "TargetCpp", "NoiseDriver", "Signal", "wire",
    "SlotSet", "POSE", "TWIST", "ALL",
]
