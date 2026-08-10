"""manta — Python-first, CasADi-backed rigid-body sim + EKF.

Three layers:

  1. **Model** — declarative. Build a `World` containing `Craft`s,
     `Planet`s, `Coupling`s, and shared `Field`s with `Disturbance`s.
     Parts declare `Parameter` / `State` / `Input` / `Output` /
     `Noise` channels at class scope.

  2. **Transform** — compile to symbolic + emit the typed Module IR.
     `Sim(world)` linearizes the world tick; `EKF(world)` writes the
     Kalman recursion over it; `LQR(world, …)` solves Riccati. Each
     exposes `.module()` — a typed `Module` (state + kernels + entry
     points). None is directly callable.

  3. **Target** — lower a Module to a backend. `TargetNumpy(x)` returns
     the matching native-Python view over the one kernel engine (sim
     `.step(dt, u=…)`/`.reading()`, filter `.update()`/`.predict()`, …);
     `TargetCpp(x, …)` emits a typed C++ library for embedded use. You own
     the driving loop — the same shape in every backend.

Standard usage::

    from manta import World, Craft, Sim, EKF, TargetNumpy
    from manta.fields import GravityField
    from manta.parts import IMU, Mass, Thruster

    drone = Craft("drone")
    drone.add(Mass("body", mass=1.5))
    drone.add(Thruster("t", force=(0,0,1)))
    drone.add(IMU("imu", gyro_noise_sigma=0.005, gyro_bias_sigma=1e-4,
                  accel_noise_sigma=0.05))   # every EKF sensor needs σ>0

    w = World().add_field(GravityField(g=(0,0,-9.81)))
    w.add_craft(drone, position=(0,0,5))

    sim = TargetNumpy(Sim(w))
    ekf = TargetNumpy(EKF(w))

    for _ in range(N):
        sim.step(dt, t=t)                    # sim holds `sim.state`
        ekf.update("drone.imu.gyro",         # fold the reading...
                   sim.reading("drone.imu.gyro"))
        ekf.predict(dt, t=t)                 # ...then predict (you own order)
        t += dt

The low-level IR (`manta.ir`) is still exported for advanced use
(building bare CasADi graphs directly), but the typical user works
through the World/Target API above.
"""

__version__ = "0.1.0"   # keep in lockstep with pyproject.toml

from . import ir, smoothing
from .craft import Craft
from .world import World
from .sim import Sim
from .planets import Planet
from .couplings import Coupling
from .estimation.ekf import EKF
from .estimation.ukf import UKF
from .estimation.imu_integrator import IMUIntegrator
from .estimation.madgwick import Madgwick
from .estimation.mahony import Mahony
from .ir.state_spec import ALL, POSE, TWIST, SlotSet
from .control import LQR, LQRSolution, MPC, MpcPlan, PID
from .recurrence import RecurrenceBlock
from .codegen import NoiseDriver, TargetCpp, TargetJax, TargetNumpy, TargetWasm
from .fit import (Fit, FitResult, Free, NoiseFit, NoiseFitResult, Prior,
                  Tied, Window)
from .rates import CommandLatch, RateGate

__all__ = [
    "__version__",
    "ir", "smoothing", "Craft", "World", "Coupling", "Sim", "Planet",
    "EKF", "UKF",
    "LQR", "LQRSolution", "MPC", "MpcPlan", "PID",
    "Madgwick", "Mahony", "IMUIntegrator", "RecurrenceBlock",
    "TargetNumpy", "TargetCpp", "TargetWasm", "TargetJax", "NoiseDriver",
    "Fit", "FitResult", "Free", "NoiseFit", "NoiseFitResult", "Prior",
    "Tied", "Window",
    "RateGate", "CommandLatch",
    "SlotSet", "POSE", "TWIST", "ALL",
]
