"""manta — Python-first, CasADi-backed vehicle modeling and estimation.

Three layers:

  1. **Model** — declarative. Build a `World` containing `Craft`s,
     `Planet`s, `Coupling`s, and shared `Field`s with `Disturbance`s.
     Parts declare `Parameter` / `State` / `Input` / `Output` /
     `Noise` channels at class scope.

  2. **Transform** — compile to symbolic + emit the typed Module IR.
     `Sim(world)` compiles the world tick; `EKF(world)` / `UKF(world)`
     propagate it as dynamics-driven filters; `INS(world, imu=…)` uses
     IMU strapdown propagation; `LQR(world, …)` solves Riccati. Each
     exposes `.module()` — a typed `Module` (state + kernels + entry
     points). None is directly callable.

  3. **Target** — lower a Module to a backend. `TargetNumpy(x)` returns
     the matching native-Python view over the one kernel engine (sim
     `.step(dt, u=…)`/`.outputs()` for all readings or `.reading(name)` for
     one sensor, filter `.update()`/`.predict()`, …);
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

from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("mantapilot")

from . import ir, smoothing
from .codegen import (
    CompilationError,
    FilterCheckpoint,
    FilterReplayProgram,
    FilterReplayResult,
    NativeFilterReplay,
    NoiseCheckpoint,
    NoiseDriver,
    ReplayBoundary,
    ReplayCheckpointResult,
    ReplayOperation,
    ReplayPredict,
    ReplayUpdate,
    SimCheckpoint,
    TargetCpp,
    TargetFilterReplay,
    TargetJax,
    TargetNumpy,
    TargetWasm,
    UpdateResult,
    compile_functions,
)
from .control import (
    LQR,
    MPC,
    PID,
    CraftHorizonReference,
    LQRSolution,
    MPCReference,
    MPCResult,
    MPCTimings,
)
from .couplings import Coupling
from .craft import Craft
from .estimation.ekf import EKF
from .estimation.imu_integrator import IMUIntegrator
from .estimation.ins import INS
from .estimation.madgwick import Madgwick
from .estimation.mahony import Mahony
from .estimation.ukf import UKF
from .fit import (
    Fit,
    FitDerivationReport,
    FitResult,
    Free,
    NoiseFit,
    NoiseFitResult,
    Prior,
    Tied,
    Window,
)
from .ir.state_spec import ALL, POSE, TWIST, SlotSet
from .model import ModelArtifact, ModelValidationReport
from .model_layout import state_spec_from_craft, state_spec_from_world
from .planets import Planet
from .recurrence import RecurrenceBlock
from .sim import Sim
from .world import World

__all__ = [
    "ALL",
    "EKF",
    "INS",
    "LQR",
    "MPC",
    "PID",
    "POSE",
    "TWIST",
    "UKF",
    "CompilationError",
    "Coupling",
    "Craft",
    "CraftHorizonReference",
    "FilterCheckpoint",
    "FilterReplayProgram",
    "FilterReplayResult",
    "Fit",
    "FitDerivationReport",
    "FitResult",
    "Free",
    "IMUIntegrator",
    "LQRSolution",
    "MPCReference",
    "MPCResult",
    "MPCTimings",
    "Madgwick",
    "Mahony",
    "ModelArtifact",
    "ModelValidationReport",
    "NativeFilterReplay",
    "NoiseCheckpoint",
    "NoiseDriver",
    "NoiseFit",
    "NoiseFitResult",
    "Planet",
    "Prior",
    "RecurrenceBlock",
    "ReplayBoundary",
    "ReplayCheckpointResult",
    "ReplayOperation",
    "ReplayPredict",
    "ReplayUpdate",
    "Sim",
    "SimCheckpoint",
    "SlotSet",
    "TargetCpp",
    "TargetFilterReplay",
    "TargetJax",
    "TargetNumpy",
    "TargetWasm",
    "Tied",
    "UpdateResult",
    "Window",
    "World",
    "__version__",
    "compile_functions",
    "ir",
    "smoothing",
    "state_spec_from_craft",
    "state_spec_from_world",
]
