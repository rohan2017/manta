# manta

A Python-first, CasADi-backed framework for small vehicles — drones,
rockets, underwater vehicles, satellites — spanning rigid-body
**simulation**, **state estimation** (EKF), and **control synthesis**
(LQR). You declare the craft and its physics; manta builds the symbolic
graph, runs it natively in Python, and (when you're ready) lowers it to
embedded C++ via codegen.

```python
import numpy as np
from manta import World, Craft, Sim, EKF, TargetNumpy
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

drone = Craft("drone")
drone.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
drone.add(Thruster("t", force=(0, 0, 1)))
drone.add(IMU("imu", gyro_noise_sigma=0.005, accel_noise_sigma=0.05))
drone.add(PositionSensor("gps", position_noise_sigma=0.02))

w = World().add_field(GravityField(g=(0, 0, -9.81)))
w.add_craft(drone, position=(0, 0, 5))

sim = TargetNumpy(Sim(w))      # truth model
ekf = TargetNumpy(EKF(w))      # estimator over the same world

sim.state["drone"]["t.throttle"] = 1.5 * 9.81   # hover
for t in np.arange(0, 3, 0.005):
    sim.step(0.005, t=t)
    reading = sim.outputs()
    ekf.update("imu.gyro", reading["drone"]["imu.gyro"])
    ekf.update("gps.position", reading["drone"]["gps.position"])
    ekf.predict(dt=0.005, t=t, u={"t.throttle": 1.5 * 9.81})

print(ekf.state_dict()["drone"]["position"])
```

## Where to go next

<div class="grid cards" markdown>

- **[Tutorials](tutorials/index.md)** — learn by building: a hovering
  drone, a closed-loop quadcopter, a camera interceptor, a system-ID fit.
- **[Explanation](explanation/architecture.md)** — how manta is put
  together: the three-layer pipeline, fields, manifolds, the error-state
  EKF, codegen.
- **[How-to](how-to/index.md)** — task recipes: deploy to C++, fit
  parameters, write a custom Part or Field.
- **[API reference](reference/index.md)** — the full public surface,
  generated from the source.

</div>

## Installation

manta is not yet published to PyPI. Install from a checkout:

```bash
pip install -e .                 # core: numpy + casadi
pip install -e ".[examples]"     # + matplotlib / scipy / rerun for the demos
pip install -e ".[jax]"          # + the TargetJax backend
pip install -e ".[docs]"         # + this documentation site
```

manta targets Python 3.12+.
