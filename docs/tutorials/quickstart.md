# Quickstart — a hovering drone

This tutorial builds a minimal but complete manta program: a single-craft
**truth simulation**, a matching **EKF** estimating its state, and the
loop that drives both. By the end you'll understand the
model → transform → target flow that every manta program follows.

## 1. Declare the model

A [`Craft`][manta.Craft] is a tree of parts. We give ours mass and
inertia, a vertical thruster, and two sensors:

```python
import numpy as np
from manta import World, Craft, Sim, EKF, TargetNumpy
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

drone = Craft("drone")
drone.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
drone.add(Thruster("t", force=(0, 0, 1)))            # body +z thrust
drone.add(IMU("imu", gyro_noise_sigma=0.005, accel_noise_sigma=0.05))
drone.add(PositionSensor("gps", position_noise_sigma=0.02))
```

The craft lives in a [`World`][manta.World] with a uniform gravity field,
positioned 5 m up:

```python
w = World().add_field(GravityField(g=(0, 0, -9.81)))
w.add_craft(drone, position=(0, 0, 5))
```

Nothing has executed yet — `w` is pure description.

## 2. Build the transforms and lower them

[`Sim(w)`][manta.Sim] is the truth model; [`EKF(w)`][manta.EKF] is an
estimator over the *same* world. Each is compiled once and lowered to a
native-Python runtime by [`TargetNumpy`][manta.TargetNumpy]:

```python
sim = TargetNumpy(Sim(w))
ekf = TargetNumpy(EKF(w))
```

## 3. Run the loop

The sim runtime holds the truth state — mutate `sim.state`, then step by
`dt`. Each step you read the sensors, **fold them into the EKF, then
predict** (the update-then-predict order is yours to keep: a reading
sampled at the interval start belongs against the pre-predict state).

```python
sim.state["drone"]["t.throttle"] = 1.5 * 9.81        # over-thrust → climb

for t in np.arange(0, 3, 0.005):
    sim.step(0.005, t=t)                             # advance truth
    reading = sim.outputs()                          # sensor readings
    ekf.update("imu.gyro", reading["drone"]["imu.gyro"])
    ekf.update("gps.position", reading["drone"]["gps.position"])
    ekf.predict(dt=0.005, t=t, u={"t.throttle": 1.5 * 9.81})

print(ekf.state_dict()["drone"]["position"])
```

`sim.outputs()` returns the readings nested by craft; `ekf.state_dict()`
returns the current estimate in the same nested shape.

!!! tip "You own the loop"
    manta never hides the driving loop. The exact same `update`/`predict`
    calls run against a `TargetCpp` build on an embedded target — see
    [Deploy a model to C++](../how-to/deploy-cpp.md).

## Next steps

- Close the loop with a regulator: [quadcopter tutorial](quadcopter.md).
- Understand what just happened:
  [the three-layer pipeline](../explanation/architecture.md).
