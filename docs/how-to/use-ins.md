# INS with your own sensor models

The hardware integration skeleton lives in the separate
[INS handoff repository](https://github.com/Ulysses-Ecosystem-Engineering/ins-handoff):

- `models.py`: editable body, IMU, DVL, GPS and barometer equations.
- `run_ins.py`: driver hooks, the ordinary INS runtime and C preintegrator export.

Install its pinned Manta dependency, implement the initial-state and measurement
driver hooks, and run `python run_ins.py`. It requires real acquisitions and an
explicit initial state; no simulated data or assumed heading is supplied.

## The same filter API

```python
from manta import EKF, UKF, INS, TargetNumpy

ekf = TargetNumpy(EKF(world, sensors=aiding_outputs))
ukf = TargetNumpy(UKF(world, sensors=aiding_outputs))
ins = TargetNumpy(INS(world, imu="imu", sensors=aiding_outputs))
```

All expose `reset`, `predict`, `update`, `state_dict`, `x`, `P`, and checkpoints.
INS uses its selected IMU for strapdown; EKF/UKF use declared vehicle dynamics.
EKF and UKF also differ in how they approximate uncertainty; the INS is an
error-state EKF with an IMU-driven prediction. Sensor models and the
`reset`/`predict`/`update` pattern are shared, not a separate navigation API.
A custom IMU need not inherit the built-in model. It declares `gyro` and
`accel` outputs plus three-axis `gyro_noise` and `accel_noise` white channels
in `PartFrame`. The contract is calibrated inertial rate (rad/s) and specific
force (m/s²), with optional additive `gyro_bias` / `accel_bias` states.

For point samples or remote preintegration, the handoff explicitly selects:

```python
ins = TargetNumpy(INS(world, imu="imu", sensors=aiding_outputs,
    propagation="preintegrated", accel_integration="endpoint",
    covariance="conventional_reset"))
```

This is an error-state EKF with quaternion nominal attitude, additive bias
errors, modified velocity error coordinates, Joseph updates, and differential
covariance reset. Low-level compatibility defaults remain `raw`, `left_hold`,
and `linearized`; advanced covariance modes remain explicit. Set a credible
initial physical covariance using `reset(P=...)`, as the handoff does.

## Global coordinates, without reframing

With no `navigation_frame` override, **all three filters use Manta's global,
inertial `WorldFrame`**, including position, velocity, and orientation. The
planet supplies spatial gravity and its rotation. These Cartesian coordinates
and quaternions have no dateline or polar coordinate singularity. There is no
local origin to move when travelling around the planet.
For reporting, `planet.world_to_planet(position, velocity, t=ins.time)` returns
planet-fixed position and ground-relative velocity without changing the filter.

The sensor equations handle reference frames:

- GPS predicts planet-fixed Cartesian position: `R_planet_from_world(t) @ (p-center)`. On Earth this is ECEF. Convert receiver latitude/longitude/height into those Cartesian metres before updating.
- Bottom-track DVL predicts `R_sensor_from_world @ (v - omega_planet × (p-center))`, including mount kinematics supplied by Manta. Inertial velocity is not ground velocity.
- Barometer predicts the planet fluid field's absolute pressure in pascals.
- IMU measures inertial rotation and specific force in its own axes. Do not subtract Earth rate before strapdown.

Use one synchronized clock for the filter, packet timestamps, and planet's
rotation model. `reset(..., t=start_time)` sets the initial state's epoch.
A stationary instrument still moves and rotates in inertial coordinates;
gyrocompassing emerges from gravity, the planet-relative aiding models, and
attitude/bias cross-covariances. Bias calibration/prior and excitation still
matter. Global coordinates remove reframing, not accumulated sensor drift.

Gyrocompassing is inference from the planet's rotation and gravity together
with suitable observations or motion constraints. It is not an unconditional
"find north" switch or unique to the INS class: a dynamics EKF/UKF can also
infer heading when its IMU measurement model makes rotation observable.
The INS routes that information through strapdown and aiding corrections.
No magnetometer is required, but unknown gyro bias, poor IMU precision and
insufficient constraints can prevent useful alignment. The handoff still
requires an initial attitude estimate/prior; it contains no cold-start aligner.

## Explicit driver loop

Construct `TargetNumpy(INS(...))` directly. The driver keeps adjacent IMU
acquisitions, runs `IMUPreintegrator`, closes an interval at a chosen prediction
or aiding boundary, and calls `predict(packet)`. Corrections use the same
`update` method as EKF/UKF. No scheduler is hidden in the estimator. The handoff
shows the whole loop; received MCU packets skip host preintegration entirely.

Use the planet's gravity in the global inertial World for the default path.
An explicit `NavigationFrame` remains supported for separately authored
rotating-coordinate models, with gravity conventions owned by that model.

## Orientation math and model authoring

Use `manta.types` for symbolic Part equations and
`scipy.spatial.transform.Rotation` for numerical orientation work in drivers,
initialization and output. `manta.ir._rotation` is private and should not be
imported by applications. See [model authoring types](../reference/types.md)
for the tested quaternion conventions and examples. SciPy is not required by
Manta's estimator kernels and cannot replace symbolic algebra inside a Part.

## Preintegrate on an MCU, predict on the host

```python
from manta import PreintegratedIMUPacket

# Same process: construct from IMUPreintegrator runtime readouts.
packet = PreintegratedIMUPacket(readouts, start_time=t0)
# Remote alternative: the driver delivers one complete opaque payload.
packet = PreintegratedIMUPacket.from_bytes(payload)

inputs = ins.preintegrated_inputs(packet)
ins.predict(packet)                         # advances to packet.end_time
ins.update("dvl.velocity", measured_dvl, u=inputs)
```

`u=inputs` preserves the endpoint gyro/accelerometer and their shared noise
when an aiding model needs them (lever arms or ModelForce). It is the same
explicit input mechanism used by EKF/UKF. Measurements must be at the current
state epoch; delayed observations require application-level replay.

Export the preintegrator and standard binary encoder:

```sh
# From the INS handoff checkout:
python run_ins.py --export-preintegrator /tmp/imu-firmware
```

The output contains standalone **C numerical kernels**, a C++ convenience
wrapper, `imu_preintegrator.h` for C99 state/workspace bookkeeping, and
`imu_packet.h`, containing `manta_imu_packet_encode`. Firmware encodes the
readouts into an opaque byte payload; the host calls
`PreintegratedIMUPacket.from_bytes(payload)`. Your driver owns the transport
and reassembles complete payloads, not internal field packing. The named C
struct remains available for custom protocols.
The C kernels and codec do not require Python, CasADi at runtime, or Eigen;
the optional C++ wrapper uses Eigen. See [the packet contract](../explanation/imu-packet-format.md)
for kernel calls, ordering, units, and firmware responsibilities.

Model-force aiding is optional and off in the handoff. For underwater use,
fit dynamics on submerged data and apply force updates only while submerged
and the model is valid. Sensor noise is characterized separately. Broad
attitude/bias posterior consistency remains a known limitation; the integration skeleton
and coordinate tests do not qualify every alignment or dropout regime.
