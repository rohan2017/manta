# Use the INS with your own model

Manta's INS needs only Manta, NumPy, and CasADi. It does not need Shiver,
Mako, a simulator, a controller, or model-force aiding. Install this repository
with Python 3.12 or newer (`python -m pip install .`).

The complete [standalone example](https://github.com/rohan2017/manta/blob/main/examples/vehicles/ins_standalone.py)
defines its own craft and feeds synthetic stationary measurements:

```sh
python -m examples.vehicles.ins_standalone --duration 120
# Optional C compiler; compiles and caches kernels before running:
python -m examples.vehicles.ins_standalone --duration 120 --native
```

## Build and initialize

Define a `Craft` with mass/inertia, an `IMU` or `ConstantBiasIMU`, and the
aiding sensors you use. Set mounting transforms and measurement noise in
the sensor models. No hull, thruster, or hydrodynamic model is required.
`ConstantBiasIMU` retains estimable constant gyro/accelerometer biases;
use `IMU` with measured nonzero bias random-walk densities for drifting biases.

Put the craft in a local estimator `World` with an explicit `GravityField`.
For gyrocompassing, derive gravity and rotation from the same planet:

```python
from manta import INSStream, NavigationEnvironment, World

# planet is Earth() or your own Planet subclass. anchor is Cartesian
# PlanetFrame position in metres, not latitude/longitude.
environment = NavigationEnvironment.from_scene(
    planet.scene_at(anchor), frame_id="dive_site", epoch="t=0")
world = World().add_field(environment.gravity)
world.add_craft(my_craft)
ins = INSStream(world, imu="imu", sample_rate_hz=100, predict_every=2,
                navigation_frame=environment.frame,
                sensors=["dvl.velocity", "gps.position"])
```

All positions/velocities, sensor references, and other environmental fields
in this estimator World use those local axes. Default scene axes are north,
west, up; `heading=-pi/2` selects east, north, up. Do not put inertial
planet-scale simulation coordinates into this World. The helper uses the
planet's registered gravity and rotation, preserves spatial gravity variation,
and includes centrifugal acceleration exactly once. Rebuild after changing
the planet or anchor. A manual `NavigationFrame` and gravity field remain
available for other coordinate conventions.

Call `ins.reset(initial_state, P=P)` with your initial pose/velocity/bias mean
and physical tangent covariance. The example builds `P` by named slots using
`ins.filter.module.port("prior_x").spec`; quaternions are scalar-first
`[w,x,y,z]`, while attitude uncertainty is a three-component rotation vector
in radians. Internal shared-endpoint noise states are initialized by Manta.
Do not use arbitrary defaults as a calibrated startup prior.

## Feed measurements

```python
ins.update_imu(accel_m_s2, gyro_rad_s, t=timestamp_seconds)
# At an actual IMU endpoint, when a new aiding observation arrives:
result = ins.update("dvl.velocity", velocity_in_sensor_axes, t=timestamp_seconds)
# GPS Cartesian position must use the estimator's navigation frame.
ins.update("gps.position", position_in_navigation_axes, t=timestamp_seconds)
ins.flush()                    # finish a partial packet at end of log
state = ins.state_dict()["vehicle"]
```

Supply paired, increasing-timestamp **point acquisitions** in IMU axes:
accelerometer specific force (approximately +g upward at rest) and **inertial**
gyro rate, including planet rotation. Do not subtract planet rate yourself.
The first pair establishes a boundary; a second is required before prediction
or aiding. Aiding must coincide with the latest observed IMU endpoint and
flushes any partial packet. The driver rejects stale/future aiding; it does
not interpolate, hold missing samples, or perform delayed-data replay.
`ins.time` identifies the estimated-state epoch, which can lag the latest
sample while a packet is pending.

Sensor white-noise sigmas are **per acquisition**, in rad/s and m/s². For
gyro ARW `a` in degrees/√hour at sample rate `f`, use
`gyro_noise_sigma = deg2rad(a) / 60 * sqrt(f)` under Manta's independent
sample convention. Bias prior standard deviation in degrees/hour converts
with `deg2rad(b)/3600`; it is not a random-walk density. Filtered or correlated
hardware measurements need an appropriate acquisition/noise model.

## Filter and gyrocompassing

`INSStream` explicitly selects the single-Gaussian error-state EKF with
`covariance="conventional"`, preintegrated endpoint gyro **and** acceleration,
quaternion nominal attitude, additive bias errors, modified velocity error
coordinates, and Joseph measurement corrections. This mode retains covariance
on feedback as a local approximation; `covariance="conventional_reset"`
selects the differential reset. Neither is a general broad-prior alignment
solution. The low-level `INS` defaults remain `linearized/raw/left_hold` for
compatibility; choose those options explicitly when comparing modes.

Gravity and planet rotation are nonparallel reference vectors away from the
poles. A heading error produces propagation errors that aiding observations
constrain; Kalman cross-covariances then correct attitude and biases. There
is no hidden magnetometer or separate gyro-to-north update. Constant gyro bias
can imitate a heading error, so gyrocompassing needs a credible bias prior,
calibration, or informative motion. A nonrotating planet or a polar location
does not provide this heading reference.

Tests cover integration, acquisition covariance, coordinate transformations,
and native/interpreted parity. The broad attitude/gyro-bias/accelerometer-bias
joint posterior still has known statistical-consistency limitations. Do not
interpret a small heading error or these tests as universal covariance
qualification. Validate the intended sensor, prior, motion, and aiding regime.

## Optional model aiding and native kernels

Model aiding adds a calibrated `ModelForce` observation to the estimator model.
It does **not** replace IMU navigation propagation. Fit/reduce vehicle dynamics
using representative **submerged** data, validate on held-out submerged data,
and supply the required accepted fit evidence to `ModelForce`. Sensor noise
comes from sensor characterization, not a reduced-order sensor fit.
Use `fit_result.evidence(held_out, sensor="imu.accel",
known_white_sigma=imu.accel_noise_sigma)` to keep known acquisition variance
out of the fitted correlated dynamics error.

The stream never applies force aiding automatically. When submerged and the
model and achieved controls are valid, call
`ins.update("model_force.specific_force", accel_sample, t=t, u=controls)` once
at the matching endpoint. Use the same raw acceleration acquisition as
`update_imu`; Manta retains its shared prediction/measurement uncertainty.
Pass current controls to prediction/update and flush at an observed boundary
before changing controls within a packet. During surface operation, omit
force updates. Healthy DVL can make force aiding largely redundant; model
mismatch can make it harmful.

For repeated use, call `ins.compile(sensors=["dvl.velocity", "gps.position"])`
before the live loop. It defaults to O1 and a 20,000-instruction compilation
limit. If passing per-reading `R` to updates, compile with
`covariance="per_sample"`. Compilation errors propagate; there is no silent
fallback. Measure latency on your own model and computer. The underlying
`ins.model`, `ins.filter`, and `ins.preintegrator` expose the ordinary Manta
transform/runtime APIs for custom scheduling or code generation.
