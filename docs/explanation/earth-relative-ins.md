# Fixed planet-attached strapdown INS

This branch implements the frame mechanics and an executable qualification
suite. **It is a mechanics-only draft, not a qualified gyrocompassing release.**
The synthetic weak-bias and noisy-gyro cases fail covariance consistency. See
[qualification evidence](../qualification/earth-relative-ins.md) before using
these changes in an estimator artifact.

## Construction and ownership

```python
from manta import INS, NavigationFrame

frame = NavigationFrame(
    frame_id="deployment/site-enu", epoch="1",
    angular_velocity=planet_rotation_in_navigation_axes,
    origin_from_rotation_center=planet_center_to_anchor_in_navigation_axes,
    gravity_convention="effective",
)
ins = INS(local_world, imu="craft.imu", navigation_frame=frame)
packet_ins = INS(local_world, imu="craft.imu", navigation_frame=frame,
                 propagation="preintegrated")
```

Both vectors are expressed in the named navigation axes. The origin vector
runs from the planet's rotation center to the navigation origin. The caller
owns the actual Cartesian/geodetic transform, its epoch, and the accuracy
budget of its local gravity field. Keep the complete planet-fixed Cartesian
anchor and basis in the deployment artifact; the INS frame ID refers to that
externally resolved frame. Manta does not derive latitude, longitude, height,
tides, or water-surface elevation.

The frame is immutable, validated at construction, and included in both the
transform-profile identity and Module metadata/hash. Changing anchor, rate,
gravity convention, frame ID, or epoch changes the estimator artifact identity.
`navigation_frame=None` retains the non-rotating equations. An explicit frame
with zero angular velocity reduces to the same equations to floating-point
precision. Do not pass an inertial Earth-world model as the `local_world`:
ordinary measurement models must describe the local navigation coordinates.

The Earth-axis projection is supplied explicitly. For ENU at geodetic latitude
phi it is `[0, Omega*cos(phi), Omega*sin(phi)]`. Tests construct an independent
ECEF basis and project the Cartesian planet axis, and separately exercise the
ordinary Manta IMU in an inertial world containing a rotating Earth. The
repository Earth preset's rate remains unchanged. Planet rate is configurable;
Manta's INS does not insert a second default constant. The authoritative WGS-84
reference and constants are published by [NGA](https://earth-info.nga.mil/?dir=wgs84&action=wgs84).

## Equations

The navigation frame is rigidly attached to a uniformly rotating planet. Its
axes remain fixed relative to the planet as the craft travels. There is no
moving-local-level transport rate.

Let `R` map body vectors to navigation axes, `Omega` be frame rotation in
navigation axes, and `w_ib` the calibrated, bias-corrected inertial body gyro.
The relative rate used by local sensor models and lever transforms is

```
w_nb = w_ib - R.T @ Omega
```

The finite attitude step uses the Hamilton quaternion convention:

```
q_next = Exp(-Omega * dt) * q * delta_q_inertial
```

For raw samples, `delta_q_inertial = Exp(w_ib * dt)`. Earth rotation is applied
once, on the companion side, and remains in the process Jacobian by automatic
differentiation. No duplicate gyro observation or heading measurement is added.

The continuous translation equation is

```
p_dot = v
v_dot = R @ f + g_effective - 2 * cross(Omega, v)
```

The declared GravityField convention is mandatory when a frame is supplied:

- `effective`: the field already includes centrifugal acceleration (as normal
  gravity does); use it directly.
- `gravitation`: subtract `cross(Omega, cross(Omega, origin + p))` exactly once.

Coriolis uses implicit midpoint integration. It preserves the norm of an
unforced velocity, avoids explicit-Euler energy growth, and does not divide by
the small Earth rate. Position uses the matching midpoint Coriolis correction
and the existing force delta-position convention. Small rotation-polynomial
solves are expressed as scalar arithmetic, preserving native C and SX/JAX
lowering without introducing a pivoting linear-solver dependency.

For a displaced raw IMU, existing relative tangential and centripetal lever
terms remain, with the additional Coriolis term for relative lever velocity.
The packet path propagates the sensor origin and removes the relative lever
velocity at both endpoints. The end correction uses the **end** attitude.
The retained delta/start/end gyro covariance and dynamic Schmidt state remain
part of the differentiated recurrence.

## Packet schema 2: deterministic timing moments

The MCU/preintegrator still integrates inertial motion in its starting sensor
frame. It receives no planet reference. The existing delta covariance,
bias Jacobian, and boundary noise correlations retain their meanings.

Eight deterministic doubles are added, in two length-four vectors:

```
velocity_time_moments[j] = sum(h_k * t_k**j)
position_time_moments[j] = sum(h_k * (T-t_k-h_k/2) * t_k**j)
```

Here `t_k` is a sample's left endpoint relative to packet start, `h_k` its held
interval, `T` total duration, and `j=0..3`. Arbitrary positive, nonuniform sample
intervals are supported. These moments have no stochastic covariance because
timestamp integrity is a separate device/transport contract.

Why additional fields are needed: an Earth-fixed body's inertial force vector
rotates during a packet, but its navigation-frame force stays constant. The
existing packet uses left-held force quadrature. Correcting its accumulated
force with a continuous midpoint rotation leaves an artificial half-sample
Earth-rate drift. Duration and the two force deltas alone do not describe the
quadrature for nonuniform acquisition intervals. The moments preserve that
information without placing Earth kinematics in the MCU.

With `W=[Omega]x`, the companion builds cubic rotation quadratures
`sum(W**j * moment[j] / j!)`. Their normalized inverses map the packet's force
deltas back to navigation-frame force averages. This is exact for constant
navigation-frame force up to the fourth-order rotation remainder. It is an
approximation for force varying during a packet: short-packet refinement is
required, and the rotation remainder alone is **not** a bound on motion error.
Raw one-sample and packet propagation use the same translation correction.

The kernel refuses invalid moment normalizations/bounds and packet rotation
larger than `1e-3 rad`. At that angle the cubic exponential remainder is below
`4.2e-14` relative to the constant-force quadrature. The guard does not qualify
a packet's application-specific duration or motion bandwidth. As with duration
mismatch, invalid inputs poison the generated navigation result rather than
substituting a plausible value; Python rejects the non-finite state.

Both preintegrator and INS Modules advertise `preintegration_packet_schema=2`.
An old transport packet is not silently upgraded. Rebuild generated MCU code,
packet adapters, fixtures, and the companion artifact together. This branch
makes no Shiver wire-ABI changes; integration must explicitly carry both new
vectors and retain acquisition/boundary integrity checks.

## Limits of this implementation

Correct mechanization exposes a weak heading channel through the ordinary
process transition and velocity aiding. It does not remove the yaw/gyro-bias
ambiguity of a stationary instrument. Qualification found overconfidence in
weak-bias cases, including a strong degradation with noisy gyros. An
observability-preserving filter/update design needs separate investigation;
noise floors, fake stationary measurements, and sensor-grade-based disabling
of Earth rotation would not fix that defect.

Moving-frame transport rate, qualified hardware gyrocompassing, Shiver artifact
migration, MCU deployment, Jetson timing, moving-trajectory Monte Carlo, and a
full DVL/depth/GNSS dropout qualification are not delivered by this draft.
