# Finite INS coordinates for local current

When an INS carries world-frame water velocity, force aiding observes vehicle
velocity relative to that water. The earlier finite chart rotated vehicle
velocity while retaining additive current errors. Even linear body-frame drag
then had a finite heading/current curvature term that repeated Jacobian
corrections could interpret as information. Noisy submerged tests exposed both
premature startup uncertainty reduction and an attitude-accuracy regression.

`INSStateSpec` now includes tracked `R3Manifold(frame=WorldFrame)` slots in the
same finite-vector retraction as craft position and velocity. Selection uses
declared frame metadata, so renamed current fields work and sensor-frame IMU
biases are excluded. The existing forward map, inverse, physical-prior mapping
and full covariance reset all use that vector list. INS mechanics, observation
noise and the choice of analytic versus sigma-point prediction are unchanged.

Charts containing additional world-frame vectors identify themselves as
`gravity_and_earth_referenced_swing_twist_v3`. The existing chart without those
additional vectors retains its v2 identity. Rebuild affected estimator artifacts;
their finite-coordinate meaning has changed.

Two focused contracts failed before the fix and pass with it in both geometric
and nonlinear modes:

- Linear drag remains affine under joint finite yaw and relative-velocity
  perturbations, including renamed current fields. Retraction/inverse round trips
  retain the full increment.
- For an independent Gaussian heading prior, the native current mean is mapped
  consistently: a planar physical current mean becomes `exp(-sigma_yaw²/2)`
  times that mean in the initialized chart. This is a coordinate mapping, not
  a physical decay of water velocity.

The motivating FOG-reference submerged cohort used eight 300 s trials with no
GPS or heading observation and a 5° initial heading standard deviation. Before
the fix, mean attitude RMS was 1.457° with current aiding versus 1.163° in the
baseline. A scoped diagnostic applying the shared current retraction restored
1.163° and passed its declared normal and DVL-outage gates. All four normal and
DVL-outage hardware-profile cohorts also pass through the public implementation,
using AHRS-10P and P-1775-derived noise/bias profiles; their normal nonlinear
prediction cohorts pass too. P-1775 is retained historical evidence. The user-
selected FOG reference is Boreas A50; the Shiver report documents its published
biases and explicitly assumed white-noise sensitivity cases. The A50 normal,
DVL-outage, nonlinear and low/high-noise cohorts pass, while the fixed 15°
heading diagnostic still exceeds its absolute attitude/current accuracy limits. The affected Manta suites
pass 65 tests, and Shiver's current/preintegration/qualification suites pass 34.

This does not resolve every earlier confidence failure: the extremely tight
synthetic calibrated-IMU startup cohort already exposes a baseline limitation
shared by geometric and nonlinear prediction. Retain those failures rather than
claiming universal statistical acceptance from the current-specific correction.

Detailed profiles, seeds, retained before/after data, limitations and commands
are in the [Shiver isolated qualification report](../../../shiver/docs/qualification/local-current/isolated/README.md).
