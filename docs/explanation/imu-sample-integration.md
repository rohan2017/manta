# Integrating timestamped IMU samples

For paired point acquisitions, use `INSStream` or explicitly select both
`gyro_integration="endpoint"` and `accel_integration="endpoint"` in
`IMUPreintegrator`, with `sample_rate_hz` set to the acquisition bandwidth.
The matching INS needs `propagation="preintegrated"`,
`accel_integration="endpoint"`, and a non-linearized covariance chart.
See the [standalone guide](../how-to/use-ins.md) for the driving loop.

## Mean integration

For bias-corrected endpoint gyro rates `w0,w1` over interval `h`, endpoint
acceleration mode uses the rotation increment

```text
DeltaR = Exp((w0+w1)*h/2 + cross(w0,w1)*h²/12)
```

It rotates the start, interpolated midpoint, and end specific forces into
packet-start axes. Simpson quadrature integrates velocity; its first time
moment integrates position. This removes the leading half-sample lag of
left-held integration. It assumes smooth motion between observed points;
refine the acquisition/packet duration to test motion-dependent error.

The legacy defaults remain `left_hold` for both gyro and acceleration.
Selecting endpoint gyro alone uses `Exp(w0*h/2) Exp(w1*h/2)` with left-held
force quadrature. These are distinct acquisition contracts: choose the one
that matches the data. Hardware increments, filtered outputs, and duplicated
held samples are not independent point acquisitions.

## Shared acquisition uncertainty

An interior sample contributes to both neighboring intervals. Treating
interval averages as independent would understate long-term variance.
Both gyro and accelerometer endpoint errors are retained. For delta error
`d` and standardized six-axis acquisition errors `eta`,

```text
d_next = A*d + L*eta_left + E*eta_right
P_next = A*P*A' + L*L' + E*E' + A*C_left*L' + L*C_left'*A'
Cov(d_next, eta_right) = E
```

Here `C_left = Cov(d, eta_left)`. Jacobians differentiate the actual mean
recurrence and both endpoints with respect to the same bias. At packet
boundaries, the INS retains the endpoint error state for subsequent packets,
lever-arm aiding, and any `ModelForce` update using that acquisition.
Independent model-force measurement variance excludes the sensor variance
already carried by that state.

Point-sample noise is `sample_sigma = density * sqrt(sample_rate_hz)`.
The duration can vary without changing the assigned uncertainty of a reused
acquisition. This does not characterize unknown timing jitter, missing-sample
motion, or vendor filtering.

## Packet compatibility

Schema **4** carries delta covariance, bias sensitivities, gyro and accel
start/end cross-covariances, timing moments, and the `accel_endpoint` mode tag.
The INS rejects an incompatible accelerometer integration mode. Rebuild
transport schemas, generated code, and estimator artifacts together.

Do not call `frame_preintegrated_packet` on an endpoint-integrated packet:
that helper attaches an independent gyro endpoint and clears its correlation.
Use the actual paired right acquisition. Partial flushes must retain this
same acquisition and its covariance. `INSStream` implements that contract
for ordered measurements at observed boundaries; applications with delayed
or asynchronous data must supply their own timing/replay layer.
