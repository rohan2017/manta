# Transforms — Sim, EKF, UKF, INS, LQR

The compile-time siblings. Each takes a `World`, writes its math
symbolically over the shared linearized system, and emits a typed
`Module`. Lower one with a [target](targets.md) to get a callable runtime.

## Sim

::: manta.Sim

## EKF

::: manta.EKF

## UKF

::: manta.UKF

## INS

::: manta.INS

## LQR

::: manta.LQR

### LQRSolution

One Riccati solve as plain data — what `LQR.resolve_at` returns and a
regulator's `reprogram()` installs. See
[moving the operating point](../explanation/control.md#moving-the-operating-point).

::: manta.LQRSolution

## PID

::: manta.PID
