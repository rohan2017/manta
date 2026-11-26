# IR primitives

The low-level `manta.ir` layer — frames, typed values, manifolds, and the
flat-vector state spec. Most users never touch this directly; it is
exported for advanced use (building bare CasADi graphs).

For custom Part and Field equations, use [manta.types](types.md) instead.
Numerical quaternion helpers under `manta.ir._rotation` are private.

## Frames

::: manta.ir.Frame

::: manta.ir.FrameError

## Typed values

::: manta.ir.Vec3

::: manta.ir.Mat3

::: manta.ir.Quat

::: manta.ir.Scalar

## Manifolds

::: manta.ir.Manifold

::: manta.ir.ScalarManifold

::: manta.ir.R3Manifold

::: manta.ir.SO3Manifold

## State spec

::: manta.ir.StateSpec

::: manta.ir.StateSlot

::: manta.ir.SlotSet

## Wrench

::: manta.ir.Wrench
