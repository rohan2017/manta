# Parts

A `Part` is an atomic unit of behavior on a craft. The declaration
sentinels (`Parameter`, `State`, `Input`, `Output`, `Noise`) are
documented on the base classes; the stock parts below are what you add to
a craft.

## Declaration model

::: manta.parts.Part

::: manta.parts.CompositePart

::: manta.parts.Parameter

::: manta.parts.State

::: manta.parts.Input

::: manta.parts.Output

::: manta.parts.Noise

::: manta.parts.PartUpdate

## Structure

::: manta.parts.Mass

::: manta.parts.PointBuoy

::: manta.parts.Collider

::: manta.parts.ThermalMass

## Electrical

::: manta.parts.ElectricalNode

::: manta.parts.DCSource

::: manta.parts.ElectricalBus

::: manta.parts.DCConverter

::: manta.parts.Contactor

::: manta.parts.Fuse

::: manta.parts.ElectricalLoad

::: manta.parts.ResistiveLoad

::: manta.parts.ConstantCurrentLoad

::: manta.parts.ConstantPowerLoad

## Actuation

::: manta.parts.Thruster

::: manta.parts.Motor

## Articulation

::: manta.parts.RevoluteJoint

::: manta.parts.PrismaticJoint

## Aerodynamics

::: manta.parts.DragSurface

::: manta.parts.RotationalDrag

::: manta.parts.AddedMass

::: manta.parts.FossenDamping

::: manta.parts.Aerofoil

::: manta.parts.naca

::: manta.parts.ControlSurface

## Sensors

::: manta.parts.IMU

::: manta.parts.VelocitySensor

::: manta.parts.Magnetometer

::: manta.parts.PositionSensor

::: manta.parts.Barometer

::: manta.parts.ProjectiveCamera

::: manta.parts.BBoxCamera

::: manta.parts.CentroidCamera

::: manta.parts.Antenna

## Attachment and disturbance

::: manta.parts.TetherEndpoint

::: manta.parts.TrajectoryEndpoint

::: manta.parts.ProcessNoise

## Field sources

::: manta.parts.GravitySource

::: manta.parts.MagneticSource

::: manta.parts.OpticalSource
