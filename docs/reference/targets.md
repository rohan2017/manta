# Targets and runtimes

A `Target*` lowers a Module to a backend. `TargetNumpy` returns the
matching native-Python runtime view; `TargetCpp` emits an Eigen C++
library; `TargetJax` emits a jitted rollout.

## Targets

::: manta.TargetNumpy

::: manta.TargetCpp

::: manta.TargetJax

::: manta.NoiseDriver

## Numpy runtime views

The view `TargetNumpy(x)` returns is determined by the Module's shape.

::: manta.codegen.NumpyRuntime

::: manta.codegen.NumpySim

::: manta.codegen.NumpyFilter

::: manta.codegen.NumpyRegulator

::: manta.codegen.NumpyRecurrence

## Rate helpers

::: manta.RateGate

::: manta.CommandLatch
