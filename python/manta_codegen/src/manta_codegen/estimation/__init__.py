"""Python descriptors for manta's estimators.

These are codegen-side descriptions of EKF / UKF instances. Each
estimator wraps a `World` and is added to a `Target` as a Driveable —
the codegen emits the C++ wrapper class instantiation, registers
fields/planets transitively, and drives the predict + per-sensor
update calls in the binary's tick loop.

Public surface:
    EKF(world, measurements=[...], process_noise=..., initial_covariance=...)
"""
from .ekf import EKF

__all__ = ["EKF"]
