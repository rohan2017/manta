"""Python descriptors for manta's estimators.

These are codegen-side descriptions of EKF / UKF instances. Each
estimator wraps a `World` and is added to a `Target` as a Driveable —
the codegen emits the C++ wrapper class instantiation, registers
fields/planets transitively, and drives the predict + per-sensor
update calls in the binary's tick loop.

Public surface:
    EKF(world, measurements=[...], process_noise=..., initial_covariance=...)
    UKF(world, measurements=[...], process_noise=..., initial_covariance=...,
        alpha=1e-3, beta=2.0, kappa=0.0)
"""
from .ekf import EKF
from .ukf import UKF

__all__ = ["EKF", "UKF"]
