"""Typed control-runtime failures that callers may handle by policy."""


class MpcNumericalError(RuntimeError):
    """A transient MPC solver/numerical failure eligible for a cold retry.

    Model construction, invalid references, and invalid caller inputs retain
    their existing exception types and are not retryable through this class.
    """
