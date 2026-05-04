"""UKF — Python descriptor for an Unscented Kalman Filter.

Mirrors `EKF` (same wrap-a-World contract, same per-sensor measurement
list, same BoundSignal output surface) but generates a
`manta::estimation::WorldUKFOf<EstCraft<double>, MeasDim>` instance
instead of `WorldEKF<EstCraftT, MeasDim>`.

Two structural differences from EKF:

  * The wrapped craft does NOT need to be `scalar_templated`. UKF only
    ever calls `evaluate(x, dt)` with `double` scalars (the unscented
    transform doesn't take Jacobians), so plain non-templated crafts
    work too. Templated crafts work as well — the codegen instantiates
    the `<double>` form.

  * The constructor takes (alpha, beta, kappa) sigma-point tuning
    knobs. Defaults match the standard formulation:
      alpha = 1e-3   (sigma-point spread; smaller = closer)
      beta  = 2.0    (optimal for Gaussian state)
      kappa = 0.0    (secondary scaling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..signal import BoundSignal, Signal

if TYPE_CHECKING:
    from ..core import World


# Sentinel for UKF-output signals — accessor_for() recognizes the
# `$ukf/<ukf_id>` prefix and routes to the UKF's C++ accessors instead
# of any craft.
UKF_SENTINEL_PREFIX = "$ukf/"


def _state_slice_signal(name: str, accessor_method: str, n: int) -> Signal:
    return Signal(
        name=name,
        direction="out",
        n_floats=n,
        cpp_read_exprs=tuple(
            f"{{accessor}}.{accessor_method}()({i})" for i in range(n)
        ),
    )


@dataclass
class UKF:
    """Unscented Kalman Filter wrapping a World.

    Args:
        world: the World whose system model this UKF estimates. The
               world should contain exactly one Craft plus the fields /
               planets that the craft's parts need to access during
               predict.
        measurements: list of sensor-part descriptors the UKF observes.
        process_noise: scalar — Q is `process_noise * I`.
        initial_covariance: scalar — P_0 is `initial_covariance * I`.
        alpha, beta, kappa: sigma-point tuning knobs (see UKF<...>
               docstring). Defaults to the standard van-der-Merwe form.
    """
    world: "World"
    measurements: list = field(default_factory=list)
    process_noise: float = 1e-6
    initial_covariance: float = 1.0
    alpha: float = 1e-3
    beta:  float = 2.0
    kappa: float = 0.0

    STATE_DIM: int = 13

    def __post_init__(self) -> None:
        global _ukf_id_counter
        try:
            _ukf_id_counter += 1
        except NameError:
            _ukf_id_counter = 0
        self._ukf_id = _ukf_id_counter
        self._world = self.world

        if not self.world.crafts:
            raise ValueError(
                "UKF: world must have at least one craft "
                "(world.add_craft(c) before constructing the UKF).")
        if len(self.world.crafts) > 1:
            raise NotImplementedError(
                "UKF: only single-craft worlds are supported in v1.")

        # UKF doesn't require scalar_templated — `evaluate` is only
        # called with `double` (sigma-point propagation). Templated
        # crafts work too; we instantiate the <double> form.
        wrapped_craft = self.world.crafts[0].craft

        from ..core import PartDescriptor
        for m in self.measurements:
            if not isinstance(m, PartDescriptor):
                raise TypeError(
                    f"UKF.measurements: expected PartDescriptor instances, "
                    f"got {type(m).__name__} ({m!r}).")
            if getattr(m, "_craft", None) is not wrapped_craft:
                raise ValueError(
                    f"UKF: measurement part {m.name!r} is not attached to "
                    f"the UKF's wrapped craft {wrapped_craft.name!r}.")

        sd = self.STATE_DIM
        self.position        = self._make_signal("position",         "position",         3)
        self.orientation     = self._make_signal("orientation",      "orientation",      4)
        self.vel_linear      = self._make_signal("vel_linear",       "vel_linear",       3)
        self.vel_angular     = self._make_signal("vel_angular",      "vel_angular",      3)
        self.full_state      = self._make_signal("full_state",       "full_state",       sd)
        self.position_stddev    = self._make_signal("position_stddev",    "position_stddev",    3)
        self.orientation_stddev = self._make_signal("orientation_stddev", "orientation_stddev", 4)
        self.vel_linear_stddev  = self._make_signal("vel_linear_stddev",  "vel_linear_stddev",  3)
        self.vel_angular_stddev = self._make_signal("vel_angular_stddev", "vel_angular_stddev", 3)

    def _make_signal(self, name: str, accessor_method: str, n: int) -> BoundSignal:
        sig = _state_slice_signal(name, accessor_method, n)
        return BoundSignal(
            part_name=f"{UKF_SENTINEL_PREFIX}{self._ukf_id}",
            signal=sig,
            craft_ref=self,
        )

    def cpp_var_name(self) -> str:
        return f"ukf_{self._ukf_id}"

    def measurement_dim(self) -> int:
        total = 0
        for m in self.measurements:
            for sig in getattr(type(m), "signals", []) or []:
                if sig.direction == "out" and sig.name.startswith("last_"):
                    total += sig.n_floats
        return total
