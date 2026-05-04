"""EKF — Python descriptor for an autodiff Extended Kalman Filter.

Locked design (recap from the phase-C discussions):

  * The EKF *owns* a World — it is the world's tick driver. The user
    builds a World containing the est craft + any fields/planets the
    craft's parts query, then constructs `EKF(world, measurements=[...])`
    and adds it to a Target's `drives=[...]`.

  * `measurements=[...]` is a list of sensor-part *descriptors* (e.g.
    `est.imu`, `est.dvl`). The EKF reads R from the part's noise sigmas
    — no override; if you want different noise, tune the part. Each
    sensor's `consume_fresh()` decides whether the EKF applies its
    measurement update on a given tick.

  * Output signals are `BoundSignal`s living on the EKF descriptor —
    `ekf.position`, `ekf.orientation`, `ekf.vel_linear`, `ekf.vel_angular`
    plus per-component stddev variants. The user routes them via the
    same `connect()` / `publish()` primitives used for craft signals.

This file is the data model only — codegen plumbing through to C++
lands in a follow-up commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._format import cpp_float as _f
from ..signal import BoundSignal, Signal

if TYPE_CHECKING:
    from ..core import PartDescriptor, World


# Sentinel for EKF-output signals — accessor_for() recognizes the
# `$ekf/<ekf_id>` prefix and routes to the EKF's C++ accessors instead
# of any craft.
EKF_SENTINEL_PREFIX = "$ekf/"


def _state_slice_signal(name: str, accessor_method: str, n: int) -> Signal:
    """Out-direction signal that reads `n` floats from an EKF accessor —
    e.g. `ekf.position()` returns a length-3 slice of the state vector."""
    return Signal(
        name=name,
        direction="out",
        n_floats=n,
        cpp_read_exprs=tuple(
            f"{{accessor}}.{accessor_method}()({i})" for i in range(n)
        ),
    )


@dataclass
class EKF:
    """Autodiff Extended Kalman Filter wrapping a World.

    Args:
        world: the World whose system model this EKF estimates. The
               world should contain exactly one Craft (multi-craft
               estimation is out of scope for v1) plus the fields /
               planets that the craft's parts need to access during
               predict.
        measurements: list of sensor-part descriptors the EKF observes.
               Each part contributes its own `consume_fresh()` gate +
               R block read from the part's sigma fields.
        process_noise: scalar — Q is `process_noise * I`.
        initial_covariance: scalar — P_0 is `initial_covariance * I`.
        scalar_templated: forced True; included for API consistency with
               Craft. Estimator crafts must be templated for autodiff.
    """
    world: "World"
    measurements: list = field(default_factory=list)
    process_noise: float = 1e-6
    initial_covariance: float = 1.0

    # Static dimension of the rigid-body state — matches CraftT::kRigidStateDim.
    STATE_DIM: int = 13

    def __post_init__(self) -> None:
        # Stable identifier used in the BoundSignal sentinel and in
        # generated C++ variable names. Distinct EKFs in the same Target
        # need distinct ids; we hand out per-instance counters.
        global _ekf_id_counter
        try:
            _ekf_id_counter += 1
        except NameError:
            _ekf_id_counter = 0
        self._ekf_id = _ekf_id_counter

        # `_world` back-pointer mirrors Craft._world — the module-level
        # publish/connect/subscribe helpers walk craft_ref._world to find
        # the owning World. EKF output signals have craft_ref = self, so
        # they need their own back-pointer.
        self._world = self.world

        # Validate the wrapped world has at least one craft.
        if not self.world.crafts:
            raise ValueError(
                "EKF: world must have at least one craft "
                "(world.add_craft(c) before constructing the EKF).")
        if len(self.world.crafts) > 1:
            raise NotImplementedError(
                "EKF: only single-craft worlds are supported in v1.")
        # The wrapped craft must be scalar-templated for Jet autodiff.
        wrapped_craft = self.world.crafts[0].craft
        if not getattr(wrapped_craft, "scalar_templated", False):
            raise ValueError(
                f"EKF: wrapped craft {wrapped_craft.name!r} must have "
                f"`scalar_templated = True` (required for Jet autodiff).")

        # Validate each measurement is a PartDescriptor attached to the
        # same world's craft. Lazy import avoids a circular dependency.
        from ..core import PartDescriptor
        for m in self.measurements:
            if not isinstance(m, PartDescriptor):
                raise TypeError(
                    f"EKF.measurements: expected PartDescriptor instances, "
                    f"got {type(m).__name__} ({m!r}).")
            if getattr(m, "_craft", None) is not wrapped_craft:
                raise ValueError(
                    f"EKF: measurement part {m.name!r} is not attached to "
                    f"the EKF's wrapped craft {wrapped_craft.name!r}.")

        # Build the output BoundSignals once. They share a synthetic
        # `craft_ref` (this EKF) so the module-level publish/connect
        # helpers find the owning context — codegen recognizes the
        # sentinel and routes accordingly.
        sd = self.STATE_DIM
        self.position        = self._make_signal("position",         "position",         3)
        self.orientation     = self._make_signal("orientation",      "orientation",      4)
        self.vel_linear      = self._make_signal("vel_linear",       "vel_linear",       3)
        self.vel_angular     = self._make_signal("vel_angular",      "vel_angular",      3)
        self.full_state      = self._make_signal("full_state",       "full_state",       sd)
        # Stddev variants — codegen reads sqrt(P[i, i]) from corresponding
        # rows of the covariance matrix.
        self.position_stddev    = self._make_signal("position_stddev",    "position_stddev",    3)
        self.orientation_stddev = self._make_signal("orientation_stddev", "orientation_stddev", 4)
        self.vel_linear_stddev  = self._make_signal("vel_linear_stddev",  "vel_linear_stddev",  3)
        self.vel_angular_stddev = self._make_signal("vel_angular_stddev", "vel_angular_stddev", 3)

    def _make_signal(self, name: str, accessor_method: str, n: int) -> BoundSignal:
        sig = _state_slice_signal(name, accessor_method, n)
        bs = BoundSignal(
            part_name=f"{EKF_SENTINEL_PREFIX}{self._ekf_id}",
            signal=sig,
            craft_ref=self,   # sentinel; emit-side detects EKF-typed craft_ref
        )
        return bs

    # ---- helpers for codegen (used by emit/main.py in the follow-up) ----

    def cpp_var_name(self) -> str:
        """Stable identifier the codegen uses for this EKF's C++ instance."""
        return f"ekf_{self._ekf_id}"

    def measurement_dim(self) -> int:
        """Total measurement-vector width (sum of n_floats over all
        sensor parts). Used to instantiate `WorldEKF<EstCraftT, MeasDim>`."""
        from ..signal import Signal as _S
        total = 0
        for m in self.measurements:
            for sig in getattr(type(m), "signals", []) or []:
                if sig.direction == "out" and sig.name.startswith("last_"):
                    total += sig.n_floats
        return total
