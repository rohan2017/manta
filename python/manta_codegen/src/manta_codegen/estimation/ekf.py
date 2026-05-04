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


def _state_slice_signal(name: str, accessor_method: str, n: int,
                         craft_idx: int = 0) -> Signal:
    """Out-direction signal that reads `n` floats from an EKF accessor.
    The accessor methods (`position`, `vel_linear`, etc.) all take an
    optional craft_idx (default 0) — for multi-craft worlds we bake the
    index in at codegen time so `ekf.crafts['my_drone'].position` reads
    that craft's slice of the joint state."""
    return Signal(
        name=name,
        direction="out",
        n_floats=n,
        cpp_read_exprs=tuple(
            f"{{accessor}}.{accessor_method}({craft_idx})({i})" for i in range(n)
        ),
    )


class _CraftEkfSignals:
    """Per-craft BoundSignal bundle. `ekf.crafts['drone_0'].position`
    reads `ekf_0.position(0)`; `ekf.crafts['drone_1'].position` reads
    `ekf_0.position(1)` from the joint state vector.

    Mirrors the surface of the EKF descriptor for a single craft slice.
    """
    def __init__(self, name: str):
        self.name = name
        # Signals are populated by the EKF descriptor's __post_init__
        # since they need the EKF reference for the craft_ref sentinel.
        self.position:           "BoundSignal | None" = None
        self.orientation:        "BoundSignal | None" = None
        self.vel_linear:         "BoundSignal | None" = None
        self.vel_angular:        "BoundSignal | None" = None
        self.position_stddev:    "BoundSignal | None" = None
        self.orientation_stddev: "BoundSignal | None" = None
        self.vel_linear_stddev:  "BoundSignal | None" = None
        self.vel_angular_stddev: "BoundSignal | None" = None


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
        # Multi-craft worlds work natively under the templated-World
        # architecture: state is the concat of every craft's 13-DOF rigid
        # state, and the Jet shadow World propagates Jacobians through
        # any inter-craft physics (tethers, contacts, fluid coupling).
        # Every craft in the wrapped world must be scalar-templated —
        # the EKF's Jet shadow World instantiates each craft as
        # `<JetType>` for the Jacobian step.
        world_crafts = [e.craft for e in self.world.crafts]
        for c in world_crafts:
            if not getattr(c, "scalar_templated", False):
                raise ValueError(
                    f"EKF: craft {c.name!r} must have "
                    f"`scalar_templated = True` (required for Jet autodiff).")

        # Validate each measurement is a PartDescriptor attached to one of
        # the wrapped world's crafts.
        from ..core import PartDescriptor
        for m in self.measurements:
            if not isinstance(m, PartDescriptor):
                raise TypeError(
                    f"EKF.measurements: expected PartDescriptor instances, "
                    f"got {type(m).__name__} ({m!r}).")
            if getattr(m, "_craft", None) not in world_crafts:
                names = [c.name for c in world_crafts]
                raise ValueError(
                    f"EKF: measurement part {m.name!r} is not attached to "
                    f"any craft in the EKF's wrapped world (crafts: {names}).")

        # Build the per-craft signal tree. For each craft in the wrapped
        # world, expose a `_CraftEkfSignals` instance with position/
        # orientation/vel_linear/vel_angular + stddev variants that read
        # that craft's slice of the joint state (state[13*idx + ...]).
        # Indexed by craft name: `ekf.crafts["drone_0"].position`.
        self.crafts: dict[str, _CraftEkfSignals] = {}
        for idx, c in enumerate(world_crafts):
            cs = _CraftEkfSignals(c.name)
            cs.position           = self._make_craft_signal(idx, "position",         "position",         3)
            cs.orientation        = self._make_craft_signal(idx, "orientation",      "orientation",      4)
            cs.vel_linear         = self._make_craft_signal(idx, "vel_linear",       "vel_linear",       3)
            cs.vel_angular        = self._make_craft_signal(idx, "vel_angular",      "vel_angular",      3)
            cs.position_stddev    = self._make_craft_signal(idx, "position_stddev",    "position_stddev",    3)
            cs.orientation_stddev = self._make_craft_signal(idx, "orientation_stddev", "orientation_stddev", 4)
            cs.vel_linear_stddev  = self._make_craft_signal(idx, "vel_linear_stddev",  "vel_linear_stddev",  3)
            cs.vel_angular_stddev = self._make_craft_signal(idx, "vel_angular_stddev", "vel_angular_stddev", 3)
            self.crafts[c.name] = cs

        # Single-craft shortcuts at the top level — `ekf.position` is
        # equivalent to `ekf.crafts[<sole_craft>].position`. Multi-craft
        # callers should use the explicit dict path.
        sd = self.STATE_DIM
        first = world_crafts[0]
        first_cs = self.crafts[first.name]
        self.position        = first_cs.position
        self.orientation     = first_cs.orientation
        self.vel_linear      = first_cs.vel_linear
        self.vel_angular     = first_cs.vel_angular
        self.position_stddev    = first_cs.position_stddev
        self.orientation_stddev = first_cs.orientation_stddev
        self.vel_linear_stddev  = first_cs.vel_linear_stddev
        self.vel_angular_stddev = first_cs.vel_angular_stddev
        # Full state spans the whole joint vector; not per-craft.
        self.full_state = self._make_full_state_signal(sd)

    def _make_craft_signal(self, craft_idx: int, name: str,
                           accessor_method: str, n: int) -> BoundSignal:
        """BoundSignal that reads craft_idx's slice via accessor(idx)."""
        sig = _state_slice_signal(name, accessor_method, n,
                                  craft_idx=craft_idx)
        return BoundSignal(
            part_name=f"{EKF_SENTINEL_PREFIX}{self._ekf_id}",
            signal=sig,
            craft_ref=self,
        )

    def _make_full_state_signal(self, n: int) -> BoundSignal:
        # `full_state()` returns the whole concat'd state, no craft_idx.
        sig = Signal(
            name="full_state",
            direction="out",
            n_floats=n,
            cpp_read_exprs=tuple(
                f"{{accessor}}.full_state()({i})" for i in range(n)
            ),
        )
        return BoundSignal(
            part_name=f"{EKF_SENTINEL_PREFIX}{self._ekf_id}",
            signal=sig,
            craft_ref=self,
        )

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
