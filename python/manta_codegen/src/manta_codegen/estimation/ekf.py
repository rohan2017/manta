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
        initial_covariance: scalar — P_0 is `initial_covariance * I`,
               then any of the per-block overrides below replace
               specific diagonal entries.

        Process noise (Q) is now built entirely from the noise channels
        registered on the parts (see PartDescriptor.noise_channels —
        IMU's accel_sigma / gyro_sigma / gyro_bias_sigma, Thruster's
        force_noise_sigma, etc.). Set σ ≥ 0 on the relevant part to
        contribute to Q automatically; the EKF wrapper assembles
        L · Σ · Lᵀ from autodiff-extracted noise-input gains every
        predict step. To inject noise on a DOF that has no obvious
        physical noise source, declare a `force_noise` on whatever part
        affects that DOF — that's the architectural answer to "model-
        error fudge factor."

        Per-craft init knobs accept any of these shapes — same contract
        for state vectors AND variances. The codegen resolves them to a
        per-craft list at emit time:

          * None (default)
              - State fields: read from `world.add_craft(c, pos=...,
                ori=..., vel=..., vel_angular=...)` — the world's
                per-craft init is the source of truth.
              - Variance fields: fall back to `initial_covariance` for
                that block on every craft.

          * A single value (tuple for state, scalar for variance) —
            broadcast to every craft.

          * A list of values, length = number of crafts — one entry
            per craft in `world.crafts` order.

          * A dict keyed by craft name — only the named crafts are
            overridden; the rest fall back to the broadcast / world-
            default behavior. Useful when only some crafts in a swarm
            need specific init.

        Per-craft INITIAL STATE overrides:
            initial_position           — 3-tuple (px, py, pz)
            initial_orientation        — 4-tuple (qw, qx, qy, qz)
            initial_velocity           — 3-tuple (vx, vy, vz) in scene
            initial_angular_velocity   — 3-tuple (wx, wy, wz) in body

        Per-block INITIAL VARIANCE overrides (scalar applied to each
        diagonal entry of the named block):
            initial_position_var
            initial_attitude_var
            initial_velocity_var
            initial_angular_velocity_var

        Tuning the *_var knobs is how you tell the EKF "I trust this
        block of the initial state — don't burn measurement information
        relearning it." For sensor suites that don't observe absolute
        attitude (IMU + DVL + gyro, all body-frame), giving the
        attitude block a moderate variance (e.g. 1e-4) keeps q close
        to the seeded initial during the run; the EKF still updates
        within that block but doesn't drift to a self-consistent
        rotated solution.

        Examples (multi-craft swarm):

            # Same init for both, default world placement:
            EKF(w, ..., initial_attitude_var=1e-4)

            # craft_0 starts at origin, craft_1 at (10, 0, 0); variances
            # broadcast across both:
            EKF(w, ..., initial_position=[(0,0,0), (10,0,0)],
                       initial_position_var=1e-4)

            # Only craft "leader" gets a tight attitude lock, "follower"
            # uses the default isotropic P_0:
            EKF(w, ..., initial_attitude_var={"leader": 1e-9})

    Note: every craft in the EKF's wrapped world is automatically
    flipped to `scalar_templated = True` at construction time. Users
    don't need to set that flag manually — templating is purely a
    codegen detail (it switches the emitted C++ class shape to
    `MyCraftT<Scalar>` so the Jet shadow world can instantiate it).
    """
    world: "World"
    measurements: list = field(default_factory=list)
    initial_covariance: float = 1.0

    initial_position:         "tuple | list | None" = None
    initial_orientation:      "tuple | list | None" = None
    initial_velocity:         "tuple | list | None" = None
    initial_angular_velocity: "tuple | list | None" = None

    initial_position_var:         float | None = None
    initial_attitude_var:         float | None = None
    initial_velocity_var:         float | None = None
    initial_angular_velocity_var: float | None = None

    # Block-decomposed predict — assume crafts don't physically couple
    # (no tether, contact, or cross-craft fluid term) so F is block-
    # diagonal: ∂x_i_post/∂x_j_pre = 0 for i ≠ j. Each tick runs
    # NumCrafts separate Jet passes, each with 13 partials (seeding
    # one craft with identity, the others with zero-deriv values),
    # instead of one monolithic pass with 13·NumCrafts partials. Cost
    # per tick scales as NumCrafts (linear) instead of NumCrafts²
    # (quadratic) — the full Jet world pass becomes ~NumCrafts× faster
    # at NumCrafts ≥ ~5.
    #
    # Default False: the monolithic predict is correct for any coupling
    # topology (tethers, contacts, fluids — the autodiff through the
    # joint state captures every cross-craft Jacobian). Use True only
    # when crafts are guaranteed independent (decoupled swarm).
    block_decomposed: bool = False

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
        # The EKF's Jet shadow World instantiates each wrapped craft as
        # `<JetType>` for the Jacobian step, so every craft here MUST be
        # scalar-templated. Templating is purely a codegen detail (it
        # picks the C++ class shape — `MyCraftT<Scalar>` instead of
        # `MyCraft`) — users shouldn't have to think about it. Just
        # flip the flag automatically; the codegen sees `True` either
        # way.
        world_crafts = [e.craft for e in self.world.crafts]
        for c in world_crafts:
            c.scalar_templated = True

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
        sensor parts). Used to instantiate `EKF<EstCraftT, MeasDim>`."""
        from ..signal import Signal as _S
        total = 0
        for m in self.measurements:
            for sig in getattr(type(m), "signals", []) or []:
                if sig.direction == "out" and sig.name.startswith("last_"):
                    total += sig.n_floats
        return total
