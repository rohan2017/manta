"""Craft — full 6-DOF rigid-body dynamics with Newton-Euler integration.

This module holds the `Craft` container (a part tree + initial-state
helpers), the `TickContext` passed to each `Part.update()`, and the
compile-time inertial/wrench helpers (`_aggregate_inertials`,
`_wrench_rotate_to_craft` + `_shift_wrench`). The dynamics pipeline
described below — the symbolic Newton-Euler tick that consumes those
helpers — is compiled in `tick/world_tick.py` (the sole tick path; see
`Sim(world)`). The description here is the physics contract that pipeline
implements.

Scope:
- 13-DOF rigid-body state: position (3) + orientation quaternion (4) +
  linear velocity (3) + angular velocity (3).
- Parts have static position offsets (`Part.transform`) in craft frame.
- Mass parts declare a diagonal MOI tensor about their own origin.
- Aggregation: total mass, COM in craft frame, MOI about craft origin
  (parallel-axis lifts from each part's position).
- Newton-Euler:
      a_com_scene = R_craft_to_scene · (F_net / m_total)
      I_com · α   = τ_com − ω × (I_com · ω)             (in craft frame)
  with τ_com = τ_origin − r_com × F_net.
  Origin acceleration is recovered via
      a_origin = a_com + R · [α × r_OC + ω × (ω × r_OC)]
  where r_OC = -r_com (origin minus COM in craft frame).
- Integration: position via symplectic-flavored Euler; orientation via
  SO3 boxplus on ω·dt; velocities via Euler.
- Single-phase parts: each Part implements exactly one `update(ctx)`
  function. `ctx.acceleration_world` / `ctx.acceleration_body` /
  `ctx.angular_acceleration` reflect **current-tick** dynamics — the
  framework runs `update()` against MX placeholders, then substitutes
  the real Newton-Euler outputs into the emitted sensor expressions
  before compiling the graph. Wrenches must not depend on those
  placeholders (the substitution would create an unsolved fixpoint);
  the compile step validates this and raises otherwise.

Articulation: nested joint chains compose symbolically through the
kinematic pass; `r_com`, `I_com`, and per-part offsets pick up joint-
angle dependence via `inertia.symbolic_inertia_rollup`, and the
joint-space block solve (`tick/joint_space.py`) makes stacked chains
dynamically exact — a 2-axis gimbal is two stacked `RevoluteJoint`s,
full coupling included. Native multi-DOF joint PARTS (ball, universal)
are just ergonomics on top of that, still future work.

Known omissions: non-identity static orientation between part and craft
frames; field disturbances tied to per-craft motion (only queried, not
contributed-to by parts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ir.frames import PartFrame
from .ir.types import Mat3, Quat, Scalar, Vec3
from .parts.base import Part
from .ir.wrench import Wrench


# ---------------------------------------------------------------------------
# Frame-indexed kinematic accessor
# ---------------------------------------------------------------------------

class _FrameView:
    """A kinematic quantity indexed by reference frame: `view[Frame]`
    returns the quantity measured relative to that frame, in that frame's
    coords. Backed by a `{Frame: Vec3}` dict the kinematic pass built."""

    __slots__ = ("_by_frame", "_what")

    def __init__(self, by_frame: dict, what: str) -> None:
        self._by_frame = by_frame
        self._what = what

    def __getitem__(self, frame):
        try:
            return self._by_frame[frame]
        except (KeyError, TypeError):
            avail = ", ".join(sorted(f.__name__ for f in self._by_frame))
            name = getattr(frame, "__name__", repr(frame))
            raise KeyError(
                f"ctx.{self._what}[{name}]: unavailable frame; "
                f"choose one of {{{avail}}}") from None


# ---------------------------------------------------------------------------
# TickContext
# ---------------------------------------------------------------------------

class TickContext:
    """The per-tick context passed to each `Part.update(ctx)` call.

    A part's `update()` works in the part's OWN frame (`PartFrame`): it
    authors native vectors (thrust, sensor axes) as `Vec3[PartFrame]` and
    returns its `Wrench` there; the framework rotates that to body coords
    and lifts it to the craft origin. `ctx.orientation : Quat[WorldFrame,
    PartFrame]` is the part's own world attitude (body attitude composed
    with any joints above it) — for a part on the craft root, PartFrame ≡
    CraftFrame.

    Kinematic quantities are **frame-indexed**: `ctx.X[Frame]` returns X
    measured RELATIVE TO `Frame`, in that frame's coords. Available frames:
    `WorldFrame` (inertial/absolute), `CraftFrame` (the craft body),
    `ParentFrame` (the immediate parent's frame), `PartFrame` (this part).
    Identities: `X[PartFrame]` is 0 (a point is at rest in its own frame);
    `X[CraftFrame]` is the joint-induced motion w.r.t. the airframe — 0 for
    a rigidly-mounted part, nonzero only through a joint; `X[WorldFrame]` is
    the absolute/inertial value; `X[ParentFrame]` equals `X[CraftFrame]` for
    a part on the root and isolates the local joint otherwise.

      t                    : Scalar  — world clock.
      dt                   : Scalar  — integrator timestep.
      orientation          : Quat[WorldFrame, PartFrame] — part attitude.
      position[F]          : Vec3[F] — the part's mount-point position.
                             [WorldFrame] is its world position (already
                             chain-composed — don't re-add `transform`).
      velocity[F]          : Vec3[F] — linear velocity of the mount point.
                             [WorldFrame] includes the body + joint lever
                             arms (don't re-add ω×r).
      acceleration[F]      : Vec3[F] — linear acceleration of the mount
                             point. A sensor's proper/specific force is
                             `acceleration[WorldFrame] - g_world`, rotated
                             into PartFrame via `ctx.orientation`.
      angular_velocity[F]  : Vec3[F] — the part frame's angular velocity.
                             A gyro reads `orientation.conjugate().apply(
                             angular_velocity[WorldFrame])`.
      angular_acceleration[F] : Vec3[F] — the part frame's angular accel.
      R_craft_from_part   : Mat3[CraftFrame, PartFrame] — PartFrame→body
                             rotation. Rarely needed (the framework rotates
                             wrenches/reads); exposed for advanced use.

    Field access:
      ctx.field(FieldCls) → registered field of that class, or an empty
      default instance if none is registered. The empty default's
      `value_at_sym(p, t)` returns the zero contribution, so a part can
      call `ctx.field(GravityField).value_at_sym(p, t)` regardless of
      whether a GravityField is attached to the world.
      ctx.has_field(FieldCls) → True iff a matching field is registered.

    Note on the acceleration fields: these reflect the **current**
    tick's Newton-Euler output (`α`, `a_origin`) lifted to this part's
    mount point (rigid lever arm + any joint-relative Coriolis/relative
    terms), then expressed in the part frame. At compile time the
    framework hands `update()` MX placeholder symbols for a/α (and each
    joint's θ̈) and substitutes the real expressions after Newton-Euler.
    Wrenches must not depend on these fields (the substitute pass doesn't
    solve fixpoints); the compile step validates and raises otherwise.
    """

    __slots__ = ("t", "dt", "orientation",
                 "position", "velocity", "acceleration",
                 "angular_velocity", "angular_acceleration",
                 "R_craft_from_part",
                 "_world", "_fields", "_sample_records")

    def __init__(self,
                 *,
                 t: Scalar,
                 dt: Scalar,
                 orientation: Quat,
                 position: dict,
                 velocity: dict,
                 acceleration: dict,
                 angular_velocity: dict,
                 angular_acceleration: dict,
                 R_craft_from_part: Mat3,
                 fields=(),
                 world=None) -> None:
        self.t = t
        self.dt = dt
        self.orientation = orientation
        # Frame-indexed kinematic accessors: ctx.position[Frame] etc.
        # Each `dict` maps a Frame tag → the quantity relative to that
        # frame, in that frame's coords (see KinematicState.frame_views).
        self.position             = _FrameView(position, "position")
        self.velocity             = _FrameView(velocity, "velocity")
        self.acceleration         = _FrameView(acceleration, "acceleration")
        self.angular_velocity     = _FrameView(angular_velocity,
                                               "angular_velocity")
        self.angular_acceleration = _FrameView(angular_acceleration,
                                               "angular_acceleration")
        self._world = world
        self._fields = tuple(fields)
        # Body-frame rotation of the part's frame (Mat3[CraftFrame,
        # PartFrame]). Identity for a part on the craft root. The framework
        # uses it to map a part's emitted wrench to body coords; parts
        # rarely need it directly now.
        self.R_craft_from_part = R_craft_from_part
        # Rate declarations collected during this part's update():
        # (id(value), rate_hz, kind) tuples from `sample()` / `hold()`.
        # The compiler matches the ids against the part's emitted outputs
        # / bound inputs to build the tick's `sample_rates` map.
        self._sample_records: list = []

    # ----- Field / world introspection ----------------------------------

    def _iter_fields(self):
        """Yield every field visible to this tick. World-attached fields
        come first; per-tick fields (set when there is no world) come
        next. Either source can satisfy a `field(cls)` lookup."""
        if self._world is not None:
            yield from self._world.fields
        yield from self._fields

    def has_field(self, cls: type) -> bool:
        """True iff a field of type `cls` (or a subclass) is registered.
        Use this for parts whose behaviour is gated on a field's
        presence (rather than just returning zero on a missing field)."""
        for f in self._iter_fields():
            if isinstance(f, cls):
                return True
        return False

    def field(self, cls: type):
        """Return the registered field of type `cls` (or subclass), or
        an empty default instance `cls()` if none is registered. Calling
        `value_at_sym` on the empty default returns the field's zero
        value, so a part can write
            `ctx.field(GravityField).value_at_sym(p, t)`
        unconditionally — missing field ⇒ zero contribution."""
        for f in self._iter_fields():
            if isinstance(f, cls):
                return f
        return cls()

    def get_field(self, cls: type):
        """Return the registered field of type `cls`, or None.
        Symmetric to `World.get_field`. Use `field(cls)` instead when
        you want a never-None lookup that falls back to an empty
        default."""
        for f in self._iter_fields():
            if isinstance(f, cls):
                return f
        return None

    def planet(self, cls: type):
        """Return the registered planet of type `cls` (or subclass),
        or None. Parts declaring `requires_planet = EarthClass` reach
        for the concrete instance through this accessor."""
        if self._world is None:
            return None
        for p in self._world.planets:
            if isinstance(p, cls):
                return p
        return None

    # ----- Rate declaration (Approach A: metadata only) ------------------

    def sample(self, value, rate=None):
        """Declare that an Output is sampled at `rate` Hz, and return
        `value` unchanged.

        This records metadata only — the compiled tick stays a pure
        function (no sample-and-hold state enters the kernel, so it never
        complicates autodiff or the EKF/LQR linearization). The Sim
        runtime reads the rate off the model and gates the matching
        output *port*: a fresh reading is published once per 1/rate
        window and held in between. `rate=None` ⇒ every tick. Use inside
        `update()`::

            return PartUpdate(outputs={
                "gyro": ctx.sample(gyro_vec, rate=self.rate)})
        """
        if rate is not None:
            self._sample_records.append((id(value), float(rate), "output"))
        return value

    def hold(self, value, rate=None):
        """Declare that an Input is accepted at `rate` Hz (zero-order
        hold), and return `value` unchanged.

        Like `sample()`, this is metadata only — the kernel is unchanged.
        The Sim and EKF runtimes gate the matching command *port*: a new
        command is latched once per 1/rate window and held in between, so
        truth and the estimator's predict see the *same* held command.
        `rate=None` ⇒ every tick. Use inside `update()` on the Input the
        part actuates::

            throttle = ctx.hold(self.throttle, rate=self.command_rate)
        """
        if rate is not None:
            self._sample_records.append((id(value), float(rate), "input"))
        return value


# ---------------------------------------------------------------------------
# Inertial aggregation (pure Python, runs at compile time)
# ---------------------------------------------------------------------------

def _aggregate_inertials(parts: list[Part]) -> dict[str, Any]:
    """Compute total mass, COM offset, and MOI-about-craft-origin from
    a list of parts. Returns concrete Python/numpy values — these are
    constants in the traced graph, not symbolic nodes.

    Walks `part.mass`, `part.moi`, `part.transform` on every part. Parts
    without `mass` (Thruster, sensors, drag surfaces, …) are skipped.
    A joint has no `.mass` of its own — its rotor `Mass` children are
    enumerated separately in the parts list, so the body sees the
    rotor's inertia distribution via the same code path as a plain
    `Mass`.
    """
    m_total = 0.0
    com_sum = np.zeros(3)        # m_i · r_i, then divide
    I_about_origin = np.zeros((3, 3))

    for part in parts:
        m = float(getattr(part, "mass", 0.0))
        if m <= 0.0:
            continue
        r = np.array(part.transform, dtype=float)
        moi_diag = getattr(part, "moi", (0.0, 0.0, 0.0))
        I_own = np.diag([float(moi_diag[0]),
                          float(moi_diag[1]),
                          float(moi_diag[2])])

        m_total += m
        com_sum += m * r
        # Parallel-axis lift: I_about_origin += I_own + m·(|r|²·I − r·r^T).
        I_about_origin += I_own + m * (
            float(r @ r) * np.eye(3) - np.outer(r, r))

    if m_total <= 0.0:
        return {"m_total": 0.0, "com": np.zeros(3),
                "I_origin": np.zeros((3, 3)), "I_com": np.zeros((3, 3))}

    com = com_sum / m_total
    # I_com = I_origin − m_total · (|com|² · I − com·com^T)
    I_com = I_about_origin - m_total * (
        float(com @ com) * np.eye(3) - np.outer(com, com))

    return {
        "m_total":  m_total,
        "com":      com,            # in CraftFrame
        "I_origin": I_about_origin,
        "I_com":    I_com,
    }


# ---------------------------------------------------------------------------
# Wrench transformation (part frame → craft frame)
# ---------------------------------------------------------------------------

def _wrench_rotate_to_craft(wrench_part: Wrench,
                            R_craft_from_part: Mat3) -> Wrench:
    """Rotate a part-emitted wrench into body coords WITHOUT lifting it.

    Rotation-only step: `F_body = R · F_input`,
    `τ_body = R · τ_input`. The torque stays referenced about the part's
    OWN origin (no `r × F` lift). The bottom-up wrench cascade
    (`world_tick`) lifts incrementally via `_shift_wrench` as it walks the
    tree, so the rotate and the lift are kept separate here.
    """
    if wrench_part.frame is not PartFrame:
        from .ir.frames import FrameError, _capture_user_source
        raise FrameError(
            "_wrench_rotate_to_craft",
            expected="part Wrench in PartFrame",
            got=f"frame={wrench_part.frame.__name__}",
            source=_capture_user_source(),
        )
    return Wrench(
        force=R_craft_from_part @ wrench_part.force,
        torque=R_craft_from_part @ wrench_part.torque,
    )


def _shift_wrench(wrench_craft: Wrench, delta_r: Vec3) -> Wrench:
    """Move a body-frame wrench's reference point by `delta_r` (CraftFrame).

    Reduces a child's wrench (referenced about the child origin) onto its
    parent's origin: `τ_parent = τ_child + delta_r × F`, where
    `delta_r = r_child − r_parent`. Force is unchanged. Used by the
    bottom-up cascade. Telescoping these per-hop shifts up a non-joint
    chain reproduces the single `r × F` lift to the root (rotate via
    `_wrench_rotate_to_craft`, then shift), so a flat craft aggregates
    exactly as a single lift-to-origin would.
    """
    return Wrench(
        force=wrench_craft.force,
        torque=wrench_craft.torque + delta_r.cross(wrench_craft.force),
    )


# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

class Craft:
    """A collection of parts with shared rigid-body dynamics.

    Internally a craft is a tree of parts rooted at `Craft.root` (a
    `RootPart`). `craft.add(part)` is sugar for `craft.root.add(part)`;
    `craft.parts` returns a flat tuple of all parts in the tree (DFS
    order). Nested composition (e.g. a joint hosting another joint
    for a pan-tilt gimbal) is supported via the standard composite
    `add()` chain on individual parts.

    State (13 DOF):
        position         : Vec3[WorldFrame]
        orientation      : Quat[WorldFrame, CraftFrame]
        velocity         : Vec3[WorldFrame]
        angular_velocity : Vec3[CraftFrame]
    plus one Scalar per `R1` State slot declared on any of the parts.
    """

    def __init__(self, name: str) -> None:
        from .parts.base import RootPart
        self.name = name
        self.root = RootPart(f"{name}_root")
        # Set by Sim(world) — used by TickContext helpers
        # (`ctx.has_field`, `ctx.get_field`, `ctx.planet`) so parts can
        # introspect optional registrations.
        self._world: "World | None" = None

    def add(self, part: Part) -> Part:
        """Attach a part to the craft's root. Equivalent to
        `craft.root.add(part)`."""
        return self.root.add(part)

    @property
    def parts(self) -> tuple[Part, ...]:
        """Flat tuple of every part in the tree, root first (DFS order).
        Excludes the root itself."""
        return tuple(p for p in self.root.walk() if p is not self.root)

    @property
    def total_mass(self) -> float:
        return sum(float(getattr(p, "mass", 0.0)) for p in self.parts)

    def aggregate_inertials(self) -> dict[str, Any]:
        """Public-facing accessor: see `_aggregate_inertials`. Useful for
        external inspection and tests."""
        return _aggregate_inertials(list(self.parts))

    # ----- Helpers --------------------------------------------------------

    def sample_noise(self, rng) -> dict:
        """Draw one tick of white-Gaussian samples for every declared
        `Noise` slot on every part. Returns a dict of
        `"<part>.<noise>" → np.ndarray` ready to merge into the state
        dict before calling the compiled tick.

        Slots whose sigma is 0 return zero vectors without consuming
        RNG state (so a deterministic-seed sim stays reproducible
        regardless of which noise channels are active).
        """
        out: dict[str, Any] = {}
        for part in self.parts:
            for nname, ndecl in part.noise_declarations().items():
                sigma = float(getattr(part, f"{nname}_sigma"))
                # Inert RW channels skip RNG entirely; everyone else
                # samples into the channel's driver-input name.
                if ndecl.contributes_state and sigma <= 0.0:
                    continue
                key = f"{part.name}.{ndecl.driver_input_name(nname)}"
                d = ndecl.signal_manifold.ambient_dim
                if d == 1:
                    out[key] = (rng.normal(0.0, sigma)
                                if sigma > 0.0 else 0.0)
                else:
                    out[key] = (rng.normal(0.0, sigma, d)
                                if sigma > 0.0
                                else np.zeros(d, dtype=float))
        return out

    def initial_state(self, **overrides) -> dict:
        """Build the initial state dict for the compiled tick.

        Returns a dict with the rigid-body slots (position, orientation,
        velocity, angular_velocity) AND a `"<part_name>.<state_name>"`
        entry for every part that declares state. Defaults come from each
        State declaration's `init`; keyword overrides replace them by name.
        """
        state: dict[str, Any] = {
            "position":         np.asarray((0.0, 0.0, 0.0), dtype=float),
            "orientation":      np.asarray((1.0, 0.0, 0.0, 0.0), dtype=float),
            "velocity":         np.asarray((0.0, 0.0, 0.0), dtype=float),
            "angular_velocity": np.asarray((0.0, 0.0, 0.0), dtype=float),
        }
        for part in self.parts:
            for sname, sdecl in part.state_declarations().items():
                if sdecl.manifold.kind == "scalar":
                    state[f"{part.name}.{sname}"] = float(sdecl.init)
                else:
                    # vec / quat — `init` is a fixed-length tuple
                    # validated at declaration time. Store as ndarray
                    # for symmetry with rigid-body slots.
                    state[f"{part.name}.{sname}"] = np.asarray(
                        sdecl.init, dtype=float)
            # Input slots: seed from the part's current attribute (which
            # is either the constructor-time override or the declaration
            # default). These pass through Sim.step's merge so
            # the user can update them per-tick or leave them alone.
            for iname in part.input_declarations():
                state[f"{part.name}.{iname}"] = float(getattr(part, iname))
            # Noise / RW-bias slots. Seed everything at zero.
            #   * White: one slot `<part>.<nname>` (the per-tick driver).
            #     EKF leaves it at zero; sim overwrites via
            #     `craft.sample_noise(rng)`.
            #   * RW (sigma > 0): two slots — `<part>.<nname>` is the
            #     bias state, `<part>.<nname>_driver` is the per-tick
            #     driver. RW channels with sigma == 0 are inert.
            for nname, ndecl in part.noise_declarations().items():
                # Each channel declares which slots it contributes to
                # the seed dict (white: just the signal; active RW:
                # bias + driver; inert RW: nothing).
                for k, v in ndecl.initial_state_entries(nname, part).items():
                    state[f"{part.name}.{k}"] = v
        unknown = set(overrides) - set(state)
        if unknown:
            raise KeyError(
                f"Craft.initial_state: unknown slot(s) {sorted(unknown)}. "
                f"Available: {sorted(state)}")
        for k, v in overrides.items():
            current = state[k]
            if isinstance(current, np.ndarray):
                state[k] = np.asarray(v, dtype=float)
            else:
                state[k] = float(v)
        return state

    # ----- Introspection --------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(p.name for p in self.parts)
        return f"<Craft '{self.name}' parts=[{parts}]>"
