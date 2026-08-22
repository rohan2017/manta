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
- Parts have a static pose (`Part.mount_offset` + `mount_orientation`)
  relative to their parent.
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
  SO3 boxplus on ω·dt; linear velocity via Euler. Angular velocity is
  NOT an Euler step on α — it advances the body-frame generalized
  momentum `p = A·[ω; q̇]` and re-solves for ω (and the joint rates
  with it), so a stacked chain conserves axial momentum exactly; see
  `tick/world_tick._integrate_angular_momentum`.
- Single-phase parts: each Part implements exactly one `update(ctx)`
  function. `ctx.acceleration[Frame]` / `ctx.angular_acceleration[Frame]`
  reflect **current-tick** dynamics — the
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

Known omissions: field disturbances tied to per-craft motion (only
queried, not contributed-to by parts).
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
                             chain-composed — don't re-add `mount_offset`).
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
      ctx.has_field(FieldCls) → True iff a field of that class (or a
      subclass) is registered with the world. ctx.field(FieldCls) →
      that registered instance; raises if none is registered — there is
      NO silent empty-field default. A part whose physics needs the
      field present declares `requires_fields = [FieldCls]` (validated
      at transform build, so `ctx.field` cannot fail); a part for which
      the field is genuinely optional branches explicitly::

          if ctx.has_field(MagField):
              B = ctx.field(MagField).value_at_sym(p, ctx.t)
          else:
              B = Vec3[WorldFrame].constant((0.0, 0.0, 0.0))

      (Gravity is not optional at this level: every World must declare
      a GravityField — `GravityField.none()` for deliberate zero-g — so
      `gravity_at(ctx, p)` always has a registered field to read.)

      Both work for user-authored Field subclasses — lookup is by
      isinstance against the world's registered fields, never against a
      built-in list of field kinds.

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
                 "_world", "_fields")

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

    # ----- Field / world introspection ----------------------------------

    def _iter_fields(self):
        """Yield every field visible to this tick. World-attached fields
        come first; per-tick fields (set when there is no world) come
        next. Either source can satisfy a `field(cls)` lookup."""
        if self._world is not None:
            yield from self._world.fields
        yield from self._fields

    def has_field(self, cls: type) -> bool:
        """True iff a field of type `cls` (or a subclass) is registered
        with the world. The explicit presence check for parts whose
        field is optional — pair with `field(cls)` inside the branch."""
        return any(isinstance(f, cls) for f in self._iter_fields())

    def field(self, cls: type):
        """Return the registered field of type `cls` (or subclass).

        Raises ValueError if none is registered — there is no silent
        empty-field default. Either declare the field on the part
        (`requires_fields = [cls]`, validated at transform build) so
        this can't fail, or guard the query with `ctx.has_field(cls)`.
        """
        for f in self._iter_fields():
            if isinstance(f, cls):
                return f
        raise ValueError(
            f"ctx.field({cls.__name__}): no {cls.__name__} is registered "
            f"with this world. Either add_field one, declare "
            f"`requires_fields = [{cls.__name__}]` on the part (so the "
            f"transform validates it up front), or make the query "
            f"conditional with `ctx.has_field({cls.__name__})`.")


# ---------------------------------------------------------------------------
# Inertial aggregation (pure Python, runs at compile time)
# ---------------------------------------------------------------------------

def _aggregate_inertials(root_part: Part) -> dict[str, Any]:
    """Total mass, COM offset, and MOI about craft origin and COM for
    the part tree rooted at `root_part`, as concrete Python/numpy
    values — constants in the traced graph, not symbolic nodes.

    This is the SAME walk the dynamics use, evaluated at rest
    (`tick.inertia.aggregate_inertials_at_rest`): only parts carrying
    the `contributes_inertia` trait are counted (a `mass` attribute
    alone is not inertial — `TrajectoryEndpoint.mass` is a feedforward
    gain), each part's craft-frame pose composes `mount_offset` and
    `mount_orientation` through the parent chain, and joints contribute
    their declared rest-pose rotation/slide.
    """
    from .tick.inertia import aggregate_inertials_at_rest
    return aggregate_inertials_at_rest(root_part)


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
        from .ir.module import check_name
        from .parts.base import RootPart
        self.name = check_name(name, who="Craft")
        self.root = RootPart(f"{name}_root")

    def add(self, part: Part) -> Part:
        """Attach a part to the craft's root. Equivalent to
        `craft.root.add(part)`."""
        return self.root.add(part)

    def remove(self, part: Part | str) -> Part:
        """Detach a part anywhere in this craft's tree.

        Existing transforms retain their private model revision. A later
        transform captures the edited tree.
        """
        match = next(
            (candidate for candidate in self.parts
             if candidate is part
             or (isinstance(part, str) and candidate.name == part)),
            None,
        )
        if match is None:
            label = part if isinstance(part, str) else getattr(part, "name", part)
            raise KeyError(f"Craft('{self.name}').remove: no part {label!r}")
        return match.parent.remove(match)

    @property
    def parts(self) -> tuple[Part, ...]:
        """Flat tuple of every part in the tree, root first (DFS order).
        Excludes the root itself."""
        return tuple(p for p in self.root.walk() if p is not self.root)

    @property
    def total_mass(self) -> float:
        """Sum of the declared `mass` of every genuinely inertial part
        (`contributes_inertia` trait) — gain-like `mass` parameters
        (e.g. TrajectoryEndpoint's feedforward) don't count."""
        return sum(float(p.mass) for p in self.parts
                   if p.contributes_inertia)

    def aggregate_inertials(self) -> dict[str, Any]:
        """Public-facing accessor: see `_aggregate_inertials`. Useful for
        external inspection and tests."""
        return _aggregate_inertials(self.root)

    # ----- Helpers --------------------------------------------------------

    def sample_noise(self, rng) -> dict:
        """Draw one tick of white-Gaussian samples for every declared
        `Noise` slot on every part. Returns a dict of
        `"<part>.<noise>" → np.ndarray` ready to merge into the state
        dict before calling the compiled tick.

        This is the *model-side* draw, keyed by full state name. The
        running sim does not call it — `NumpySim` drives noise through
        the Module's NOISE port with a `NoiseDriver`. Keep it for
        hand-driven ticks and for tests that want the per-channel
        sigmas without a backend.

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
            #     EKF leaves it at zero; `NumpySim` overwrites it from an
            #     attached `NoiseDriver` (see `codegen/numpy/_noise.py`).
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
