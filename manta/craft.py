"""Craft — full 6-DOF rigid-body dynamics with Newton-Euler integration.

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

Articulation: nested `Joint` chains compose symbolically through the
kinematic pass; `r_com`, `I_com`, and per-part offsets pick up joint-
angle dependence via `inertia.symbolic_inertia_rollup`. Native multi-
DOF joints (ball, universal) are still future work — for now stack
single-DOF Joints to build them.

Known omissions: non-identity static orientation between part and craft
frames; field disturbances tied to per-craft motion (only queried, not
contributed-to by parts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ir.frames import CraftFrame
from .ir.types import Mat3, Quat, Scalar, Vec3
from .parts.base import Part, WhiteNoise
from .ir.wrench import Wrench


# ---------------------------------------------------------------------------
# TickContext
# ---------------------------------------------------------------------------

class TickContext:
    """The per-tick context passed to each `Part.update(ctx)` call.

    Exposes the symbolic per-part state + environment to part code so a
    Part's `update()` can read everything it physically depends on:

      t                : Scalar                      — world clock.
      dt               : Scalar                      — integrator timestep.
      position         : Vec3[WorldFrame]           — the PART's mount-
                                                       point position in
                                                       world frame (chain-
                                                       composed for parts
                                                       on a joint subtree).
      orientation      : Quat[WorldFrame,CraftFrame]— craft attitude.
      velocity         : Vec3[WorldFrame]           — world-frame linear
                                                       velocity at the
                                                       part's mount point.
      angular_velocity : Vec3[CraftFrame]            — body angular velocity
                                                       ω. What a rate gyro
                                                       reads.
      velocity_body    : Vec3[CraftFrame]            — body linear velocity
                                                       (R^T · v_anchor).
                                                       What a DVL reads.
      R_craft_from_input : Mat3[CraftFrame, CraftFrame]
                                                     — rotates a vector
                                                       in the part's
                                                       INPUT-frame coords
                                                       into body-frame
                                                       coords. Identity
                                                       for parts mounted
                                                       directly on the
                                                       craft root; for a
                                                       Joint nested
                                                       inside another
                                                       Joint, carries
                                                       the outer joint's
                                                       angle.
      acceleration_world  : Vec3[WorldFrame]      — world-frame
                                                       inertial accel
                                                       at the part's
                                                       mount point.
      acceleration_body    : Vec3[CraftFrame]        — same, in body-
                                                       frame coords.
                                                       Specific force
                                                       (accelerometer
                                                       reading) is
                                                       `acceleration_body
                                                       - g_body` where
                                                       `g_body` is sampled
                                                       via `ctx.field(...)`
                                                       and rotated to
                                                       body frame.
      angular_acceleration : Vec3[CraftFrame]        — body α.

    Field access:
      ctx.field(FieldCls) → registered field of that class, or an empty
      default instance if none is registered. The empty default's
      `state_at_sym(p, t)` returns the zero contribution, so a part can
      call `ctx.field(GravityField).state_at_sym(p, t)` regardless of
      whether a GravityField is attached to the world.
      ctx.has_field(FieldCls) → True iff a matching field is registered.

    Note on the acceleration fields: these reflect the **current**
    tick's Newton-Euler output (`α`, `a_origin`) with the lever-arm
    contribution lifted to this part's mount point using current-tick
    ω and `r_in_craft`. At compile time the framework hands `update()`
    MX placeholder symbols for a/α and substitutes the real
    expressions after Newton-Euler. Wrenches must not depend on these
    fields (the substitute pass doesn't solve fixpoints); the compile
    step validates and raises otherwise.
    """

    __slots__ = ("t", "dt", "position", "orientation", "velocity",
                 "angular_velocity", "velocity_body",
                 "R_craft_from_input",
                 "acceleration_world", "acceleration_body",
                 "angular_acceleration",
                 "_world", "_fields")

    def __init__(self,
                 *,
                 t: Scalar,
                 dt: Scalar,
                 position: Vec3,
                 orientation: Quat,
                 velocity: Vec3,
                 angular_velocity: Vec3,
                 velocity_body: Vec3,
                 R_craft_from_input: Mat3,
                 acceleration_world: Vec3,
                 acceleration_body: Vec3,
                 angular_acceleration: Vec3,
                 fields=(),
                 world=None) -> None:
        self.t = t
        self.dt = dt
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.velocity_body = velocity_body
        self._world = world
        self._fields = tuple(fields)
        # Body-frame rotation of the part's INPUT frame. Identity for any
        # part mounted directly on the craft root; for a Mass / Joint
        # child of an outer Joint, this carries the outer joint's angle.
        # Articulated parts use this to express their input-frame axes
        # (e.g., a Joint's spin axis) in body-frame coords.
        self.R_craft_from_input = R_craft_from_input
        self.acceleration_world  = acceleration_world
        self.acceleration_body    = acceleration_body
        self.angular_acceleration = angular_acceleration

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
        `state_at_sym` on the empty default returns the field's zero
        value, so a part can write
            `ctx.field(GravityField).state_at_sym(p, t)`
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


# ---------------------------------------------------------------------------
# Inertial aggregation (pure Python, runs at compile time)
# ---------------------------------------------------------------------------

def _aggregate_inertials(parts: list[Part]) -> dict[str, Any]:
    """Compute total mass, COM offset, and MOI-about-craft-origin from
    a list of parts. Returns concrete Python/numpy values — these are
    constants in the traced graph, not symbolic nodes.

    Walks `part.mass`, `part.moi`, `part.transform` on every part. Parts
    without `mass` (Thruster, sensors, drag surfaces, …) are skipped.
    `Joint` exposes `.mass` and `.moi` as @property aggregates over its
    rotor children, so the body sees the rotor's inertia distribution
    via the same code path as a plain `Mass`.
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

def _wrench_to_craft(wrench_part: Wrench, r_in_craft: Vec3) -> Wrench:
    """Lift a part-emitted wrench to a wrench acting at the craft origin.

    The part's wrench is assumed already expressed in body-frame
    (CraftFrame) coords — any part whose native quantities live in a
    rotated input frame (Joint axis, articulated thruster, …) is
    responsible for rotating them into body-frame inside its own
    `update()` using `ctx.R_craft_from_input`. The lift here is
    purely the parallel-axis-style force-to-torque coupling:

        F_at_origin = F_at_part
        τ_at_origin = τ_at_part + r × F_at_part

    `r_in_craft` is the part's body-frame position (from the
    symbolic kinematic pass), which for a flat craft simplifies to
    `part.transform` and for a part on a joint chain composes through
    the chain.
    """
    if wrench_part.frame is not CraftFrame:
        from .ir.frames import FrameError, _capture_user_source
        raise FrameError(
            "_wrench_to_craft",
            expected="part Wrench in CraftFrame",
            got=f"frame={wrench_part.frame.__name__}",
            source=_capture_user_source(),
        )
    extra_torque = r_in_craft.cross(wrench_part.force)
    return Wrench(
        force=wrench_part.force,
        torque=wrench_part.torque + extra_torque,
    )


# ---------------------------------------------------------------------------
# Craft
# ---------------------------------------------------------------------------

class Craft:
    """A collection of parts with shared rigid-body dynamics.

    Internally a craft is a tree of parts rooted at `Craft.root` (a
    `RootPart`). `craft.add(part)` is sugar for `craft.root.add(part)`;
    `craft.parts` returns a flat tuple of all parts in the tree (DFS
    order). Nested composition (e.g. `Joint` hosting another `Joint`
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
        # Set by World.compile() — used by TickContext helpers
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
    def _parts(self) -> list[Part]:
        """Flat part list used by the world-tick compile / _aggregate
        helpers that haven't yet been refactored to walk the tree."""
        return list(self.parts)

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
        for part in self._parts:
            for nname, ndecl in part.noise_declarations().items():
                sigma = float(getattr(part, f"{nname}_sigma"))
                if isinstance(ndecl, WhiteNoise):
                    key = f"{part.name}.{nname}"
                else:
                    # Inert RW channels (sigma == 0) aren't in the
                    # graph; skip emitting a sample.
                    if sigma <= 0.0:
                        continue
                    key = f"{part.name}.{nname}_driver"
                if ndecl.shape == "scalar":
                    out[key] = (rng.normal(0.0, sigma)
                                if sigma > 0.0 else 0.0)
                else:
                    out[key] = (rng.normal(0.0, sigma, 3)
                                if sigma > 0.0
                                else np.zeros(3, dtype=float))
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
        for part in self._parts:
            for sname, sdecl in part.state_declarations().items():
                if sdecl.manifold == "R1":
                    state[f"{part.name}.{sname}"] = float(sdecl.init)
                else:
                    # R3 — `init` is a 3-tuple, validated at declaration
                    # time. Store as ndarray for symmetry with rigid-body
                    # slots.
                    state[f"{part.name}.{sname}"] = np.asarray(
                        sdecl.init, dtype=float)
            # Input slots: seed from the part's current attribute (which
            # is either the constructor-time override or the declaration
            # default). These pass through CompiledWorld.step's merge so
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
                shape = 3 if ndecl.shape == "vec3" else 1
                zero  = (np.zeros(shape, dtype=float)
                         if shape > 1 else 0.0)
                if isinstance(ndecl, WhiteNoise):
                    state[f"{part.name}.{nname}"] = zero
                else:
                    sigma = float(getattr(part, f"{nname}_sigma"))
                    if sigma <= 0.0:
                        continue
                    state[f"{part.name}.{nname}"] = (
                        np.zeros(shape, dtype=float)
                        if shape > 1 else 0.0)
                    state[f"{part.name}.{nname}_driver"] = (
                        np.zeros(shape, dtype=float)
                        if shape > 1 else 0.0)
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
        parts = ", ".join(p.name for p in self._parts)
        return f"<Craft '{self.name}' parts=[{parts}]>"
