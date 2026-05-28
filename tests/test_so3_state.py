"""Per-Part SO(3) state slots — `State(manifold=SO3Manifold(from, to))`.

The SO(3) manifold for user-declared Part state is wired through
compile_world_tick, initial_state, and StateSpec.from_world. Canonical
use case: an IMU integrator / Madgwick filter / attitude UKF that
carries its own orientation estimate independent of the framework's
rigid-body orientation.
"""

import numpy as np
import pytest

from manta import Craft, TargetNumpy, World
from manta.fields import GravityField
from manta.ir.frames import CraftFrame, PartFrame, WorldFrame
from manta.ir.manifold import SO3 as SO3Value
from manta.ir.slot_manifold import SO3Manifold
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import Mass
from manta.parts.base import Output, Part, PartUpdate, State


# ---------------------------------------------------------------------------
# Test Part: a no-op SO(3) state carrier (passthrough)
# ---------------------------------------------------------------------------

class _SO3PassthroughPart(Part):
    """Carries an SO(3) state slot; passes it through unchanged each
    tick. Exposes it as an Output for round-trip verification."""

    orientation_est = State(
        init=(1.0, 0.0, 0.0, 0.0),
        manifold=SO3Manifold(from_frame=WorldFrame, to_frame=CraftFrame),
    )
    quat_out = Output(shape="vec4")

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"quat_out": self.orientation_est},
            new_state={"orientation_est": self.orientation_est},
        )


# ---------------------------------------------------------------------------
# Declaration validation
# ---------------------------------------------------------------------------

def test_so3_state_requires_both_frames():
    with pytest.raises(ValueError, match="from_frame and to_frame"):
        State(init=(1, 0, 0, 0),
              manifold=SO3Manifold(from_frame=WorldFrame))
    with pytest.raises(ValueError, match="from_frame and to_frame"):
        State(init=(1, 0, 0, 0),
              manifold=SO3Manifold(to_frame=CraftFrame))


def test_so3_state_requires_length_4_init():
    with pytest.raises(ValueError, match="length-4"):
        State(init=(1.0, 0.0, 0.0),
              manifold=SO3Manifold(from_frame=WorldFrame, to_frame=CraftFrame))
    with pytest.raises(ValueError, match="length-4 quaternion"):
        State(init=42,
              manifold=SO3Manifold(from_frame=WorldFrame, to_frame=CraftFrame))


# ---------------------------------------------------------------------------
# Initial state seeds the SO(3) slot as a 4-vector quaternion
# ---------------------------------------------------------------------------

def test_initial_state_seeds_so3_as_quaternion():
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_SO3PassthroughPart("att"))
    init = c.initial_state()
    np.testing.assert_allclose(init["att.orientation_est"],
                                (1.0, 0.0, 0.0, 0.0))


def test_initial_state_override_accepts_quaternion():
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_SO3PassthroughPart("att"))
    # 90° rotation about z: (cos(π/4), 0, 0, sin(π/4))
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    init = c.initial_state(**{"att.orientation_est": q})
    np.testing.assert_allclose(init["att.orientation_est"], q)


# ---------------------------------------------------------------------------
# World + TargetNumpy: passthrough preserves the SO(3) state through ticks
# ---------------------------------------------------------------------------

def test_so3_state_passthrough_through_world_tick():
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_SO3PassthroughPart("att"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    # Seed at a non-identity quaternion (90° about z).
    q0 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    state["c"]["att.orientation_est"] = q0
    for _ in range(20):
        state = sim.step(state, dt=0.01)
    # Normalization is defensive — passthrough preserves the quaternion
    # to within roundoff of the renormalize step.
    np.testing.assert_allclose(state["c"]["att.orientation_est"], q0,
                                atol=1e-12)


def test_so3_state_output_round_trips_the_quaternion():
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_SO3PassthroughPart("att"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    q = np.array([0.5, 0.5, 0.5, 0.5])   # already unit
    state["c"]["att.orientation_est"] = q
    state = sim.step(state, dt=0.01)
    np.testing.assert_allclose(state["c"]["att.quat_out"], q, atol=1e-12)


# ---------------------------------------------------------------------------
# An IMU-integrator-style Part: integrate ω over dt onto an SO(3) state.
# Verify the integrated orientation matches the closed-form quaternion
# product over N small steps.
# ---------------------------------------------------------------------------

class _AttitudeIntegrator(Part):
    """Integrates a body-frame angular rate input onto an SO(3) state.

    Uses the SO(3) value-wrapper's boxplus (i.e. q ← q · exp(ω·dt) in
    the wrapper's left-trivialization). Same pattern an IMU integrator
    or a Madgwick filter would use; this version takes ω as a direct
    Input rather than from a gyro sensor.
    """

    omega_cmd = State(init=(0.0, 0.0, 0.0), manifold="R3", frame=CraftFrame)
    orientation_est = State(
        init=(1.0, 0.0, 0.0, 0.0),
        manifold=SO3Manifold(from_frame=WorldFrame, to_frame=CraftFrame),
    )

    def update(self, ctx) -> PartUpdate:
        # SO3.boxplus uses left trivialization: the tangent vector lives
        # in the SO3's from_frame (WorldFrame here). Rotate the
        # body-frame ω·dt into world coords before integrating —
        # same pattern world_tick uses for the rigid-body orientation.
        delta_body  = self.omega_cmd * ctx.dt
        delta_world = self.orientation_est.apply(delta_body)
        new_so3 = SO3Value.from_quat(self.orientation_est).boxplus(delta_world)
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            new_state={
                "omega_cmd":       self.omega_cmd,
                "orientation_est": new_so3,        # SO3 wrapper — auto-unwraps
            },
        )


def test_attitude_integrator_matches_closed_form_quaternion_product():
    """A constant ω_z for N small steps should integrate to a single
    rotation by N·ω·dt about z."""
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_AttitudeIntegrator("att"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    sim = TargetNumpy(w.compile())
    state = sim.initial_state()
    omega_z = 0.3   # rad/s
    state["c"]["att.omega_cmd"] = np.array([0.0, 0.0, omega_z])
    dt = 0.005
    N = 400        # → final angle = 0.6 rad (~34°)
    for _ in range(N):
        state = sim.step(state, dt=dt)
    # Closed-form: q = (cos(θ/2), 0, 0, sin(θ/2)) where θ = N · ω · dt.
    theta = N * omega_z * dt
    q_expected = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    # Allow some looseness — the integrator is exp(ω·dt) per step in
    # the boxplus convention; for a constant ω, exp(ω·dt)^N = exp(N·ω·dt),
    # exact in the manifold sense up to renormalize roundoff.
    q_got = state["c"]["att.orientation_est"]
    # Quaternion equality up to global sign.
    assert (np.allclose(q_got,  q_expected, atol=1e-9) or
            np.allclose(q_got, -q_expected, atol=1e-9)), (
                f"q_got={q_got}, q_expected={q_expected}")


# ---------------------------------------------------------------------------
# StateSpec picks up the SO(3) slot with 4-ambient / 3-tangent dims
# ---------------------------------------------------------------------------

def test_state_spec_picks_up_so3_state_slot():
    from manta.estimation.state_spec import StateSpec
    c = Craft("c"); c.add(Mass("body", mass=1.0))
    c.add(_SO3PassthroughPart("att"))
    w = World().add_field(GravityField(g=(0, 0, 0)))
    w.add_craft(c)
    spec = StateSpec.from_world(w)
    slot = spec.slot("c.att.orientation_est")
    assert slot.dim              == 4
    assert slot.tangent_dim      == 3
    assert slot.manifold.kind    == "quat"
