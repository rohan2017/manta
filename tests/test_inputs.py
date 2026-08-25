"""Per-tick Input declarations — values supplied by the caller each step.

Uses a `RevoluteJoint` with a single rotor `Mass` child to exercise the
torque_cmd Input across the integration paths (World.step, EKF.predict).
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.estimation import EKF
from manta.fields import GravityField
from manta.ir.frames import PartFrame, WorldFrame
from manta.parts import (
    Input,
    Mass,
    Part,
    RevoluteJoint,
    Wrench,
)


def _flywheel(name: str, I_axial: float, **kw) -> RevoluteJoint:
    """Make a saturating RevoluteJoint with a single rotor Mass whose MOI about
    the spin axis equals I_axial. Used by the per-tick-input tests
    below that previously instantiated FlywheelMotor."""
    j = RevoluteJoint(name, mode="saturating", stall_torque=1e9, **kw)
    j.add(Mass(f"{name}_rotor", mass=1e-6, moi=(0.0, 0.0, I_axial)))
    return j


# ---------------------------------------------------------------------------
# Default Input value seeds the state
# ---------------------------------------------------------------------------

def test_input_default_appears_in_initial_state():
    c = Craft("with_input")
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    c.add(_flywheel("motor", I_axial=0.01))

    state = c.initial_state()
    assert "motor.torque_cmd" in state
    assert state["motor.torque_cmd"] == 0.0


def test_input_constructor_override_seeds_state():
    """A torque_cmd kwarg sets the default value used when nothing else
    is specified — it's not a frozen constant."""
    c = Craft("preseeded")
    c.add(Mass("body", mass=1.0))
    c.add(_flywheel("motor", I_axial=0.01, torque_cmd=0.3))

    state = c.initial_state()
    assert state["motor.torque_cmd"] == 0.3


# ---------------------------------------------------------------------------
# Inputs change the dynamics per tick
# ---------------------------------------------------------------------------

def test_zero_torque_default_keeps_flywheel_at_rest():
    c = Craft("idle")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    c.add(_flywheel("motor", I_axial=0.01))    # default torque_cmd=0

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    for _ in range(1000):
        sim.step(dt=0.001)

    assert np.isclose(sim.state["idle"]["motor.rate"],  0.0, atol=1e-12)
    assert np.isclose(sim.state["idle"]["motor.angle"], 0.0, atol=1e-12)


def test_per_tick_torque_drives_flywheel():
    c = Craft("driven")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    c.add(_flywheel("motor", I_axial=0.01))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))

    # First second: idle (default 0 torque).
    for _ in range(1000):
        sim.step(dt=0.001)
    assert np.isclose(sim.state["driven"]["motor.rate"], 0.0, atol=1e-9)

    # Switch torque on — input persists between steps because of the merge.
    sim.state["driven"]["motor.torque_cmd"] = 1.0
    for _ in range(1000):
        sim.step(dt=0.001)
    # α = 1.0 / 0.01 = 100 rad/s² for 1 s ⇒ ω = 100.
    assert np.isclose(sim.state["driven"]["motor.rate"], 100.0, atol=1e-6)

    # Switch torque off — rate stays at 100 (no friction).
    sim.state["driven"]["motor.torque_cmd"] = 0.0
    for _ in range(1000):
        sim.step(dt=0.001)
    assert np.isclose(sim.state["driven"]["motor.rate"], 100.0, atol=1e-6)


def test_input_value_can_change_each_tick():
    """A controller-style loop where the input varies each step."""
    c = Craft("ramping")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    c.add(_flywheel("motor", I_axial=0.01))

    w = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))

    # Apply a sinusoidal torque over 2 seconds.
    dt = 0.001
    integral = 0.0
    for i in range(2000):
        tau = np.sin(2 * np.pi * i * dt / 2.0)   # 0.5 Hz sine
        sim.state["ramping"]["motor.torque_cmd"] = float(tau)
        integral += tau * dt
        sim.step(dt=dt)

    # The integral of a full sine cycle over its period is ~0, so the
    # flywheel rate ends near zero.
    expected_rate = integral / 0.01   # ω = ∫τ/I dt
    assert np.isclose(sim.state["ramping"]["motor.rate"], expected_rate, atol=1e-3)


# ---------------------------------------------------------------------------
# World + Inputs
# ---------------------------------------------------------------------------

def test_world_carries_inputs_through_step():
    """Sim.step's merge pattern carries Inputs across ticks."""
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    c = Craft("driven")
    c.add(Mass("body", mass=100.0, moi=(1000.0, 1000.0, 1000.0)))
    c.add(_flywheel("motor", I_axial=0.05))
    w.add_craft(c)

    cw = TargetNumpy(Sim(w))
    cw.state["driven"]["motor.torque_cmd"] = 1.0

    for _ in range(500):
        cw.step(dt=0.001)

    # Relative rate = τ/I_axial + τ/I_stator (the −â·α mount feedback):
    # (20 + 0.001) rad/s² · 0.5 s = 10.0005 rad/s.
    assert np.isclose(cw.state["driven"]["motor.rate"],
                      (1.0 / 0.05 + 1.0 / 1000.0) * 0.5, atol=1e-4)


# ---------------------------------------------------------------------------
# EKF can compile with Input parts
# ---------------------------------------------------------------------------

def test_ekf_constructs_with_input_parts():
    """The EKF wrapper can compile a craft that has Input slots; for now
    inputs are baked in at their declared default. Per-tick u in the EKF
    is a future extension."""
    c = Craft("driven_ekf")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(_flywheel("motor", I_axial=0.05, torque_cmd=0.3))

    _ekf_world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    _ekf_world.add_craft(c)
    ekf = TargetNumpy(EKF(_ekf_world))
    ekf.predict(dt=0.01)
    # τ=0.3: rotor 6 rad/s² + the stator counter-rotating at
    # 0.3/(0.15−0.05) = 3 rad/s² (coupled solve) → relative rate after
    # 0.01 s = (6 + 3)·0.01 = 0.09.
    assert np.isclose(ekf.state_dict()["driven_ekf"]["motor.rate"], 0.09,
                      atol=1e-9)


# ---------------------------------------------------------------------------
# Multiple Inputs on one part (no RevoluteJoint dependency)
# ---------------------------------------------------------------------------

def test_multiple_inputs_on_one_part():
    from manta.ir.types import Vec3

    class TwoInputPart(Part):
        scale_a: float = Input(default=1.0)
        scale_b: float = Input(default=2.0)

        def update(self, ctx):
            from manta.fields import GravityField
            g_world = ctx.field(GravityField).value_at_sym(
                ctx.position[WorldFrame], ctx.t)
            g_part  = ctx.orientation.conjugate().apply(g_world)
            f = g_part * (self.scale_a + self.scale_b)
            zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
            return Wrench(force=f, torque=zero)

    c = Craft("two_in")
    c.add(Mass("body", mass=1.0))
    c.add(TwoInputPart("p"))
    init = c.initial_state()
    assert init["p.scale_a"] == 1.0
    assert init["p.scale_b"] == 2.0

    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    state = sim.step(dt=0.001)
    assert "position" in state["two_in"]


def test_input_decl_introspection():
    motor = _flywheel("m", I_axial=0.02)
    decls = motor.input_declarations()
    assert set(decls.keys()) == {"torque_cmd"}
    assert decls["torque_cmd"].default == 0.0
