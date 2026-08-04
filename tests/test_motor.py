"""Motor — voltage-commanded DC motor on a revolute DOF.

Physics checks against the closed-form quasi-static DC-motor model
(k = torque_constant, R = resistance, J = rotor axial MOI, direct
drive):

    τ = k·(V − k·ω)/R        ω_nl = V/k        τ_mech = J·R/k²
"""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, Motor

K_T   = 0.05      # N·m/A (= V·s/rad)
RES   = 1.0       # Ω
J_Z   = 0.002     # rotor MOI about the motor axis
TAU_M = J_Z * RES / K_T**2   # mechanical time constant, s (= 0.8)


def _motor_world(**motor_overrides):
    """A motor-driven rotor on a body heavy enough to approximate a
    fixed stator (body MOI 5000× the rotor's)."""
    c = Craft("rig")
    c.add(Mass("body", mass=50.0, moi=(10.0, 10.0, 10.0)))
    m = c.add(Motor("m", axis=(0.0, 0.0, 1.0),
                    torque_constant=K_T, resistance=RES,
                    **motor_overrides))
    m.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, J_Z),
               transform=(0.0, 0.0, 0.05)))
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


def test_no_load_spinup_reaches_v_over_k():
    """Step voltage: the rate converges to the no-load speed V/k."""
    V = 1.0
    sim = TargetNumpy(Sim(_motor_world()))
    sim.state["rig"]["m.voltage"] = V
    dt = 0.001
    for _ in range(int(5 * TAU_M / dt)):        # 5 time constants
        sim.step(dt)
    omega = float(sim.state["rig"]["m.rate"])
    assert np.isclose(omega, V / K_T, rtol=0.02)


def test_spinup_follows_first_order_time_constant():
    """The spin-up transient is first-order with τ = J·R/k²:
    ω(τ) ≈ (1 − 1/e)·ω_nl."""
    V = 1.0
    sim = TargetNumpy(Sim(_motor_world()))
    sim.state["rig"]["m.voltage"] = V
    dt = 0.001
    for _ in range(int(TAU_M / dt)):
        sim.step(dt)
    omega = float(sim.state["rig"]["m.rate"])
    assert np.isclose(omega, (1.0 - np.exp(-1.0)) * V / K_T, rtol=0.03)


def test_zero_voltage_dynamic_braking():
    """Unpowered motor: the shorted winding brakes the shaft —
    ω decays exponentially instead of freewheeling."""
    omega0 = 20.0
    sim = TargetNumpy(Sim(_motor_world()))
    sim.state["rig"]["m.rate"] = omega0
    dt = 0.001
    for _ in range(int(TAU_M / dt)):
        sim.step(dt)
    omega = float(sim.state["rig"]["m.rate"])
    assert np.isclose(omega, omega0 * np.exp(-1.0), rtol=0.03)


def test_current_limit_caps_stall_torque():
    """From rest the unlimited inrush current is V/R = 1 A; a 0.2 A
    limit caps the initial torque (and so the early spin-up slope) at
    exactly limit/inrush of the unlimited one."""
    V, i_lim = 1.0, 0.2
    dt, n = 0.001, 20     # early window: rate stays ≪ back-EMF knee

    free = TargetNumpy(Sim(_motor_world()))
    free.state["rig"]["m.voltage"] = V
    lim = TargetNumpy(Sim(_motor_world(current_limit=i_lim)))
    lim.state["rig"]["m.voltage"] = V
    for _ in range(n):
        free.step(dt)
        lim.step(dt)
    ratio = float(lim.state["rig"]["m.rate"]) / float(
        free.state["rig"]["m.rate"])
    assert np.isclose(ratio, i_lim / (V / RES), rtol=0.02)


def test_gear_ratio_scales_stall_torque():
    """A G:1 reduction multiplies the from-rest shaft torque by G
    (early spin-up slope ∝ G at fixed rotor inertia). The window must
    be very short: gearing also shrinks the mechanical time constant by
    G², so the geared back-EMF knee arrives 16× sooner."""
    V, G = 1.0, 4.0
    dt, n = 0.001, 2

    direct = TargetNumpy(Sim(_motor_world()))
    direct.state["rig"]["m.voltage"] = V
    geared = TargetNumpy(Sim(_motor_world(gear_ratio=G)))
    geared.state["rig"]["m.voltage"] = V
    for _ in range(n):
        direct.step(dt)
        geared.step(dt)
    ratio = float(geared.state["rig"]["m.rate"]) / float(
        direct.state["rig"]["m.rate"])
    assert np.isclose(ratio, G, rtol=0.05)


def test_free_craft_conserves_angular_momentum():
    """The motor torque is internal (rotor↔body exchange): a
    free-floating craft's total angular momentum stays zero while the
    motor spins up — the body counter-rotates."""
    c = Craft("sat")
    c.add(Mass("body", mass=1.0, moi=(0.05, 0.05, 0.05)))
    m = c.add(Motor("wheel", axis=(0.0, 0.0, 1.0),
                    torque_constant=K_T, resistance=RES))
    m.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, J_Z)))
    w = World()          # no gravity, free floating
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    sim.state["sat"]["wheel.voltage"] = 1.0
    for _ in range(500):
        sim.step(0.001)
    omega_body = np.array(sim.state["sat"]["angular_velocity"]).ravel()
    rate_rel = float(sim.state["sat"]["wheel.rate"])
    # Rotor spun up, body counter-rotates.
    assert rate_rel > 1.0
    assert omega_body[2] < -1e-4
    # L_z = I_body·ω_z + J_rotor·(ω_z + rate) = 0.
    L_z = 0.05 * omega_body[2] + J_Z * (omega_body[2] + rate_rel)
    assert abs(L_z) < 1e-6


def test_motor_rejects_mode():
    with pytest.raises(TypeError, match="mode"):
        Motor("m", mode="passive")


def test_motor_rejects_nonpositive_constants():
    with pytest.raises(ValueError, match="resistance"):
        Motor("m", resistance=0.0)
    with pytest.raises(ValueError, match="torque_constant"):
        Motor("m", torque_constant=-1.0)


def test_joint_subclass_extra_state_survives():
    """A joint subclass declaring its OWN state (beyond the DOF pair)
    keeps it: only the DOF slots are integrated centrally, everything
    else follows the normal part-state path. Regression: the world tick
    once skipped ALL states on an ArticulatedJoint, so an extra slot was
    declared as a graph input but never emitted — a KeyError deep in the
    linearization engine, contradicting joint.py's 'a new actuation
    model is just another RevoluteDOF subclass' promise."""
    import math

    from manta.ir.types import Scalar
    from manta.parts._declarations import State
    from manta.parts._trace import scalar_mx

    class RevCounter(Motor):
        """Motor with an output-shaft revolution counter state."""
        revs = State(init=0.0, manifold="R1")

        def update(self, ctx):
            base = super().update(ctx)
            rate = scalar_mx(self.rate)
            revs = scalar_mx(self.revs)
            base.new_state["revs"] = Scalar(
                revs + scalar_mx(ctx.dt) * rate / (2.0 * math.pi))
            return base

    c = Craft("rig")
    c.add(Mass("body", mass=50.0, moi=(10.0, 10.0, 10.0)))
    m = c.add(RevCounter("m", axis=(0.0, 0.0, 1.0),
                         torque_constant=K_T, resistance=RES))
    m.add(Mass("rotor", mass=0.1, moi=(0.001, 0.001, J_Z),
               transform=(0.0, 0.0, 0.05)))
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)

    sim = TargetNumpy(Sim(w))
    for _ in range(200):
        sim.step(0.01, u={"m.voltage": 1.0})
    revs = float(sim.state["rig"]["m.revs"])
    rate = float(sim.state["rig"]["m.rate"])
    assert rate > 1.0, "motor did not spin up"
    assert revs > 0.0, "extra state never integrated"


def test_motor_parameters_are_promotable():
    """torque_constant / resistance promote to a live params port for
    system ID."""
    w = _motor_world()
    sim = Sim(w, parameters=["m.torque_constant", "m.resistance"])
    mod = sim.module()
    port = mod.port("params")
    names = [f.name for f in port.fields]
    assert "rig.m.torque_constant" in names
    assert "rig.m.resistance" in names
