"""PrismaticJoint — 1-DOF sliding joint, pinned against analytic oracles.

The slide DOF reads the axial component of the subtree's external force
(so gravity along the axis drives a piston at g on a heavy base), the
about-COM centrifugal transport (a radial slider on a spinning body is
flung outward at d̈ = ω²ρ), and the actuator/damping terms. The pass-up
wrench is unchanged — the moving-COM machinery closes the linear books,
so a free-floating craft firing an internal piston conserves linear
momentum to machine precision.

Tier note (mirrors the revolute fixed-base approximation): the mount's
LINEAR-acceleration feedback is omitted, so these oracles use heavy /
momentum-balanced bases — the same tier at which the passive-pendulum
tests treat their heavy anchor as an inertial pivot.
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PrismaticJoint, RevoluteJoint


G = 9.81


def _sim(craft, fields=(), **overrides):
    w = World()
    for f in fields:
        w.add_field(f)
    w.add_craft(craft, **overrides)
    return TargetNumpy(Sim(w))


def _slider_craft(hub_mass=100.0, bead_mass=0.5, axis=(1.0, 0.0, 0.0),
                  mode="passive", **joint_kw):
    c = Craft("rig")
    c.add(Mass("hub", mass=hub_mass, moi=(50.0, 50.0, 50.0)))
    s = PrismaticJoint("slide", mode=mode, axis=axis, **joint_kw)
    s.add(Mass("bead", mass=bead_mass, moi=(1e-9, 1e-9, 1e-9)))
    c.add(s)
    return c


# ---------------------------------------------------------------------------
# Centrifugal extension: d̈ = ω²·ρ (about the system COM, exactly)
# ---------------------------------------------------------------------------

def test_centrifugal_extension_matches_omega_squared_rho():
    """A bead on a radial slide of a body spinning at ω about z is flung
    outward at d̈ = ω²·ρ, where ρ = d·(1 − m/m_total) is the bead's
    distance from the SYSTEM COM (the hub circles the COM too)."""
    hub_m, bead_m = 100.0, 0.5
    omega, d0 = 2.0, 0.3
    rt = _sim(_slider_craft(hub_m, bead_m),
              **{"slide.displacement": d0})
    rt.state["rig"]["angular_velocity"] = np.array([0.0, 0.0, omega])
    dt = 1e-5
    rt.step(dt)
    d_ddot = rt.state["rig"]["slide.rate"] / dt
    rho = d0 * (1.0 - bead_m / (hub_m + bead_m))
    np.testing.assert_allclose(d_ddot, omega**2 * rho, rtol=1e-6)


def test_centrifugal_scales_with_omega_squared():
    def d_ddot(omega):
        rt = _sim(_slider_craft(1000.0, 0.1),
                  **{"slide.displacement": 0.4})
        rt.state["rig"]["angular_velocity"] = np.array([0.0, 0.0, omega])
        dt = 1e-5
        rt.step(dt)
        return rt.state["rig"]["slide.rate"] / dt

    np.testing.assert_allclose(d_ddot(6.0), 4.0 * d_ddot(3.0), rtol=1e-6)


# ---------------------------------------------------------------------------
# Free piston: linear momentum conservation + recoil ratio
# ---------------------------------------------------------------------------

def test_free_piston_recoil_conserves_linear_momentum():
    """A free-floating body firing an internal piston: the body recoils
    so that  M_total·v_com = 0  exactly — the moving-COM machinery, fed
    by the slide's d̈ placeholder, closes the linear books."""
    body_m, slug_m, F = 10.0, 1.0, 5.0
    c = Craft("rec")
    c.add(Mass("body", mass=body_m, moi=(5.0, 5.0, 5.0)))
    gun = PrismaticJoint("gun", mode="saturating", stall_force=1e9,
                         axis=(1.0, 0.0, 0.0))
    gun.add(Mass("slug", mass=slug_m, moi=(1e-9, 1e-9, 1e-9)))
    c.add(gun)
    rt = _sim(c, **{"gun.force_cmd": F})
    dt = 1e-3
    for _ in range(200):
        rt.step(dt)
    v_body = float(rt.state["rec"]["velocity"][0])
    d_rate = float(rt.state["rec"]["gun.rate"])
    # Body is unrotated, so slug world velocity = v_body + ḋ.
    p_total = body_m * v_body + slug_m * (v_body + d_rate)
    assert abs(p_total) < 1e-9, f"linear momentum leaked: {p_total:.3e}"
    # Recoil ratio |v_body| / ḋ_world: M·v = −m·v_slug → v_body/v_slug = −m/M.
    v_slug = v_body + d_rate
    np.testing.assert_allclose(v_body / v_slug, -slug_m / body_m, rtol=1e-9)


# ---------------------------------------------------------------------------
# Gravity along / across the slide axis
# ---------------------------------------------------------------------------

def test_gravity_driven_slide_falls_at_g_on_heavy_base():
    """Vertical slide on a heavy base: the piston's first-tick d̈ is −g
    (the heavy-anchor tier — the same idealization as the passive
    pendulum's θ̈ = −(g/L)sinθ)."""
    c = Craft("drop")
    c.add(Mass("base", mass=1e6, moi=(1e6, 1e6, 1e6)))
    p = PrismaticJoint("piston", mode="passive", axis=(0.0, 0.0, 1.0))
    p.add(Mass("plug", mass=2.0, moi=(1e-9, 1e-9, 1e-9)))
    c.add(p)
    rt = _sim(c, fields=[GravityField(g=(0.0, 0.0, -G))])
    dt = 1e-5
    rt.step(dt)
    np.testing.assert_allclose(rt.state["drop"]["piston.rate"] / dt, -G,
                               rtol=1e-6)


def test_gravity_across_slide_axis_does_not_drive_it():
    """Gravity ⟂ axis has no axial component — the slide holds."""
    c = Craft("hold")
    c.add(Mass("base", mass=1e3, moi=(1e3, 1e3, 1e3)))
    p = PrismaticJoint("rail", mode="passive", axis=(1.0, 0.0, 0.0))
    p.add(Mass("cart", mass=1.0, moi=(1e-9, 1e-9, 1e-9)))
    c.add(p)
    rt = _sim(c, fields=[GravityField(g=(0.0, 0.0, -G))],
              **{"rail.displacement": 0.2})
    dt = 1e-3
    for _ in range(100):
        rt.step(dt)
    np.testing.assert_allclose(rt.state["hold"]["rail.displacement"], 0.2,
                               atol=1e-9)
    np.testing.assert_allclose(rt.state["hold"]["rail.rate"], 0.0, atol=1e-9)


def test_deflected_slide_holds_without_force():
    """No gravity, no actuation, displaced slide on a stationary craft:
    nothing moves, forever (no spurious wrench from the deflection)."""
    rt = _sim(_slider_craft(), **{"slide.displacement": 0.3})
    dt = 1e-3
    for _ in range(100):
        rt.step(dt)
    np.testing.assert_allclose(rt.state["rig"]["slide.displacement"], 0.3,
                               atol=1e-12)
    np.testing.assert_allclose(rt.state["rig"]["velocity"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Actuator clamp + viscous damping
# ---------------------------------------------------------------------------

def test_saturating_clamp_limits_force():
    """force_cmd far above stall_force accelerates the slide at
    stall_force/m, not force_cmd/m."""
    stall, m = 0.5, 2.0
    c = Craft("sat")
    c.add(Mass("base", mass=1e6, moi=(1e6, 1e6, 1e6)))
    p = PrismaticJoint("ram", mode="saturating", stall_force=stall,
                       axis=(1.0, 0.0, 0.0))
    p.add(Mass("rod", mass=m, moi=(1e-9, 1e-9, 1e-9)))
    c.add(p)
    rt = _sim(c, **{"ram.force_cmd": 100.0})
    dt = 1e-3
    for _ in range(100):
        rt.step(dt)
    np.testing.assert_allclose(rt.state["sat"]["ram.rate"],
                               (stall / m) * 0.1, rtol=1e-6)


def test_damping_decays_rate_exponentially():
    """m·d̈ = −c·ḋ → ḋ(t) = ḋ₀·e^(−c·t/m)."""
    m, cdamp = 2.0, 4.0
    c = Craft("damp")
    c.add(Mass("base", mass=1e6, moi=(1e6, 1e6, 1e6)))
    p = PrismaticJoint("rail", mode="passive", axis=(1.0, 0.0, 0.0),
                       damping=cdamp)
    p.add(Mass("cart", mass=m, moi=(1e-9, 1e-9, 1e-9)))
    c.add(p)
    rt = _sim(c, **{"rail.rate": 1.0})
    dt = 1e-4
    T = 0.5
    for _ in range(int(T / dt)):
        rt.step(dt)
    np.testing.assert_allclose(rt.state["damp"]["rail.rate"],
                               np.exp(-cdamp * T / m), rtol=2e-3)


# ---------------------------------------------------------------------------
# Sensor on the slide: the d̈ placeholder reaches rider accelerometers
# ---------------------------------------------------------------------------

def test_imu_on_slide_reads_slide_acceleration():
    """An IMU riding an actuated piston reads a_body + d̈ along the axis:
    with F internal, d̈_rel = F/m and a_body = −F/(M+m), so the absolute
    (= specific, no gravity) acceleration is F/m − F/(M+m)."""
    M, m, F = 9.0, 1.0, 2.0
    c = Craft("probe")
    c.add(Mass("body", mass=M, moi=(5.0, 5.0, 5.0)))
    p = PrismaticJoint("piston", mode="saturating", stall_force=1e9,
                       axis=(1.0, 0.0, 0.0))
    p.add(Mass("head", mass=m, moi=(1e-9, 1e-9, 1e-9)))
    p.add(IMU("imu"))
    c.add(p)
    rt = _sim(c, **{"piston.force_cmd": F})
    rt.step(1e-5)
    accel = np.asarray(rt.outputs()["probe"]["imu.accel"]).ravel()
    expected = F / m - F / (M + m)
    np.testing.assert_allclose(accel[0], expected, rtol=1e-6)
    np.testing.assert_allclose(accel[1:], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Composition: prismatic under a spinning revolute (nested chain smoke)
# ---------------------------------------------------------------------------

def test_prismatic_under_revolute_flung_at_pan_rate_squared():
    """A slide riding a spinning pan joint (body at rest): the upstream
    joint's velocity products reach the slide DOF through the subtree
    a_rel reduction, so the tip is flung outward at the PAN's ω²·d even
    though the body itself isn't rotating."""
    omega, d0 = 3.0, 0.4
    c = Craft("turret")
    c.add(Mass("hub", mass=50.0, moi=(20.0, 20.0, 20.0)))
    pan = RevoluteJoint("pan", mode="passive", axis=(0.0, 0.0, 1.0))
    pan.add(Mass("arm", mass=1.0, moi=(0.1, 0.1, 0.1)))
    rail = PrismaticJoint("rail", mode="passive", axis=(1.0, 0.0, 0.0))
    rail.add(Mass("tip", mass=0.2, moi=(1e-9, 1e-9, 1e-9)))
    pan.add(rail)
    c.add(pan)
    rt = _sim(c, **{"pan.rate": omega, "rail.displacement": d0})
    dt = 1e-5
    rt.step(dt)
    d_ddot = rt.state["turret"]["rail.rate"] / dt
    np.testing.assert_allclose(d_ddot, omega**2 * d0, rtol=1e-6)
    # Longer run stays finite and the torque-free craft doesn't run away.
    for _ in range(200):
        rt.step(1e-4)
    st = rt.state["turret"]
    assert np.all(np.isfinite(st["velocity"]))
    assert st["rail.rate"] > 0.0
    assert np.linalg.norm(st["velocity"]) < 0.1


# ---------------------------------------------------------------------------
# Construction-time validation (inherited from ArticulatedJoint)
# ---------------------------------------------------------------------------

def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        PrismaticJoint("bad", mode="nonsense")


def test_dof_state_names():
    p = PrismaticJoint("p")
    r = RevoluteJoint("r")
    assert p.dof_state_names() == ("displacement", "rate")
    assert r.dof_state_names() == ("angle", "rate")
