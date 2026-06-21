"""ControlSurface — combined wing+flap aero, one-state quasi-static servo
with saturation and blowback, and the flat-craft (no-joint) fast path."""

import math

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import ControlSurface, Mass
from manta.parts.aero.control_surface import _flap_effectiveness


def _craft(surface, velocity, fluid=None, mass=3.0, moi=(0.6, 0.6, 0.6)):
    w = (World()
         .add_field(GravityField().add_uniform((0, 0, 0)))
         .add_field(fluid if fluid is not None
                    else FluidField().add_uniform(density=1.225)))
    c = Craft("w")
    c.add(Mass("body", mass=mass, moi=moi))
    c.add(surface)
    w.add_craft(c, velocity=velocity)
    return w


def _run(world, cmd_deg, steps=400, dt=0.002, name="surf"):
    sim = TargetNumpy(Sim(world))
    for _ in range(steps):
        st = sim.state["w"]
        st[f"{name}.deflection_cmd"] = math.radians(cmd_deg)
        sim.step(dt)
    return sim


def _surface(name="surf", **kw):
    base = dict(area=0.4, chord=0.25, flap_chord_fraction=0.3,
                chord_axis=(1, 0, 0), normal_axis=(0, 0, 1))
    base.update(kw)
    return ControlSurface(name, **base)


# --- flap effectiveness -----------------------------------------------------

def test_flap_effectiveness_quarter_chord():
    """A 25%-chord flap has τ ≈ 0.6 (classic thin-airfoil value) and a
    nose-down moment slope."""
    tau, cm_d = _flap_effectiveness(0.25)
    np.testing.assert_allclose(tau, 0.6, atol=0.05)
    assert cm_d < 0.0


def test_flap_effectiveness_monotonic():
    """Lift effectiveness τ grows monotonically with flap size; the moment
    slope is nose-down throughout (its magnitude is NOT monotonic — it
    vanishes at both E→0 and E→1 — so only the sign is asserted)."""
    t_small, c_small = _flap_effectiveness(0.15)
    t_big, c_big = _flap_effectiveness(0.40)
    assert t_small < t_big
    assert c_small < 0.0 and c_big < 0.0


# --- servo tracking ---------------------------------------------------------

def test_deflection_tracks_command():
    """At modest airspeed the servo drives the surface to its commanded
    angle (small aero droop tolerated)."""
    sim = _run(_craft(_surface(), velocity=(15.0, 0, 0)), cmd_deg=10.0)
    got = math.degrees(float(sim.state["w"]["surf.deflection"]))
    np.testing.assert_allclose(got, 10.0, atol=0.5)


def test_deflection_state_is_single_dof_no_joint():
    """A ControlSurface adds exactly ONE state (the deflection) and no
    articulated joint DOFs — the craft stays on the flat-body fast path."""
    w = _craft(_surface(), velocity=(15.0, 0, 0))
    spec = Sim(w).sys.spec
    assert spec.tangent_dim == 13          # 12 rigid + 1 deflection
    sim = TargetNumpy(Sim(w))
    slots = set(sim.state["w"].keys())
    assert "surf.deflection" in slots
    assert not any(k.endswith(".angle") or k.endswith(".rate") for k in slots)


# --- combined aero ----------------------------------------------------------

def test_deflection_increases_section_lift():
    """Deflecting the flap down raises the whole section's lift — the
    coupling a two-foil hinge model misses."""
    neutral = _run(_craft(_surface(), velocity=(20.0, 0, 0)), cmd_deg=0.0,
                   steps=1)
    deflected = _run(_craft(_surface(), velocity=(20.0, 0, 0)), cmd_deg=15.0,
                     steps=80)
    vz_neutral = np.array(neutral.state["w"]["velocity"]).ravel()[2]
    vz_defl = np.array(deflected.state["w"]["velocity"]).ravel()[2]
    assert vz_defl > vz_neutral + 1e-3     # extra upward lift


# --- saturation / blowback --------------------------------------------------

def test_blowback_loses_authority_at_high_airspeed():
    """An under-powered servo holds its command at low speed but gets
    blown back toward neutral when the aero hinge moment overpowers the
    stall torque at high speed. (servo_gain/hinge_damping are set so the
    surface slews to command well within the run; stall_torque is the
    authority that high-q overpowers.)"""
    weak = dict(servo_gain=40.0, stall_torque=2.0, hinge_damping=0.2)
    slow = _run(_craft(_surface(**weak), velocity=(10.0, 0, 0),
                       mass=10.0, moi=(3.0, 3.0, 3.0)),
                cmd_deg=20.0, steps=300)
    fast = _run(_craft(_surface(**weak), velocity=(120.0, 0, 0),
                       mass=80.0, moi=(40.0, 40.0, 40.0)),
                cmd_deg=20.0, steps=300)
    d_slow = abs(math.degrees(float(slow.state["w"]["surf.deflection"])))
    d_fast = abs(math.degrees(float(fast.state["w"]["surf.deflection"])))
    assert d_slow > 15.0                   # near the 20° command
    assert d_fast < 0.4 * d_slow           # blown back


def test_blowback_stronger_in_denser_fluid():
    """Same surface, same speed: a dense fluid (water) blows the surface
    back harder than thin air, since the hinge moment scales with ½ρV²."""
    weak = dict(servo_gain=40.0, stall_torque=2.0, hinge_damping=0.2)
    air = _run(_craft(_surface(**weak), velocity=(12.0, 0, 0),
                      mass=10.0, moi=(3.0, 3.0, 3.0),
                      fluid=FluidField().add_uniform(density=1.225)),
               cmd_deg=20.0, steps=300)
    water = _run(_craft(_surface(**weak), velocity=(12.0, 0, 0),
                        mass=80.0, moi=(40.0, 40.0, 40.0),
                        fluid=FluidField().add_uniform(density=1000.0,
                                                       viscosity=1e-3)),
                 cmd_deg=20.0, steps=300)
    d_air = abs(math.degrees(float(air.state["w"]["surf.deflection"])))
    d_water = abs(math.degrees(float(water.state["w"]["surf.deflection"])))
    assert d_water < d_air


# --- validation -------------------------------------------------------------

def test_control_surface_validation():
    with pytest.raises(ValueError):
        ControlSurface("s", area=0.4, chord=0.25, flap_chord_fraction=0.0)
    with pytest.raises(ValueError):
        ControlSurface("s", area=0.4, chord=0.25, flap_chord_fraction=1.0)
    with pytest.raises(ValueError):
        ControlSurface("s", area=0.4, chord=0.25, hinge_damping=0.0)
