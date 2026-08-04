"""Thermal simulation — the ThermalMass lumped network.

Purely coefficient-based (no spatial field): conduction links between
nodes, an ambient boundary input, heat generation. Physics checks
against closed-form lumped-capacitance solutions:

    two nodes, link k:   T_a − T_b decays with τ = C_a·C_b/(k·(C_a+C_b))
    heated node:         dT/dt = P/C                (linear ramp)
    ambient coupling:    T → T_amb with τ = C/g_amb
"""

import numpy as np
import pytest

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import Mass, Motor, ThermalMass


def _rig(*nodes):
    """A one-craft world hosting the given thermal nodes (no gravity —
    thermal tests don't need dynamics)."""
    c = Craft("rig")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    for n in nodes:
        c.add(n)
    w = World()
    w.add_craft(c)
    return w, c


# ---------------------------------------------------------------------------
# Conduction network
# ---------------------------------------------------------------------------

def test_two_nodes_equilibrate_to_energy_weighted_mean():
    """Hot and cold node conduct until both sit at the C-weighted mean;
    total thermal energy Σ C·T is conserved throughout."""
    a = ThermalMass("a", heat_capacity=100.0)
    b = ThermalMass("b", heat_capacity=300.0)
    a.connect(b, conductance=20.0)
    w, _ = _rig(a, b)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["a.temperature"] = 350.0
    sim.state["rig"]["b.temperature"] = 290.0
    E0 = 100.0 * 350.0 + 300.0 * 290.0

    dt = 0.01
    for _ in range(20_000):        # ≫ τ = (100·300)/(20·400) = 3.75 s
        sim.step(dt)

    Ta = float(sim.state["rig"]["a.temperature"])
    Tb = float(sim.state["rig"]["b.temperature"])
    T_eq = E0 / 400.0              # 305 K
    assert np.isclose(Ta, T_eq, atol=0.01)
    assert np.isclose(Tb, T_eq, atol=0.01)
    assert np.isclose(100.0 * Ta + 300.0 * Tb, E0, rtol=1e-9)


def test_two_node_decay_time_constant():
    """The temperature difference decays exponentially with
    τ = C_a·C_b / (k·(C_a + C_b))."""
    a = ThermalMass("a", heat_capacity=100.0)
    b = ThermalMass("b", heat_capacity=100.0)
    a.connect(b, conductance=25.0)
    tau = (100.0 * 100.0) / (25.0 * 200.0)     # 2.0 s
    w, _ = _rig(a, b)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["a.temperature"] = 320.0
    sim.state["rig"]["b.temperature"] = 300.0

    dt = 0.001
    for _ in range(int(tau / dt)):
        sim.step(dt)

    dT = (float(sim.state["rig"]["a.temperature"])
          - float(sim.state["rig"]["b.temperature"]))
    assert np.isclose(dT, 20.0 * np.exp(-1.0), rtol=0.01)


def test_three_node_chain_conserves_energy():
    """A chain a—b—c with no ambient/source: Σ C·T is invariant."""
    a = ThermalMass("a", heat_capacity=50.0)
    b = ThermalMass("b", heat_capacity=120.0)
    c = ThermalMass("c", heat_capacity=80.0)
    a.connect(b, conductance=10.0)
    b.connect(c, conductance=4.0)
    w, _ = _rig(a, b, c)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["a.temperature"] = 400.0
    sim.state["rig"]["b.temperature"] = 300.0
    sim.state["rig"]["c.temperature"] = 250.0
    E0 = 50.0 * 400.0 + 120.0 * 300.0 + 80.0 * 250.0

    for _ in range(5000):
        sim.step(0.01)

    E = (50.0 * float(sim.state["rig"]["a.temperature"])
         + 120.0 * float(sim.state["rig"]["b.temperature"])
         + 80.0 * float(sim.state["rig"]["c.temperature"]))
    assert np.isclose(E, E0, rtol=1e-9)


def test_parallel_links_sum():
    """Two links between the same pair behave as one with the summed
    conductance (same decay constant)."""
    a1 = ThermalMass("a", heat_capacity=100.0)
    b1 = ThermalMass("b", heat_capacity=100.0)
    a1.connect(b1, conductance=10.0)
    a1.connect(b1, conductance=15.0)
    w1, _ = _rig(a1, b1)

    a2 = ThermalMass("a", heat_capacity=100.0)
    b2 = ThermalMass("b", heat_capacity=100.0)
    a2.connect(b2, conductance=25.0)
    w2, _ = _rig(a2, b2)

    s1, s2 = TargetNumpy(Sim(w1)), TargetNumpy(Sim(w2))
    for s in (s1, s2):
        s.state["rig"]["a.temperature"] = 320.0
        s.state["rig"]["b.temperature"] = 300.0
        for _ in range(500):
            s.step(0.001)
    assert np.isclose(float(s1.state["rig"]["a.temperature"]),
                      float(s2.state["rig"]["a.temperature"]), atol=1e-12)


# ---------------------------------------------------------------------------
# Heat input / generation
# ---------------------------------------------------------------------------

def test_heat_input_ramps_at_p_over_c():
    """An insulated node under constant power ramps at exactly P/C."""
    n = ThermalMass("n", heat_capacity=200.0)
    w, _ = _rig(n)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["n.heat_input"] = 50.0      # W
    T0 = float(sim.state["rig"]["n.temperature"])

    dt, steps = 0.01, 1000                        # 10 s
    for _ in range(steps):
        sim.step(dt)

    T = float(sim.state["rig"]["n.temperature"])
    assert np.isclose(T - T0, 50.0 / 200.0 * dt * steps, rtol=1e-9)


def test_motor_heats_its_winding_node():
    """ThermalMass(source=motor): a near-stalled motor dumps ≈ V²/R
    into the node — the electrical and thermal models share one graph."""
    V, R, C = 2.0, 1.0, 50.0
    c = Craft("rig")
    c.add(Mass("body", mass=50.0, moi=(10.0, 10.0, 10.0)))
    # Heavy rotor ⇒ mechanical time constant J·R/k² = 80 s ≫ the 1 s
    # window, so the shaft stays near stall and i ≈ V/R throughout.
    m = c.add(Motor("m", torque_constant=0.05, resistance=R))
    m.add(Mass("rotor", mass=0.5, moi=(0.01, 0.01, 0.2)))
    c.add(ThermalMass("winding", heat_capacity=C, source=m))
    w = World()
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["m.voltage"] = V
    T0 = float(sim.state["rig"]["winding.temperature"])

    dt, steps = 0.001, 1000                       # 1 s
    for _ in range(steps):
        sim.step(dt)

    T = float(sim.state["rig"]["winding.temperature"])
    # Stalled copper loss V²/R = 4 W → dT = 4/50 per s.
    assert np.isclose(T - T0, (V**2 / R) / C * dt * steps, rtol=0.02)


# ---------------------------------------------------------------------------
# Ambient boundary (plain input, no field)
# ---------------------------------------------------------------------------

def test_ambient_relaxation_time_constant():
    """A node leaking to a fixed ambient relaxes exponentially with
    τ = C/g."""
    n = ThermalMass("n", heat_capacity=100.0, ambient_conductance=20.0)
    w, _ = _rig(n)
    tau = 100.0 / 20.0                            # 5 s
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["n.temperature"] = 290.0
    sim.state["rig"]["n.ambient_temperature"] = 310.0

    dt = 0.001
    for _ in range(int(tau / dt)):
        sim.step(dt)

    T = float(sim.state["rig"]["n.temperature"])
    assert np.isclose(310.0 - T, 20.0 * np.exp(-1.0), rtol=0.01)


def test_ambient_is_scriptable_per_tick():
    """ambient_temperature is an Input: rewriting it mid-run redirects
    the node (script it against depth/altitude from the driving loop)."""
    n = ThermalMass("n", heat_capacity=10.0, ambient_conductance=50.0)
    w, _ = _rig(n)
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["n.temperature"] = 290.0

    sim.state["rig"]["n.ambient_temperature"] = 310.0
    for _ in range(5000):                         # ≫ τ = 0.2 s
        sim.step(0.001)
    assert np.isclose(float(sim.state["rig"]["n.temperature"]),
                      310.0, atol=0.05)

    sim.state["rig"]["n.ambient_temperature"] = 280.0
    for _ in range(5000):
        sim.step(0.001)
    assert np.isclose(float(sim.state["rig"]["n.temperature"]),
                      280.0, atol=0.05)


def test_insulated_node_has_no_ambient_input():
    """With ambient_conductance == 0 the ambient input isn't plumbed:
    no dead slot in the state dict / u port."""
    n = ThermalMass("n")                          # insulated (default)
    w, _ = _rig(n)
    sim = TargetNumpy(Sim(w))
    assert "n.ambient_temperature" not in sim.initial_state()["rig"]
    assert "n.heat_input" in sim.initial_state()["rig"]


# ---------------------------------------------------------------------------
# Ambient from the FluidField (ambient="fluid")
# ---------------------------------------------------------------------------

def _fluid(*disturbances):
    from manta.fields import FluidField
    f = FluidField()
    for d in disturbances:
        f.add(d)
    return f


def test_fluid_ambient_relaxation():
    """ambient='fluid': the node relaxes to the FluidField's declared
    temperature with τ = C/g."""
    from manta.fields import UniformFluid
    n = ThermalMass("n", heat_capacity=100.0, ambient_conductance=20.0,
                    ambient="fluid")
    w, _ = _rig(n)
    w.add_field(_fluid(UniformFluid(density=1025.0, temperature=310.0,
                                    viscosity=1.35e-3)))
    tau = 100.0 / 20.0
    sim = TargetNumpy(Sim(w))
    sim.state["rig"]["n.temperature"] = 290.0

    dt = 0.001
    for _ in range(int(tau / dt)):
        sim.step(dt)

    T = float(sim.state["rig"]["n.temperature"])
    assert np.isclose(310.0 - T, 20.0 * np.exp(-1.0), rtol=0.01)


def test_fluid_ambient_follows_regime_at_position():
    """Warm air above z=0, cold water below: a hull node equilibrates
    to whichever medium its craft sits in."""
    from manta.fields import UniformFluid, below_surface

    def build(z):
        from manta import Craft, World
        c = Craft("rig")
        c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
        c.add(ThermalMass("n", heat_capacity=10.0,
                          ambient_conductance=50.0, ambient="fluid"))
        w = World()
        air = UniformFluid(density=1.225, temperature=300.0)
        sea = UniformFluid(
            density=1025.0, temperature=278.0, viscosity=1.35e-3,
            membership=below_surface(lambda p, t: p._mx[2], 0.1))
        w.add_field(_fluid(air, sea))
        w.add_craft(c, position=(0.0, 0.0, z))
        return TargetNumpy(Sim(w))

    submerged, surfaced = build(-5.0), build(5.0)
    for s in (submerged, surfaced):
        s.state["rig"]["n.temperature"] = 290.0
        for _ in range(5000):                     # ≫ τ = 0.2 s
            s.step(0.001)
    assert np.isclose(float(submerged.state["rig"]["n.temperature"]),
                      278.0, atol=0.05)
    assert np.isclose(float(surfaced.state["rig"]["n.temperature"]),
                      300.0, atol=0.05)


def test_fluid_ambient_without_fluid_field_raises():
    n = ThermalMass("n", ambient_conductance=5.0, ambient="fluid")
    w, _ = _rig(n)
    with pytest.raises(ValueError, match="FluidField"):
        Sim(w)


def test_fluid_ambient_has_no_ambient_input():
    """ambient='fluid' reads the field — the ambient_temperature input
    must not be plumbed."""
    from manta.fields import UniformFluid
    n = ThermalMass("n", ambient_conductance=5.0, ambient="fluid")
    w, _ = _rig(n)
    w.add_field(_fluid(UniformFluid(density=1000.0, temperature=290.0,
                                    viscosity=1.0e-3)))
    sim = TargetNumpy(Sim(w))
    assert "n.ambient_temperature" not in sim.initial_state()["rig"]


def test_invalid_ambient_value_raises():
    with pytest.raises(ValueError, match="ambient"):
        ThermalMass("n", ambient="field")


# ---------------------------------------------------------------------------
# Validation + EKF plumbing
# ---------------------------------------------------------------------------

def test_cross_craft_link_raises():
    a = ThermalMass("a")
    b = ThermalMass("b")
    a.connect(b, conductance=5.0)
    c1 = Craft("one")
    c1.add(Mass("m", mass=1.0))
    c1.add(a)
    c2 = Craft("two")
    c2.add(Mass("m", mass=1.0))
    c2.add(b)
    w = World()
    w.add_craft(c1)
    w.add_craft(c2)
    with pytest.raises(ValueError, match="same craft"):
        Sim(w)


def test_connect_validation():
    a = ThermalMass("a")
    with pytest.raises(ValueError, match="itself"):
        a.connect(a, conductance=1.0)
    with pytest.raises(ValueError, match="conductance"):
        a.connect(ThermalMass("b"), conductance=0.0)
    with pytest.raises(TypeError, match="ThermalMass"):
        a.connect(Mass("m", mass=1.0), conductance=1.0)
    with pytest.raises(TypeError, match="dissipated_heat"):
        ThermalMass("n", source=Mass("m", mass=1.0))
    with pytest.raises(ValueError, match="heat_capacity"):
        ThermalMass("n", heat_capacity=0.0)


def test_temperature_is_ekf_state_with_heat_noise_q():
    """Temperature slots enter the EKF spec; heat_noise (σ > 0) shows up
    as a process-noise channel for auto-Q."""
    n = ThermalMass("n", heat_capacity=100.0, heat_noise_sigma=0.5)
    w, _ = _rig(n)
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    ekf = EKF(w)
    names = [s.name for s in ekf.spec.slots]
    assert "rig.n.temperature" in names
    assert "rig.n.heat_noise" in [s.full for s in ekf.sys.noise_specs]
