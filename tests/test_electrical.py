"""Contract and physics tests for the radial lumped-DC network."""

from __future__ import annotations

import math

import numpy as np
import pytest

from manta import EKF, UKF, Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import (
    ConstantCurrentLoad,
    ConstantPowerLoad,
    Contactor,
    DCConverter,
    DCSource,
    ElectricalBus,
    Fuse,
    Mass,
    ResistiveLoad,
    ThermalMass,
)


def _world(*parts):
    craft = Craft("rig")
    craft.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    for part in parts:
        craft.add(part)
    world = World(name="electrical").add_field(GravityField.none())
    world.add_craft(craft)
    return world


def _scalar(value) -> float:
    return float(np.asarray(value).item())


def _isolated_rail(load, *, voltage=12.0, capacitance=2.0):
    """An open contactor makes a capacitor + endpoint analytical circuit."""
    source = DCSource("source", open_circuit_voltage=voltage,
                      rail_voltage=voltage, capacitance=100.0,
                      source_resistance=0.01)
    rail = Contactor("rail", rail_voltage=voltage, capacitance=capacitance,
                     series_resistance=0.01, closed=0.0,
                     brownout_voltage=0.0, recovery_voltage=0.0)
    source.connect(rail).connect(load)
    return _world(source, rail, load)


# ---------------------------------------------------------------------------
# Topology + configuration boundary
# ---------------------------------------------------------------------------


def test_topology_is_independent_of_mechanical_tree():
    source = DCSource("source")
    bus = ElectricalBus("bus")
    load = ResistiveLoad("load")
    # All three are mechanical siblings; electrical connectivity is explicit.
    source.connect(bus).connect(load)
    TargetNumpy(Sim(_world(load, source, bus)))


def test_rejects_multiple_supplies_cycle_and_load_as_supply():
    first = DCSource("first")
    second = DCSource("second")
    bus = ElectricalBus("bus")
    load = ResistiveLoad("load")
    first.connect(bus)
    with pytest.raises(ValueError, match="already has upstream"):
        second.connect(bus)
    bus.connect(load)
    with pytest.raises(ValueError, match="endpoint load"):
        load.connect(ElectricalBus("other"))

    a, b = ElectricalBus("a"), ElectricalBus("b")
    a.connect(b)
    with pytest.raises(ValueError, match="cycle"):
        b.connect(a)


def test_rejects_disconnected_and_cross_craft_nodes_at_compile():
    with pytest.raises(ValueError, match="needs exactly one upstream"):
        Sim(_world(ElectricalBus("orphan")))

    one, two = Craft("one"), Craft("two")
    one.add(Mass("m1", mass=1.0, moi=(1.0, 1.0, 1.0)))
    two.add(Mass("m2", mass=1.0, moi=(1.0, 1.0, 1.0)))
    source = one.add(DCSource("source"))
    load = two.add(ResistiveLoad("load"))
    source.connect(load)
    world = World().add_field(GravityField.none())
    world.add_craft(one)
    world.add_craft(two)
    with pytest.raises(ValueError, match="crosses crafts"):
        Sim(world)


@pytest.mark.parametrize("factory, message", [
    (lambda: DCSource("s", capacitance="1 F"), "real numeric data"),
    (lambda: ElectricalBus("b", capacitance=0.0), "capacitance must be > 0"),
    (lambda: DCConverter("r", efficiency=1.1), "efficiency must be"),
    (lambda: ConstantPowerLoad("l", current_limit=0.0),
     "current_limit must be > 0"),
])
def test_rejects_invalid_units_and_nonphysical_parameters(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


# ---------------------------------------------------------------------------
# Analytical capacitor/load cases
# ---------------------------------------------------------------------------


def test_source_rc_charge_matches_closed_form():
    resistance, capacitance, final_voltage = 2.0, 4.0, 20.0
    source = DCSource(
        "source", open_circuit_voltage=final_voltage, rail_voltage=0.0,
        source_resistance=resistance, capacitance=capacitance)
    sim = TargetNumpy(Sim(_world(source)))
    tau = resistance * capacitance
    dt = 0.001
    for _ in range(int(tau / dt)):
        sim.step(dt)
    actual = float(sim.state["rig"]["source.rail_voltage"])
    expected = final_voltage * (1.0 - math.exp(-1.0))
    assert actual == pytest.approx(expected, rel=2e-4)


def test_resistive_discharge_matches_closed_form():
    resistance, capacitance, initial = 8.0, 2.0, 12.0
    load = ResistiveLoad("load", resistance=resistance)
    sim = TargetNumpy(Sim(_isolated_rail(
        load, voltage=initial, capacitance=capacitance)))
    tau = resistance * capacitance
    dt = 0.002
    for _ in range(int(tau / dt)):
        sim.step(dt)
    actual = float(sim.state["rig"]["rail.rail_voltage"])
    assert actual == pytest.approx(initial * math.exp(-1.0), rel=2e-4)


def test_constant_current_discharge_is_linear():
    capacitance, current, initial = 2.0, 1.5, 12.0
    load = ConstantCurrentLoad("load", current=current)
    sim = TargetNumpy(Sim(_isolated_rail(
        load, voltage=initial, capacitance=capacitance)))
    duration, dt = 4.0, 0.001
    for _ in range(int(duration / dt)):
        sim.step(dt)
    expected = initial - current * duration / capacitance
    assert float(sim.state["rig"]["rail.rail_voltage"]) == pytest.approx(
        expected, abs=2e-9)


def test_constant_power_discharge_reduces_capacitor_energy_linearly():
    capacitance, power, initial = 4.0, 20.0, 24.0
    load = ConstantPowerLoad(
        "load", power=power, voltage_floor=0.1,
        brownout_voltage=1.0, recovery_voltage=2.0)
    sim = TargetNumpy(Sim(_isolated_rail(
        load, voltage=initial, capacitance=capacitance)))
    duration, dt = 2.0, 0.0002
    energy0 = 0.5 * capacitance * initial**2
    for _ in range(int(duration / dt)):
        sim.step(dt)
    voltage = float(sim.state["rig"]["rail.rail_voltage"])
    energy = 0.5 * capacitance * voltage**2
    assert energy == pytest.approx(energy0 - power * duration, rel=2e-5)


# ---------------------------------------------------------------------------
# Limits, events and conservation diagnostics
# ---------------------------------------------------------------------------


def test_brownout_prevents_constant_current_from_driving_voltage_negative():
    load = ConstantCurrentLoad(
        "load", current=5.0, brownout_voltage=4.0, recovery_voltage=5.0)
    sim = TargetNumpy(Sim(_isolated_rail(load, voltage=6.0,
                                         capacitance=0.5)))
    dt = 0.0005
    for _ in range(20_000):
        sim.step(dt)
    voltage = float(sim.state["rig"]["rail.rail_voltage"])
    assert 3.99 <= voltage <= 4.01
    assert _scalar(sim.outputs()["rig"]["load.brownout"]) > 0.999


def test_converter_respects_current_and_power_limits():
    source = DCSource("source", open_circuit_voltage=48.0,
                      rail_voltage=48.0, capacitance=10.0,
                      source_resistance=0.01, current_limit=50.0)
    regulator = DCConverter(
        "regulator", output_voltage=12.0, rail_voltage=1.0,
        capacitance=1.0, control_resistance=0.001,
        output_current_limit=3.0, output_power_limit=24.0,
        input_power_limit=30.0, efficiency=0.8,
        brownout_voltage=0.5, recovery_voltage=0.8)
    load = ConstantPowerLoad(
        "load", power=10.0, voltage_floor=0.1,
        brownout_voltage=0.2, recovery_voltage=0.5)
    source.connect(regulator).connect(load)
    sim = TargetNumpy(Sim(_world(source, regulator, load)))
    sim.step(0.001)
    out = sim.outputs()["rig"]
    # Rail injection is bounded by min(3 A, 24 W / 1 V,
    # 30 W * 0.8 / 1 V).
    next_voltage = float(sim.state["rig"]["regulator.rail_voltage"])
    delivered_current = (next_voltage - 1.0) / 0.001 + 10.0
    assert 0.0 <= delivered_current <= 3.0 + 1e-9
    assert _scalar(out["regulator.input_power"]) <= 30.0 + 1e-9
    assert _scalar(out["regulator.loss_power"]) >= 0.0


def test_contactor_opens_explicitly_and_downstream_capacitor_drains():
    load = ResistiveLoad("load", resistance=4.0)
    sim = TargetNumpy(Sim(_isolated_rail(load, voltage=12.0,
                                         capacitance=1.0)))
    initial = float(sim.state["rig"]["rail.rail_voltage"])
    for _ in range(100):
        sim.step(0.001)
    assert _scalar(sim.outputs()["rig"]["rail.open"]) == pytest.approx(1.0)
    assert float(sim.state["rig"]["rail.rail_voltage"]) < initial


def test_fuse_trips_latches_and_stops_input_current():
    source = DCSource("source", open_circuit_voltage=12.0,
                      rail_voltage=12.0, capacitance=100.0,
                      source_resistance=0.001, current_limit=100.0)
    fuse = Fuse("fuse", rail_voltage=12.0, capacitance=0.2,
                series_resistance=0.01, input_current_limit=100.0,
                rated_current=1.0, trip_time=0.002)
    load = ResistiveLoad("load", resistance=0.5)
    source.connect(fuse).connect(load)
    sim = TargetNumpy(Sim(_world(source, fuse, load)))
    for _ in range(1000):
        sim.step(0.0001)
        if float(sim.state["rig"]["fuse.trip_fraction"]) >= 1.0:
            break
    assert float(sim.state["rig"]["fuse.trip_fraction"]) == pytest.approx(1.0)
    sim.step(0.0001)
    out = sim.outputs()["rig"]
    assert _scalar(out["fuse.tripped"]) == pytest.approx(1.0)
    assert _scalar(out["fuse.open"]) == pytest.approx(1.0)
    assert _scalar(out["fuse.input_current"]) == pytest.approx(0.0)


def test_kcl_and_energy_residuals_remain_bounded_over_transient():
    source = DCSource("source", open_circuit_voltage=50.4,
                      rail_voltage=45.0, capacitance=5.0,
                      source_resistance=0.08, current_limit=30.0,
                      brownout_voltage=34.0, recovery_voltage=38.0)
    bus = ElectricalBus("bus", rail_voltage=40.0, capacitance=0.5,
                        series_resistance=0.05, input_current_limit=25.0)
    regulator = DCConverter(
        "regulator", output_voltage=12.0, rail_voltage=8.0,
        capacitance=0.2, output_current_limit=8.0,
        output_power_limit=80.0, input_power_limit=100.0,
        brownout_voltage=9.0, recovery_voltage=10.0)
    load = ConstantPowerLoad(
        "load", power=50.0, current_limit=8.0, voltage_floor=1.0,
        brownout_voltage=7.0, recovery_voltage=9.0)
    source.connect(bus).connect(regulator).connect(load)
    sim = TargetNumpy(Sim(_world(source, bus, regulator, load)))
    residuals = []
    for _ in range(2000):
        sim.step(0.0002)
        out = sim.outputs()["rig"]
        for name in ("source", "bus", "regulator"):
            residuals.extend((_scalar(out[f"{name}.kcl_residual"]),
                              _scalar(out[f"{name}.energy_residual"])))
    assert np.all(np.isfinite(residuals))
    assert max(abs(value) for value in residuals) < 1e-10
    for key, value in sim.state["rig"].items():
        if key.endswith("rail_voltage"):
            assert np.isfinite(float(value)) and float(value) >= 0.0


def test_electrical_loss_couples_into_existing_thermal_network():
    source = DCSource(
        "source", open_circuit_voltage=12.0, rail_voltage=10.0,
        source_resistance=1.0, capacitance=10.0)
    thermal = ThermalMass("source_heat", heat_capacity=100.0, source=source)
    sim = TargetNumpy(Sim(_world(source, thermal)))
    initial = float(sim.state["rig"]["source_heat.temperature"])
    sim.step(0.01)
    # Initial source current is 2 A and series loss is (12 - 10) * 2 = 4 W.
    expected = initial + 0.01 * 4.0 / 100.0
    assert float(sim.state["rig"]["source_heat.temperature"]) == pytest.approx(
        expected, abs=1e-12)


def test_stability_hint_is_explicit_and_supported_step_stays_positive():
    rail = ElectricalBus("rail", capacitance=0.5)
    hint = rail.stable_timestep_hint(
        minimum_voltage=4.0, maximum_net_current=20.0)
    assert hint == pytest.approx(0.1)


def test_ekf_and_ukf_construct_over_electrical_states():
    source = DCSource("source")
    bus = ElectricalBus("bus")
    source.connect(bus)
    world = _world(source, bus)
    ekf = EKF(world, sensors=[])
    ukf = UKF(world, sensors=[])
    names = {slot.name for slot in ekf.spec.slots}
    assert "rig.source.rail_voltage" in names
    assert "rig.bus.rail_voltage" in names
    assert ukf.spec.ambient_dim == ekf.spec.ambient_dim
