"""Powered actuator endpoints over the A1 radial electrical network."""

from __future__ import annotations

import math

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    ConstantPowerElectronicsLoad,
    DCConverter,
    DCSource,
    ExternalDCSupply,
    Mass,
    Motor,
    PoweredControlSurface,
    PoweredDuctedPropeller,
    PoweredMotor,
    PoweredThruster,
    ThermalMass,
    Thruster,
)


def _source(name="source", *, voltage=12.0, capacitance=20.0):
    return DCSource(
        name,
        open_circuit_voltage=voltage,
        rail_voltage=voltage,
        source_resistance=0.01,
        capacitance=capacitance,
        current_limit=100.0,
    )


def _world(*parts, fluid=False):
    craft = Craft("rig")
    craft.add(Mass("body", mass=20.0, moi=(10.0, 10.0, 10.0)))
    for part in parts:
        craft.add(part)
    world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    if fluid:
        world.add_field(FluidField().add_uniform(density=1025.0))
    world.add_craft(craft)
    return world


def _scalar(outputs, name):
    return float(np.asarray(outputs[name]).item())


def test_powered_motor_voltage_torque_current_and_energy_balance():
    source = _source()
    motor = PoweredMotor(
        "motor", torque_constant=0.1, resistance=1.0,
        controller_efficiency=0.9, brownout_voltage=5.0,
        recovery_voltage=6.0,
    )
    motor.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.01)))
    source.connect(motor)
    sim = TargetNumpy(Sim(_world(source, motor)))
    sim.step(1e-4, u={"motor.voltage": 6.0})
    output = sim.outputs()["rig"]

    # At rest: 6 V / 1 ohm = 6 A in the armature. PWM conversion draws
    # 36 W / 0.9 from the 12 V rail, not a fictitious 6 A rail current.
    assert _scalar(output, "motor.armature_current") == pytest.approx(6.0)
    assert _scalar(output, "motor.input_current") == pytest.approx(
        36.0 / 0.9 / 12.0)
    assert float(sim.state["rig"]["motor.rate"]) > 0.0
    assert _scalar(output, "motor.energy_residual") == pytest.approx(
        0.0, abs=1e-12)
    assert _scalar(output, "motor.input_power") == pytest.approx(
        _scalar(output, "motor.mechanical_power")
        + _scalar(output, "motor.loss_power"), abs=1e-12)


def test_powered_motor_positive_mechanical_power_and_single_thermal_source():
    source = _source()
    motor = PoweredMotor(
        "motor", torque_constant=0.1, resistance=1.0,
        controller_efficiency=0.9, brownout_voltage=5.0,
        recovery_voltage=6.0)
    motor.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.01)))
    thermal = ThermalMass(
        "motor_thermal", source=motor, heat_capacity=100.0)
    source.connect(motor)
    sim = TargetNumpy(Sim(_world(source, motor, thermal)))
    sim.state["rig"]["motor.rate"] = 20.0
    sim.step(0.001, u={"motor.voltage": 6.0})
    output = sim.outputs()["rig"]
    useful = _scalar(output, "motor.mechanical_power")
    heat = _scalar(output, "motor.loss_power")
    assert useful > 0.0
    assert _scalar(output, "motor.input_power") == pytest.approx(
        useful + heat, abs=1e-12)
    initial_temperature = 293.15
    assert float(sim.state["rig"]["motor_thermal.temperature"]) == pytest.approx(
        initial_temperature + 0.001 * heat / 100.0, abs=1e-12)


def test_motor_torque_derates_with_supply_sag():
    def initial_rate(supply_voltage):
        source = _source(voltage=supply_voltage)
        motor = PoweredMotor(
            "motor", torque_constant=0.1, resistance=1.0,
            brownout_voltage=4.0, recovery_voltage=5.0)
        motor.add(Mass("rotor", mass=0.1, moi=(0.01, 0.01, 0.01)))
        source.connect(motor)
        sim = TargetNumpy(Sim(_world(source, motor)))
        sim.step(1e-4, u={"motor.voltage": 12.0})
        return float(sim.state["rig"]["motor.rate"])

    full = initial_rate(12.0)
    sagged = initial_rate(6.0)
    assert sagged == pytest.approx(0.5 * full, rel=2e-3)


def test_load_aggregation_includes_compute_and_actuators():
    source = _source()
    compute = ConstantPowerElectronicsLoad(
        "compute", power=24.0, brownout_voltage=5.0,
        recovery_voltage=6.0, voltage_floor=1.0)
    thruster = PoweredThruster(
        "thruster", force=(10.0, 0.0, 0.0), rated_voltage=12.0,
        rated_mechanical_power=40.0, conversion_efficiency=0.8,
        power_exponent=2.0, brownout_voltage=5.0,
        recovery_voltage=6.0)
    source.connect(compute)
    source.connect(thruster)
    sim = TargetNumpy(Sim(_world(source, compute, thruster)))
    sim.step(1e-4, u={"thruster.throttle": 0.5})
    output = sim.outputs()["rig"]
    expected = 24.0 / 12.0 + (40.0 * 0.5**2 / 0.8) / 12.0
    assert _scalar(output, "source.output_current") == pytest.approx(expected)
    assert _scalar(output, "compute.loss_power") == pytest.approx(24.0)
    assert _scalar(output, "compute.output_power") == pytest.approx(0.0)


def test_external_supply_boundary_keeps_battery_state_out_of_manta():
    supply = ExternalDCSupply("battery_boundary")
    thruster = PoweredThruster(
        "thruster", force=(12.0, 0.0, 0.0), rated_voltage=12.0,
        rated_mechanical_power=48.0, conversion_efficiency=0.8,
        brownout_voltage=2.0, recovery_voltage=3.0)
    supply.connect(thruster)
    transform = Sim(_world(supply, thruster))
    assert not any("battery_boundary" in slot.name
                   for slot in transform.sys.spec.slots)
    sim = TargetNumpy(transform)
    sim.step(0.001, u={
        "battery_boundary.supplied_voltage": 6.0,
        "thruster.throttle": 1.0,
    })
    output = sim.outputs()["rig"]
    assert _scalar(output, "battery_boundary.output_current") == pytest.approx(
        _scalar(output, "thruster.input_current"))
    assert _scalar(output, "thruster.drive_fraction") == pytest.approx(0.5)


def test_thruster_force_and_power_derate_in_brownout():
    def run(voltage):
        source = _source(voltage=voltage)
        thruster = PoweredThruster(
            "thruster", force=(12.0, 0.0, 0.0), rated_voltage=12.0,
            rated_mechanical_power=60.0, conversion_efficiency=0.75,
            power_exponent=2.0, brownout_voltage=4.0,
            recovery_voltage=8.0)
        source.connect(thruster)
        sim = TargetNumpy(Sim(_world(source, thruster)))
        sim.step(0.01, u={"thruster.throttle": 1.0})
        return sim

    full = run(12.0)
    sagged = run(6.0)
    full_vx = float(np.asarray(full.state["rig"]["velocity"])[0])
    sagged_vx = float(np.asarray(sagged.state["rig"]["velocity"])[0])
    # 6 V gives a 0.5 voltage ratio and a 0.5 smooth brownout gate.
    assert sagged_vx == pytest.approx(0.25 * full_vx, rel=1e-3)
    out = sagged.outputs()["rig"]
    assert _scalar(out, "thruster.drive_fraction") == pytest.approx(0.25)
    assert _scalar(out, "thruster.energy_residual") == pytest.approx(
        0.0, abs=1e-12)


def test_converter_overload_sags_rail_and_derates_thruster():
    source = _source(voltage=24.0, capacitance=100.0)
    converter = DCConverter(
        "reg", output_voltage=12.0, rail_voltage=12.0,
        capacitance=0.01, control_resistance=0.01,
        efficiency=0.9, output_current_limit=1.0,
        output_power_limit=12.0, input_power_limit=20.0,
        brownout_voltage=3.0, recovery_voltage=4.0)
    thruster = PoweredThruster(
        "thruster", force=(100.0, 0.0, 0.0), rated_voltage=12.0,
        rated_mechanical_power=120.0, conversion_efficiency=0.8,
        brownout_voltage=4.0, recovery_voltage=6.0)
    source.connect(converter).connect(thruster)
    sim = TargetNumpy(Sim(_world(source, converter, thruster)))
    for _ in range(1500):
        sim.step(1e-4, u={"thruster.throttle": 1.0})
    output = sim.outputs()["rig"]
    assert float(sim.state["rig"]["reg.rail_voltage"]) < 8.0
    assert _scalar(output, "thruster.drive_fraction") < 0.7
    assert _scalar(output, "reg.input_power") <= 20.0 + 1e-9


def test_powered_ducted_propeller_voltage_derates_static_point():
    source = _source(voltage=6.0)
    propeller = PoweredDuctedPropeller(
        "prop", max_static_thrust=100.0, max_static_torque=10.0,
        diameter=0.2, zero_thrust_advance_speed=2.0,
        rated_voltage=12.0, rated_mechanical_power=200.0,
        conversion_efficiency=0.8,
        brownout_voltage=1.0, recovery_voltage=2.0)
    source.connect(propeller)
    sim = TargetNumpy(Sim(_world(source, propeller, fluid=True)))
    sim.step(1e-4, u={"prop.thrust_command": 100.0})
    output = sim.outputs()["rig"]
    assert _scalar(output, "prop.thrust") == pytest.approx(50.0, rel=1e-5)
    assert _scalar(output, "prop.reaction_torque") == pytest.approx(
        5.0, rel=1e-5)


def test_powered_servo_tracks_more_slowly_under_voltage_sag():
    def deflection(voltage):
        source = _source(voltage=voltage)
        surface = PoweredControlSurface(
            "fin", area=0.2, chord=0.2, servo_gain=10.0,
            stall_torque=2.0, hinge_damping=0.4,
            rated_voltage=12.0, rated_mechanical_power=10.0,
            conversion_efficiency=0.8,
            brownout_voltage=1.0, recovery_voltage=2.0)
        source.connect(surface)
        sim = TargetNumpy(Sim(_world(source, surface, fluid=True)))
        for _ in range(20):
            sim.step(0.001, u={"fin.deflection_cmd": math.radians(20.0)})
        return float(sim.state["rig"]["fin.deflection"])

    assert deflection(6.0) < 0.7 * deflection(12.0)


def test_unpowered_direct_voltage_and_throttle_apis_are_unchanged():
    assert set(Motor("m").input_declarations()) == {"voltage"}
    assert set(Thruster("t").input_declarations()) == {"throttle"}

    direct = Motor("direct", torque_constant=0.1, resistance=2.0)
    assert direct.declared_value("voltage") == 0.0
    direct_thruster = Thruster("direct_thruster", force=(1.0, 0.0, 0.0))
    assert direct_thruster.declared_value("throttle") == 0.0


def test_powered_endpoint_rejects_missing_or_multiple_supply():
    with pytest.raises(TypeError, match="requires explicit calibration"):
        PoweredThruster("uncalibrated")
    with pytest.raises(ValueError, match="voltage_floor"):
        PoweredThruster(
            "bad_floor", rated_voltage=12.0, rated_mechanical_power=1.0,
            conversion_efficiency=0.8, brownout_voltage=1.0,
            recovery_voltage=2.0, voltage_floor=1.5)

    with pytest.raises(ValueError, match="needs exactly one upstream"):
        Sim(_world(PoweredThruster(
            "thruster", rated_voltage=12.0, rated_mechanical_power=1.0,
            conversion_efficiency=0.8)))

    first, second = _source("first"), _source("second")
    thruster = PoweredThruster(
        "thruster", rated_voltage=12.0, rated_mechanical_power=1.0,
        conversion_efficiency=0.8)
    first.connect(thruster)
    with pytest.raises(ValueError, match="already has upstream"):
        second.connect(thruster)


def test_powered_load_numpy_jax_generated_kernel_parity():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from manta.codegen.jax import TargetJax as JaxModule

    source = _source()
    thruster = PoweredThruster(
        "thruster", force=(10.0, 0.0, 0.0), rated_voltage=12.0,
        rated_mechanical_power=30.0, conversion_efficiency=0.8)
    source.connect(thruster)
    transform = Sim(_world(source, thruster))
    module = transform.module()
    generated = JaxModule(module)
    x = np.asarray(generated.initial_state())
    u = np.zeros(module.port("u").size)
    fields = [field.name for field in module.port("u").fields]
    u[fields.index("rig.thruster.throttle")] = 0.6
    u[fields.index("rig.thruster.enabled")] = 1.0
    u[fields.index("rig.source.enabled")] = 1.0
    noise = np.zeros(module.port("noise").size)
    reference = module.functions["step"](x, u, noise, 0.001, 0.0)
    result = generated.kernel("step")(x, u, noise, 0.001, 0.0)
    for expected, actual in zip(reference, result):
        np.testing.assert_allclose(
            np.asarray(actual).ravel(), np.asarray(expected).ravel(),
            atol=1e-12)

    numpy_runtime = TargetNumpy(transform)
    numpy_runtime.step(0.001, u={"thruster.throttle": 0.6})
    from manta.ir.state_spec import flatten_nested
    numpy_state = module.spec.pack_any(flatten_nested(numpy_runtime.state))
    np.testing.assert_allclose(
        np.asarray(result[0]).ravel(), numpy_state, atol=1e-12)
