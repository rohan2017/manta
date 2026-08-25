"""Conservation, faults, replay, and boundary tests for the battery plant."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import GravityField
from manta.parts import ConstantCurrentLoad, ExternalDCSupply, Mass, ThermalMass
from manta.simulation import (
    BatteryCell,
    BatteryCellFaults,
    BatteryElectricalModel,
    BatteryStepInput,
    BMSPlant,
    OCVCurve,
    PassiveBalancer,
    SeriesBatteryPack,
)


def _pack(*, count=2, cell=None, initial_soc=1.0, balancers=(), seed=0):
    cell = cell or BatteryCell()
    return SeriesBatteryPack(
        [cell] * count, initial_soc=initial_soc,
        balancers=balancers, seed=seed)


def _input(pack, current=0.0, *, faults=None, temperatures=None,
           balance=None, contactor=True, trip=False, reset=False):
    count = len(pack.cells)
    return BatteryStepInput(
        requested_series_current=current,
        cell_temperatures=tuple(temperatures or [298.15] * count),
        cell_faults=tuple(faults or [BatteryCellFaults()] * count),
        balance_enabled=tuple(balance or [False] * len(pack.balancers)),
        contactor_command=contactor,
        trip_command=trip,
        reset_command=reset,
    )


def _coupled_runtime(*, thermal=False):
    source = ExternalDCSupply("pack_supply")
    load = ConstantCurrentLoad(
        "hotel", current=1.0, brownout_voltage=0.1, recovery_voltage=0.2)
    source.connect(load)
    craft = Craft("rig")
    craft.add(Mass("body", mass=1.0))
    craft.add(source)
    craft.add(load)
    thermal_parts = tuple(f"cell_{index}" for index in range(4)) if thermal else ()
    for name in thermal_parts:
        craft.add(ThermalMass(name, heat_capacity=1.0))
    world = World().add_field(GravityField(g=(0.0, 0.0, 0.0)))
    world.add_craft(craft)
    runtime = TargetNumpy(Sim(world))
    pack = _pack(count=4, initial_soc=0.8)
    battery = runtime.attach_model(BatteryElectricalModel(
        pack, craft="rig", supply="pack_supply",
        thermal_parts=thermal_parts))
    return runtime, battery


def test_battery_and_electrical_graph_share_one_atomic_sim_tick():
    runtime, battery = _coupled_runtime()
    before_soc = battery.pack.state.cells[0].soc
    runtime.step(1.0)
    assert runtime.time == pytest.approx(1.0)
    assert battery.pack.state.cells[0].soc < before_soc
    assert float(np.asarray(
        runtime.outputs()["rig"]["pack_supply.output_current"]).item()
    ) == pytest.approx(1.0)

    checkpoint = runtime.checkpoint()
    checkpoint_soc = battery.pack.state.cells[0].soc
    runtime.step(1.0)
    runtime.restore(checkpoint)
    assert runtime.time == pytest.approx(1.0)
    assert battery.pack.state.cells[0].soc == checkpoint_soc


def test_battery_cells_and_thermal_graph_share_the_sim_update():
    runtime, battery = _coupled_runtime(thermal=True)
    initial = runtime.state["rig"]["cell_0.temperature"]
    runtime.step(1.0)  # establishes current/loss telemetry
    runtime.step(1.0)  # prior loss heats the cell's ThermalMass via ZOH
    temperature = runtime.state["rig"]["cell_0.temperature"]
    assert temperature > initial
    assert battery.cell_temperatures[0] == pytest.approx(temperature)


def test_coupled_battery_failure_rolls_back_the_spatial_tick():
    runtime, battery = _coupled_runtime()
    runtime.state["rig"]["velocity"] = np.array([1.0, 0.0, 0.0])
    before = runtime.checkpoint()
    with pytest.raises(ValueError, match="SOC would leave"):
        runtime.step(60_000.0)
    assert runtime.time == before.time
    assert runtime.checkpoint().values == before.values
    assert battery.pack.state.cells[0].soc == pytest.approx(0.8)


def test_coupled_battery_exclusively_owns_its_supply_input():
    runtime, _ = _coupled_runtime()
    with pytest.raises(ValueError, match="two owners"):
        runtime.step(0.1, u={"pack_supply.supplied_voltage": 12.0})


def test_ocv_curve_is_calibratable_and_integrates_exactly():
    curve = OCVCurve((0.0, 0.5, 1.0), (3.0, 4.0, 5.0))
    assert curve.voltage(0.25) == pytest.approx(3.5)
    assert curve.voltage(1.1) == pytest.approx(5.2)  # explicit extrapolation
    assert curve.integral(1.0) == pytest.approx(4.0)


def test_constant_ocv_charge_and_energy_conservation():
    voltage, capacity, current = 3.7, 2.0, 2.0
    cell = BatteryCell(
        usable_capacity_ah=capacity, internal_resistance=0.01,
        ocv_curve=OCVCurve((0.0, 1.0), (voltage, voltage)))
    pack = _pack(count=2, cell=cell)
    faults = (BatteryCellFaults(),) * 2
    initial = sum(cell.stored_energy(state, fault)
                  for state, fault in zip(pack.state.cells, faults))
    chemical = 0.0
    for _ in range(1000):
        telemetry = pack.step(0.01, _input(pack, current, faults=faults))
        chemical += telemetry.chemical_power * 0.01
        assert abs(telemetry.energy_residual) < 1e-10
    final = sum(cell.stored_energy(state, fault)
                for state, fault in zip(pack.state.cells, faults))
    assert initial - final == pytest.approx(chemical, rel=1e-11, abs=2e-9)


def test_series_current_invariant_and_expected_terminal_sag():
    cell = BatteryCell(
        internal_resistance=0.01,
        ocv_curve=OCVCurve((0.0, 1.0), (4.2, 4.2)))
    pack = _pack(count=4, cell=cell)
    telemetry = pack.step(0.001, _input(pack, 10.0))
    assert telemetry.series_current == 10.0
    assert telemetry.cell_current == pytest.approx((10.0,) * 4)
    assert telemetry.terminal_voltage == pytest.approx(16.8 - 10.0 * 0.04)
    assert telemetry.output_power + telemetry.pack_loss_power == pytest.approx(
        telemetry.chemical_power)


def test_passive_balancing_changes_only_selected_cell():
    cell = BatteryCell(
        usable_capacity_ah=0.1, internal_resistance=0.01,
        ocv_curve=OCVCurve((0.0, 1.0), (4.0, 4.0)))
    pack = _pack(
        count=2, cell=cell, initial_soc=(0.9, 0.8),
        balancers=(PassiveBalancer(0, resistance=4.0),))
    for _ in range(1000):
        telemetry = pack.step(0.01, _input(pack, balance=(True,)))
    assert pack.state.cells[0].soc < 0.9
    assert pack.state.cells[1].soc == pytest.approx(0.8)
    assert telemetry.balance_current[0] > 0.0
    assert telemetry.balance_loss_power[0] > 0.0
    assert telemetry.soc_imbalance < 0.1
    assert abs(telemetry.energy_residual) < 1e-10


def test_open_cell_blocks_series_current_and_opens_terminal():
    pack = _pack()
    faults = (BatteryCellFaults(open_cell=True), BatteryCellFaults())
    telemetry = pack.step(0.1, _input(pack, 10.0, faults=faults))
    assert telemetry.any_open_cell
    assert telemetry.requested_series_current == 10.0
    assert telemetry.series_current == 0.0
    assert telemetry.terminal_voltage == 0.0
    assert pack.state.cells[0].soc == 1.0


def test_short_cell_removes_terminal_ocv_and_discharges_into_heat():
    cell = BatteryCell(
        usable_capacity_ah=10.0, internal_resistance=0.01,
        short_resistance=0.01, maximum_short_current=50.0,
        ocv_curve=OCVCurve((0.0, 1.0), (4.0, 4.0)))
    pack = _pack(count=2, cell=cell)
    faults = (BatteryCellFaults(short_cell=True), BatteryCellFaults())
    telemetry = pack.step(0.01, _input(pack, 2.0, faults=faults))
    assert telemetry.any_shorted_cell
    assert telemetry.pack_ocv == pytest.approx(4.0)
    assert telemetry.cell_terminal_voltage[0] == pytest.approx(-0.02)
    assert telemetry.cell_loss_power[0] > telemetry.cell_loss_power[1]
    assert pack.state.cells[0].soc < pack.state.cells[1].soc
    assert abs(telemetry.energy_residual) < 1e-10


def test_high_resistance_fault_increases_sag_and_heat():
    cell = BatteryCell(
        internal_resistance=0.01, high_resistance_multiplier=10.0,
        ocv_curve=OCVCurve((0.0, 1.0), (4.0, 4.0)))
    healthy = _pack(cell=cell).preview(_input(_pack(cell=cell), 5.0))
    fault_pack = _pack(cell=cell)
    faults = (BatteryCellFaults(high_resistance=True), BatteryCellFaults())
    failed = fault_pack.preview(_input(fault_pack, 5.0, faults=faults))
    assert failed.terminal_voltage < healthy.terminal_voltage
    assert failed.pack_loss_power > healthy.pack_loss_power


def test_capacity_loss_accelerates_soc_depletion():
    cell = BatteryCell(
        usable_capacity_ah=1.0,
        ocv_curve=OCVCurve((0.0, 1.0), (4.0, 4.0)))
    pack = _pack(cell=cell)
    faults = (BatteryCellFaults(capacity_loss_fraction=0.5),
              BatteryCellFaults())
    for _ in range(1000):
        pack.step(0.01, _input(pack, 1.0, faults=faults))
    loss0 = 1.0 - pack.state.cells[0].soc
    loss1 = 1.0 - pack.state.cells[1].soc
    assert loss0 == pytest.approx(2.0 * loss1)


def test_overtemperature_changes_resistance_without_automatic_trip_policy():
    cell = BatteryCell(
        internal_resistance=0.01,
        overtemperature_resistance_multiplier=3.0)
    pack = _pack(cell=cell)
    normal = pack.preview(_input(pack, 5.0))
    hot = pack.step(0.01, _input(
        pack, 5.0, temperatures=(350.0, 298.15)))
    assert hot.any_overtemperature
    assert hot.terminal_voltage < normal.terminal_voltage
    assert not hot.tripped
    assert hot.contactor_closed


def test_bms_trip_latches_contactor_until_explicit_reset():
    pack = _pack()
    tripped = pack.step(0.01, _input(pack, 5.0, trip=True))
    assert tripped.tripped and not tripped.contactor_closed
    held = pack.step(0.01, _input(pack, 5.0))
    assert held.tripped and held.series_current == 0.0
    reset = pack.step(0.01, _input(pack, 5.0, reset=True))
    assert not reset.tripped and reset.contactor_closed
    assert reset.series_current == 5.0


def test_bms_transition_rejects_non_boolean_commands():
    with pytest.raises(TypeError, match="trip_command"):
        BMSPlant.transition(pack_state := _pack().state.bms,
                            contactor_command=True,
                            trip_command=1, reset_command=False)
    assert not pack_state.trip_latched


def test_thermal_input_boundary_uses_existing_thermal_mass_inputs():
    pack = _pack(balancers=(PassiveBalancer(0, resistance=10.0),))
    telemetry = pack.step(0.01, _input(pack, 5.0, balance=(True,)))
    mapped = telemetry.thermal_inputs(("cell0_heat", "cell1_heat"))
    assert mapped == {
        "cell0_heat.heat_input": (telemetry.cell_loss_power[0]
                                  + telemetry.balance_loss_power[0]),
        "cell1_heat.heat_input": (telemetry.cell_loss_power[1]
                                  + telemetry.balance_loss_power[1]),
    }
    with pytest.raises(ValueError, match="one name per cell"):
        telemetry.thermal_inputs(("only_one",))


def test_external_supply_boundary_uses_a3_input_name():
    pack = _pack()
    telemetry = pack.preview(_input(pack))
    assert telemetry.supply_inputs("pack_supply") == {
        "pack_supply.supplied_voltage": telemetry.terminal_voltage,
    }


def test_seeded_self_discharge_and_checkpoint_restore_replay_exactly():
    cell = BatteryCell(
        usable_capacity_ah=1.0, self_discharge_current=0.02,
        self_discharge_log_sigma=0.5)
    pack = _pack(cell=cell, seed=17)
    command = _input(pack)
    for _ in range(20):
        pack.step(1.0, command)
    checkpoint = pack.checkpoint()
    first = [pack.step(1.0, command).cell_soc for _ in range(30)]
    pack.restore(copy.deepcopy(checkpoint))
    second = [pack.step(1.0, command).cell_soc for _ in range(30)]
    np.testing.assert_array_equal(first, second)

    other = _pack(cell=cell, seed=18)
    other_trace = [other.step(1.0, _input(other)).cell_soc for _ in range(50)]
    pack.reset(seed=17)
    original_trace = [pack.step(1.0, _input(pack)).cell_soc for _ in range(50)]
    assert original_trace != other_trace


def test_preview_does_not_advance_state_or_random_stream():
    cell = BatteryCell(self_discharge_current=0.01,
                       self_discharge_log_sigma=0.5)
    pack = _pack(cell=cell, seed=4)
    before = pack.checkpoint()
    pack.preview(_input(pack))
    after = pack.checkpoint()
    assert before["soc"] == after["soc"]
    assert before["rng_state"] == after["rng_state"]


def test_invalid_restore_is_atomic():
    pack = _pack(seed=6)
    pack.step(0.1, _input(pack, 2.0))
    before = pack.checkpoint()
    corrupt = copy.deepcopy(before)
    corrupt["soc"] = [0.5, 0.5]
    corrupt["rng_state"] = {"invalid": True}
    with pytest.raises(ValueError, match="RNG state"):
        pack.restore(corrupt)
    after = pack.checkpoint()
    assert before["soc"] == after["soc"]
    assert before["rng_state"] == after["rng_state"]


def test_invalid_energy_transition_fails_instead_of_clamping():
    cell = BatteryCell(
        usable_capacity_ah=0.001,
        ocv_curve=OCVCurve((0.0, 1.0), (4.0, 4.0)))
    pack = _pack(cell=cell, initial_soc=0.01)
    with pytest.raises(ValueError, match="SOC would leave"):
        pack.step(1.0, _input(pack, 1.0))
    assert pack.state.cells[0].soc == pytest.approx(0.01)


def test_invalid_step_is_atomic_including_random_stream():
    cell = BatteryCell(
        usable_capacity_ah=0.001, self_discharge_current=0.1,
        self_discharge_log_sigma=0.5)
    pack = _pack(cell=cell, initial_soc=0.01, seed=9)
    before = pack.checkpoint()
    with pytest.raises(ValueError, match="SOC would leave"):
        pack.step(10.0, _input(pack, 1.0))
    after = pack.checkpoint()
    assert before["soc"] == after["soc"]
    assert before["rng_state"] == after["rng_state"]


@pytest.mark.parametrize("factory, message", [
    (lambda: BatteryCell(usable_capacity_ah=0.0), "capacity"),
    (lambda: OCVCurve((0.0, 0.5), (3.0, 3.5)), "span"),
    (lambda: BatteryCellFaults(capacity_loss_fraction=1.0), r"\[0, 1\)"),
    (lambda: SeriesBatteryPack([], initial_soc=1.0), "at least one"),
    (lambda: SeriesBatteryPack([BatteryCell()], initial_soc=(0.5, 0.5)),
     "one value per cell"),
    (lambda: SeriesBatteryPack(
        [BatteryCell()], balancers=(PassiveBalancer(1),)), "outside"),
])
def test_invalid_configuration_fails_locally(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
