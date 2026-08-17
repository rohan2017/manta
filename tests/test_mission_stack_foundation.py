"""Cross-feature contracts for the mission-stack Manta foundation.

These tests deliberately compose independently developed model features.  A
failure here should identify an integration seam before Shiver bakes the
aggregate commit into a vehicle artifact.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from manta import (
    EKF,
    UKF,
    Craft,
    ReplayPredict,
    ReplayUpdate,
    Sim,
    TargetFilterReplay,
    TargetNumpy,
    World,
)
from manta.fields import FluidField, GravityField
from manta.parts import (
    IMU,
    Antenna,
    DisplacementHull,
    ExternalDCSupply,
    Mass,
    PositionSensor,
    PoweredThruster,
    VelocitySensor,
)
from manta.simulation import (
    BatteryCell,
    BatteryCellFaults,
    BatteryStepInput,
    SeriesBatteryPack,
)

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="cc is required")


def _battery_input(pack: SeriesBatteryPack, current: float) -> BatteryStepInput:
    return BatteryStepInput(
        requested_series_current=current,
        cell_temperatures=(298.15,) * len(pack.cells),
        cell_faults=(BatteryCellFaults(),) * len(pack.cells),
        balance_enabled=(False,) * len(pack.balancers),
    )


def _blueboat_world() -> tuple[World, DisplacementHull]:
    boat = Craft("blueboat")
    boat.add(Mass("body", mass=15.0, moi=(2.0, 2.5, 3.5)))
    hull = DisplacementHull(
        "hull",
        dimensions=(1.2, 0.93, 0.30),
        displacement_volume=15.0 / 1025.0,
        sample_resolution=(2, 1, 4),
    )
    boat.add(hull)

    supply = ExternalDCSupply("pack_supply")
    port = PoweredThruster(
        "port",
        force=(35.0, 0.0, 0.0),
        mount_offset=(-0.25, 0.32, -0.05),
        rated_voltage=16.0,
        rated_mechanical_power=140.0,
        conversion_efficiency=0.8,
        brownout_voltage=8.0,
        recovery_voltage=10.0,
    )
    starboard = PoweredThruster(
        "starboard",
        force=(35.0, 0.0, 0.0),
        mount_offset=(-0.25, -0.32, -0.05),
        rated_voltage=16.0,
        rated_mechanical_power=140.0,
        conversion_efficiency=0.8,
        brownout_voltage=8.0,
        recovery_voltage=10.0,
    )
    supply.connect(port)
    supply.connect(starboard)
    boat.add(supply)
    boat.add(port)
    boat.add(starboard)
    boat.add(
        Antenna(
            "starlink",
            mount_offset=(0.0, 0.0, 0.55),
            mount_orientation=(2**-0.5, 0.0, 2**-0.5, 0.0),
            frequency_hz=12.0e9,
        )
    )

    world = World(name="mission_stack_blueboat")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_field(FluidField().add_flat_ocean())
    world.add_craft(boat, position=(4.0, -3.0, 0.0))
    return world, hull


def test_compiled_blueboat_composes_hull_antenna_and_battery_boundary(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    world, hull = _blueboat_world()
    transform = Sim(world)
    module = transform.module()
    state_names = {slot.name for slot in transform.sys.spec.slots}
    input_names = {field.name for field in module.port("u").fields}

    # Battery electrochemistry and faults are a Python simulation sidecar.
    # Only its supplied voltage crosses into the differentiable Manta model.
    assert not any("battery" in name or "soc" in name for name in state_names)
    assert "blueboat.pack_supply.supplied_voltage" in input_names
    assert len(hull.samples) == 8

    pack = SeriesBatteryPack([BatteryCell()] * 4, initial_soc=0.8, seed=17)
    open_circuit = pack.preview(_battery_input(pack, 0.0))
    before_soc = pack.state.cells[0].soc

    # compile=True proves that the aggregate model survives the generated C
    # path, including the fixed-row-vector output storage used by power data.
    sim = TargetNumpy(transform, compile=True)
    controls = open_circuit.supply_inputs("pack_supply")
    controls.update({"port.throttle": 0.6, "starboard.throttle": 0.6})
    sim.step(0.01, u=controls)
    outputs = sim.outputs()["blueboat"]

    orientation = np.asarray(outputs["starlink.orientation"], dtype=float).ravel()
    np.testing.assert_allclose(
        orientation,
        (2**-0.5, 0.0, 2**-0.5, 0.0),
        atol=1e-12,
    )
    current = float(np.asarray(outputs["pack_supply.output_current"]).item())
    assert current > 0.0

    pack.step(0.01, _battery_input(pack, current))
    assert pack.state.cells[0].soc < before_soc
    # Advancing the sidecar never adds SOC, cell faults, or BMS state to the
    # compiled plant's state vector.
    assert {slot.name for slot in transform.sys.spec.slots} == state_names


def _mako_filter_world() -> World:
    mako = Craft("mako")
    mako.add(Mass("body", mass=45.0, moi=(2.0, 8.0, 8.0)))
    mako.add(IMU("imu", accel_noise_sigma=0.05, gyro_noise_sigma=0.005))
    mako.add(VelocitySensor("dvl", velocity_noise_sigma=0.03))
    mako.add(PositionSensor("sbl", position_noise_sigma=1.0))
    world = World(name="mission_stack_mako_filter")
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_field(FluidField().add_flat_ocean())
    world.add_craft(mako, position=(0.0, 0.0, -20.0))
    return world


@pytest.mark.parametrize("estimator", (EKF, UKF))
def test_mako_generated_filter_and_native_replay_coexist(
    estimator, monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    transform = estimator(_mako_filter_world(), gates={"sbl.position": 16.3})
    # Use the ordinary runtime as an independent oracle.  TargetFilterReplay
    # compiles the same transform's filter kernels into its native span runner;
    # neither path gets a second model.
    live = TargetNumpy(transform)
    replay = TargetFilterReplay(
        transform,
        max_operations=4,
        max_checkpoints=2,
    )
    operations = (
        ReplayUpdate(
            0.0,
            "mako.sbl.position",
            (0.2, -0.1, -20.0),
            measurement_covariance=np.eye(3) * 0.5,
            checkpoint=True,
        ),
        ReplayPredict(0.0, 0.01, checkpoint=True),
    )
    result = replay.run(replay.program(live.checkpoint(), operations))

    oracle_update = live.update(
        "mako.sbl.position",
        (0.2, -0.1, -20.0),
        R=np.eye(3) * 0.5,
    )
    live.predict(0.01)
    expected = live.checkpoint()
    np.testing.assert_allclose(result.final.x, expected.x, atol=1e-12)
    np.testing.assert_allclose(result.final.P, expected.P, atol=1e-11)
    assert result.final.time == pytest.approx(0.01)
    assert result.updates[0][1].accepted is oracle_update.accepted
    assert len(result.checkpoints) == 2
