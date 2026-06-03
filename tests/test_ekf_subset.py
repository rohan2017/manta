"""EKF state subsetting — `track=` lower bound + dependency closure,
opt-in sensors, and the SlotSet aliases.

The closure is symbolic (structural F sparsity): it never drops a slot a
kept slot depends on, drops fully-independent crafts cleanly, and pulls
in auxiliary states that a chosen sensor observes (auto-expand).
"""

import numpy as np

from manta import World, TargetNumpy
from manta.craft import Craft
from manta.fields import GravityField
from manta.estimation import EKF, ALL, POSE, TWIST, SlotSet, resolve_slotset
from manta.parts.structure.mass import Mass
from manta.parts.sensor.position_sensor import PositionSensor
from manta.parts.sensor.imu import IMU
from manta.parts.articulation.joint import Joint


def _gps_craft(name, sigma=0.0):
    c = Craft(name)
    c.add(Mass("body", mass=1.0))
    c.add(PositionSensor("gps", position_noise_sigma=sigma))
    return c


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

def test_slotset_aliases():
    assert (POSE | TWIST) == ALL
    assert resolve_slotset("drone", POSE) == {
        "drone.position", "drone.orientation"}
    assert resolve_slotset("drone", TWIST) == {
        "drone.velocity", "drone.angular_velocity"}
    assert resolve_slotset("drone", ALL) == {
        "drone.position", "drone.orientation",
        "drone.velocity", "drone.angular_velocity"}
    # A single-flag alias resolves to one slot.
    assert resolve_slotset("drone", SlotSet.VELOCITY) == {"drone.velocity"}


# ---------------------------------------------------------------------------
# track=None keeps the literal full spec (incl. part + bias slots)
# ---------------------------------------------------------------------------

def test_track_none_keeps_full_spec_with_aux_states():
    c = Craft("drone")
    c.add(Mass("body", mass=1.0))
    wheel = Joint("wheel", mode="passive")
    wheel.add(Mass("rim", mass=0.2, transform=(0.1, 0, 0)))
    c.add(wheel)
    c.add(IMU("imu", gyro_bias_sigma=1e-4))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c)

    ekf = EKF(w)  # track=None
    names = [s.name for s in ekf.spec.slots]
    # Rigid block + the joint state + the synthesized RW bias slot all
    # survive — the alias-less default never trims.
    assert "drone.position" in names
    assert any(n.startswith("drone.wheel.") for n in names)
    assert "drone.imu.gyro_bias" in names


# ---------------------------------------------------------------------------
# Multi-craft: tracking one craft drops the independent one entirely
# ---------------------------------------------------------------------------

def test_track_drops_independent_craft():
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(_gps_craft("a"))
    w.add_craft(_gps_craft("b"))

    ekf = EKF(w, track={"a": ALL})
    kept = [s.name for s in ekf.spec.slots]
    assert kept == ["a.position", "a.orientation",
                    "a.velocity", "a.angular_velocity"]
    assert ekf.spec.tangent_dim == 12          # not 24
    # Only craft a's sensor is registered (default sensors follow track).
    assert all(v["full"].startswith("a.") for v in ekf._sensors.values())


# ---------------------------------------------------------------------------
# Freeze-equivalence: the subset filter matches the full filter on the
# kept block when the dropped craft is genuinely independent.
# ---------------------------------------------------------------------------

def test_subset_matches_full_on_kept_block():
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(_gps_craft("a", sigma=0.2))
    w.add_craft(_gps_craft("b", sigma=0.2))

    full   = TargetNumpy(EKF(w))                 # estimates a and b
    subset = TargetNumpy(EKF(w, track={"a": ALL}))  # estimates only a

    gps_a = next(p for c in w.crafts if c.name == "a"
                 for p in c.parts if p.name == "gps")

    # Seed both with an identical (perturbed) craft-a estimate.
    P0_a = np.eye(12) * 0.5
    subset.reset(state={"a": {"position": [1.0, 0.0, 0.5]}}, P=P0_a)
    P0_full = np.eye(24) * 0.5
    full.reset(state={"a": {"position": [1.0, 0.0, 0.5]}}, P=P0_full)

    rng = np.random.default_rng(7)
    dt = 0.02
    for i in range(60):
        full.predict(dt=dt, t=i * dt)
        subset.predict(dt=dt, t=i * dt)
        z = np.array([0.0, 0.0, 100.0]) + rng.normal(0, 0.2, 3)
        full.update("a.gps.position", z)
        subset.update("a.gps.position", z)

    # Craft a is the first 12 tangent / 13 ambient slots of the full spec.
    np.testing.assert_allclose(subset.x[:13], full.x[:13], atol=1e-9)
    np.testing.assert_allclose(subset.P, full.P[:12, :12], atol=1e-9)


# ---------------------------------------------------------------------------
# Auto-expand: a chosen sensor pulls its observed aux state into K
# ---------------------------------------------------------------------------

def test_sensor_seed_pulls_in_gyro_bias():
    c = Craft("drone")
    c.add(Mass("body", mass=1.0))
    c.add(IMU("imu", gyro_bias_sigma=1e-4))
    w = World().add_field(GravityField(g=(0, 0, 0.0)))
    w.add_craft(c)

    # The gyro bias is a pure random-walk state — no rigid-body slot's
    # dynamics depend on it, so it only enters K when a sensor observes it.
    no_gyro = EKF(w, track={"drone": TWIST}, sensors=[])
    assert "drone.imu.gyro_bias" not in [s.name for s in no_gyro.spec.slots]

    with_gyro = EKF(w, track={"drone": TWIST}, sensors=["drone.imu.gyro"])
    assert "drone.imu.gyro_bias" in [s.name for s in with_gyro.spec.slots]


# ---------------------------------------------------------------------------
# Opt-in sensors: only the chosen outputs become measurements
# ---------------------------------------------------------------------------

def test_opt_in_sensor_exclusion():
    c = Craft("drone")
    c.add(Mass("body", mass=1.0))
    c.add(IMU("imu"))
    c.add(PositionSensor("gps", position_noise_sigma=0.1))
    w = World().add_field(GravityField(g=(0, 0, -9.81)))
    w.add_craft(c)

    e = EKF(w, sensors=["drone.gps.position"])
    fulls = {v["full"] for v in e._sensors.values()}
    assert fulls == {"drone.gps.position"}

    rt = TargetNumpy(e)
    # The IMU was excluded — feeding it is an error.
    try:
        rt.update("imu.gyro", np.zeros(3))
        raised = False
    except KeyError:
        raised = True
    assert raised
