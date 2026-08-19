"""Smoke tests for the Mako compile server.

Run either way::

    .venv/bin/python -m pytest server/mako/test_smoke.py -v
    .venv/bin/python -m server.mako.test_smoke

Covers: design validation, Earth-world construction + a 50-step numpy sim
(no NaNs, craft holds together in scene coordinates), sensor readings
(hydrostatic depth, GPS wet gate), seabed contact (colliders stop a sinking
hull), bundle meta (mixer/scene/waves blocks), TargetWasm emission (emcc
compile only when on PATH), and the noise-slot semantics check the browser
client depends on (are injected draws expected pre-scaled by σ?).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from manta import Craft, NoiseDriver, Sim, TargetNumpy, World
from manta.parts import Mass, PositionSensor

from .builder import analyze, build_bundle, build_world, design_hash, \
    validate_design
from .catalog import BUOY_RESERVE, CRAFT

# Everything in this module is toolchain-or-build shaped (WASM emission,
# emcc when present, full bundle builds), so the whole module carries both
# markers. Module-level pytestmark — NOT a conftest hook: a conftest's
# pytest_collection_modifyitems receives the ENTIRE session's items, and a
# previous version of this marking silently stamped all 749 repo tests.
pytestmark = [pytest.mark.cpp, pytest.mark.slow]


# --- the representative design ----------------------------------------------
# The document's natural fit: DVL nose, front agility cluster (heave/pitch/
# roll + canted surge/sway/yaw pair), brain (GPS + depth), INS, battery,
# rear agility cluster, fin-control stern (prop + 4 fins + GPS). Full 6-DOF
# authority and every sensor path exercised.

def _design() -> dict:
    return {
        "version": 2,
        "name": "smoke mako",
        "spine": [
            {"type": "nose_dvl", "options": {}},
            {"type": "agility_front", "options": {}},
            {"type": "brain", "options": {}},
            {"type": "ins", "options": {}},
            {"type": "battery", "options": {}},
            {"type": "agility_rear", "options": {}},
            {"type": "fin_control", "options": {}},
        ],
        "spawn": {"x": 0, "y": 0, "depth": 2.0, "heading_deg": 0},
    }


def _canonical():
    canonical, errors = validate_design(_design())
    assert not errors, errors
    return canonical


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_validation_catches_bad_designs():
    def errs(d):
        _, e = validate_design(d)
        return e

    assert errs("not a dict")                       # not an object
    assert errs({"spine": []})                      # empty spine
    # wrong caps: a spine-kind module first, a nose at the end, a rear in
    # the middle
    assert errs({"spine": [{"type": "brain"}, {"type": "fin_control"}]})
    assert errs({"spine": [{"type": "nose_dvl"}, {"type": "nose_camera"}]})
    assert errs({"spine": [{"type": "nose_dvl"}, {"type": "rear_thrust"},
                           {"type": "fin_control"}]})
    # unknown type
    assert errs({"spine": [{"type": "warp_drive"},
                           {"type": "fin_control"}]})
    # non-numeric option
    assert errs({"spine": [
        {"type": "nose_dvl", "options": {"dvl_noise": "quiet"}},
        {"type": "fin_control"}]})


def test_options_clamped_and_hash_stable():
    d = _design()
    d["spine"][0]["options"]["dvl_noise"] = 99.0    # clamps to 0.2
    d["spine"][0]["options"]["junk"] = 42           # ignored
    canonical, errors = validate_design(d)
    assert not errors
    assert canonical["spine"][0]["options"]["dvl_noise"] == 0.2
    assert "junk" not in canonical["spine"][0]["options"]
    # equivalent designs share a hash; name is cosmetic
    d2 = _design()
    d2["spine"][0]["options"]["dvl_noise"] = 123.0
    d2["name"] = "totally different name"
    c2, _ = validate_design(d2)
    assert design_hash(canonical) == design_hash(c2)
    # spawn clamped
    d3 = _design()
    d3["spawn"] = {"x": 9999, "depth": 0.01, "heading_deg": 725}
    c3, errors = validate_design(d3)
    assert not errors
    assert c3["spawn"]["x"] == 200.0
    assert c3["spawn"]["depth"] == 0.5
    assert abs(c3["spawn"]["heading_deg"] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# world + sim
# ---------------------------------------------------------------------------

def test_world_builds_and_sim_steps_without_nan():
    world, craft, contrib, scene = build_world(_canonical())
    analysis = analyze(contrib)
    assert np.isfinite(analysis["mass"]) and 30 < analysis["mass"] < 120
    # every module carries the same reserve buoyancy — the craft's net
    # lift is exactly mass·(BUOY_RESERVE − 1)·g (dead-ship float-up)
    expected = analysis["mass"] * (BUOY_RESERVE - 1.0) * 9.81
    assert analysis["buoyancy_n"] > 0.5
    assert abs(analysis["buoyancy_n"] - expected) < 0.1 * expected
    assert all(np.isfinite(v) for v in analysis["trim"].values())

    sim = TargetNumpy(Sim(world))
    sim.attach_driver(NoiseDriver(seed=7))
    u = dict(analysis["trim"])          # hold trim so it roughly stays put
    for _ in range(50):
        sim.step(0.02, u=u)
    rel = scene.relative(sim.state[CRAFT], t=1.0)   # scene ENU coordinates
    pos = np.asarray(rel["position"], dtype=float)
    quat = np.asarray(rel["orientation"], dtype=float)
    vel = np.asarray(rel["velocity"], dtype=float)
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(quat)) \
        and np.all(np.isfinite(vel))
    # roughly holds together: still near the spawn point (waves push it a
    # little), unit quaternion, ground-relative velocity small
    assert np.linalg.norm(pos - np.array([0.0, 0.0, -2.0])) < 1.5
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-6
    assert np.linalg.norm(vel) < 2.0


def test_control_surface_world_steps():
    # the fin-control stern carries 4 ControlSurfaces (extra craft states)
    world, craft, contrib, scene = build_world(_canonical())
    names = [i["name"] for i in contrib.inputs]
    assert sum(n.endswith(".deflection_cmd") for n in names) == 4
    sim = TargetNumpy(Sim(world))
    for _ in range(5):
        sim.step(0.02)
    assert np.all(np.isfinite(
        np.asarray(sim.state[CRAFT]["position"], dtype=float)))


def test_seabed_colliders_catch_a_sinking_hull():
    # Spawn deep (24 m, seabed at 30 m) and pull the craft down with a
    # steady external heave demand: it must come to rest ON the seabed
    # (hull axis ≈ seabed + HULL_R) instead of tunnelling through.
    d = _design()
    d["spawn"]["depth"] = 24.0
    canonical, errors = validate_design(d)
    assert not errors
    world, craft, contrib, scene = build_world(canonical)
    sim = TargetNumpy(Sim(world))
    # drive all four vertical agility pods DOWN hard
    u = {f"{CRAFT}.{t['part']}.throttle": -220.0
         for t in contrib.thrusters if abs(t["axis"][2]) > 0.9}
    t = 0.0
    for _ in range(600):                 # 6 s at 10 ms — enough to settle
        sim.step(0.01, u=u)
        t += 0.01
    rel = scene.relative(sim.state[CRAFT], t=t)
    z = float(rel["position"][2])
    assert np.isfinite(z)
    # resting on (slightly into) the contact plane, NOT through the floor
    assert -30.5 < z < -29.5, f"hull axis settled at z = {z:.2f}"


# ---------------------------------------------------------------------------
# sensors + bundle meta
# ---------------------------------------------------------------------------

def test_sensors_read_physically():
    world, craft, contrib, scene = build_world(_canonical())
    sim = TargetNumpy(Sim(world))
    sim.step(0.02)
    # SPEC part naming: readings resolve for every sensor path
    for name in (f"{CRAFT}.ins3_imu.gyro",
                 f"{CRAFT}.nose_dvl0_dvl.velocity",
                 f"{CRAFT}.brain2_gps.position",
                 f"{CRAFT}.fin_control6_gps.position"):
        assert np.all(np.isfinite(
            np.asarray(sim.reading(name), dtype=float)))
    # the barometer sees the hydrostatic column: ~101325 + ρ·g·2 at 2 m down
    p = float(np.asarray(
        sim.reading(f"{CRAFT}.brain2_depth.pressure")).ravel()[0])
    assert 110_000 < p < 132_000
    # the gps antenna (z = +0.16 on the hull) is underwater at spawn depth
    wet = float(np.asarray(
        sim.reading(f"{CRAFT}.brain2_gps.wet")).ravel()[0])
    assert wet > 0.9


def test_bundle_meta_contract():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        meta = build_bundle(_canonical(), tmp, name="smoke", compile=False)

        # trim finite and inside the input limits
        limits = {i["name"].removeprefix(f"{CRAFT}."): (i["min"], i["max"])
                  for i in meta["inputs"]}
        for k, v in meta["trim"].items():
            lo, hi = limits[k]
            assert np.isfinite(v) and lo <= v <= hi

        # mixer: one alloc row per thruster input, 6 wrench columns each
        mx = meta["mixer"]
        assert len(mx["alloc"]) == len(mx["inputs"])
        assert all(len(row) == 6 for row in mx["alloc"])
        assert all(np.isfinite(v) for row in mx["alloc"] for v in row)
        assert len(mx["wrench_cap"]) == 6
        # the starter has full 6-DOF authority + 4 fins (2 yaw + 2 pitch)
        assert all(mx["axes"].values()), mx["axes"]
        assert sorted(f["axis"] for f in mx["fins"]) == \
            ["pitch", "pitch", "yaw", "yaw"]
        assert all(f["k"] > 0 for f in mx["fins"])
        # roll channel: signed roll_k, opposing within each pair (collective
        # deflection → pure pitch/yaw, differential → pure roll)
        assert all(abs(f["roll_k"]) > 0 for f in mx["fins"])
        for axis in ("pitch", "yaw"):
            pair = [f["roll_k"] for f in mx["fins"] if f["axis"] == axis]
            assert len(pair) == 2 and pair[0] == -pair[1], pair
        # a pure heave demand loads the vertical pods, not the stern prop
        w = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
        u = np.asarray(mx["alloc"]) @ w
        by_name = dict(zip(mx["inputs"], u))
        assert all(abs(v) < 1.0 for n, v in by_name.items() if "_prop" in n)
        assert sum(abs(v) > 5.0 for n, v in by_name.items() if "_v" in n) == 4

        # scene: rotation about +z at the sidereal rate, anchored at the
        # north-pole sea surface (the WGS-84 polar radius)
        sc = meta["scene"]
        assert abs(sc["omega"] - 7.2921159e-5) < 1e-9
        assert sc["axis"] == [0.0, 0.0, 1.0]
        assert abs(sc["anchor_planet"][2] - 6.356752314e6) < 1.0
        R = np.asarray(sc["R_planet_from_scene"])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)   # orthonormal

        # waves: deep-water dispersion ω² = g·k
        wv = meta["waves"]
        assert abs(wv["omega"] ** 2 - sc["g0"] * wv["k"]) < 1e-6
        assert wv["amplitude"] > 0 and wv["wavelength"] > 0

        # meta.json written and matches the return value
        on_disk = json.loads((Path(tmp) / "meta.json").read_text())
        assert on_disk["mixer"] == meta["mixer"]
        assert on_disk["inputs"] == meta["inputs"]


# ---------------------------------------------------------------------------
# WASM emission (+ optional emcc compile)
# ---------------------------------------------------------------------------

def test_wasm_emission_produces_expected_files():
    with tempfile.TemporaryDirectory() as tmp:
        has_emcc = shutil.which("emcc") is not None
        meta = build_bundle(_canonical(), tmp, name="smoke",
                            compile=has_emcc)
        out = Path(tmp)
        expected = ["meta.json"]
        for base in ("sim",):
            expected += [f"{base}.js", f"{base}.descriptor.json",
                         f"{base}_kernels.c", f"{base}_abi.c",
                         f"build_{base}.sh"]
            if has_emcc:
                expected += [f"{base}.mjs", f"{base}.wasm"]
        missing = [f for f in expected if not (out / f).exists()]
        assert not missing, f"missing bundle files: {missing}"
        assert meta["files"] == {"sim": "sim.js"}
        # meta.inputs order matches the sim descriptor's input order
        import json
        desc = json.loads((out / "sim.descriptor.json").read_text())
        assert [i["name"] for i in meta["inputs"]] == \
            [f["name"] for f in desc["inputs"]]
        # every noise field carries its σ for the client's Box-Muller draws
        assert all("sigma" in f for f in desc["noise"])


# ---------------------------------------------------------------------------
# CRITICAL: noise-slot semantics (what the browser must inject)
# ---------------------------------------------------------------------------

def test_noise_slot_expects_pre_scaled_draws():
    """The Sim `step` kernel adds the noise inputs VERBATIM — σ lives in the
    driver, not the kernel. So the browser must inject σ·N(0,1) (Box-Muller
    unit normals SCALED by each noise field's descriptor σ), NOT raw unit
    normals.

    Two checks: (1) inject a known vector into the noise slot of a static
    craft and see it appear 1:1 in the reading (a σ-scaling kernel would
    multiply it by σ=0.1); (2) NoiseDriver itself samples N(0, σ²), i.e.
    already-scaled draws.
    """
    sigma = 0.1
    c = Craft("t")
    c.add(Mass("m", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(PositionSensor("gps", position_noise_sigma=sigma))
    w = World()                       # no gravity → truly static craft
    w.add_craft(c, position=(0.0, 0.0, 0.0))
    sim = TargetNumpy(Sim(w))

    inject = np.array([1.0, -2.0, 3.0])
    sim.state["t"]["gps.position_noise"] = inject   # manual noise threading
    sim.step(0.01)
    reading = np.asarray(sim.reading("t.gps.position"), dtype=float).ravel()
    truth = np.asarray(sim.state["t"]["position"], dtype=float).ravel()
    err_verbatim = np.abs(reading - (truth + inject)).max()
    err_scaled = np.abs(reading - (truth + sigma * inject)).max()
    assert err_verbatim < 1e-12, (
        f"kernel does NOT add noise verbatim (|err|={err_verbatim:.2e}, "
        f"σ-scaled |err|={err_scaled:.2e})")
    assert err_scaled > 1.0          # proves the two hypotheses differ

    # NoiseDriver draws are N(0, σ²): std ≈ σ, nowhere near 1.
    drv = NoiseDriver(seed=0)
    sim2 = TargetNumpy(Sim(w))
    sim2.attach_driver(drv)
    draws = np.array([drv.sample()["t.gps.position_noise"]
                      for _ in range(4000)]).ravel()
    assert abs(draws.std() - sigma) < 0.01 * 3      # ±3 mc-error bands


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        print(f"{n} ... ", end="", flush=True)
        f()
        print("ok")
    print(f"\n{len(fns)} smoke tests passed")
