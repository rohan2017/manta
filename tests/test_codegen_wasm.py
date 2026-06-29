"""The WASM backend: emission shape + Python ↔ C-ABI ↔ JS roundtrips.

`TargetWasm` reuses the C++ backend's exact math path (densify +
`emit_kernel_list`) and adds only the marshalling glue — a flat-double C ABI
over CasADi's `arg[]`/`res[]` arrays, a JSON descriptor, and a JS runtime.
The tests prove, at every layer the toolchain on PATH allows:

  1. emission shape — files + a self-consistent descriptor (always);
  2. the emitted C ABI computes the same numbers as `TargetNumpy`, compiled
     by a plain C compiler — no Eigen, no Emscripten (skips without `cc`);
  3. the JS runtime module is valid ES (skips without `node`);
  4. the full Emscripten `.wasm` + JS `Sim` matches numpy end-to-end (skips
     without `emcc` + `node`).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manta import EKF, LQR, Craft, Sim, TargetNumpy, TargetWasm, World
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster


def _hover_world():
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g"))
    c.add(PositionSensor("gps"))
    w = World(name="hover_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


def _numpy_reference(n=200, dt=0.005):
    """`(final position, final gps reading)` from the numpy oracle — the
    ground truth every backend must reproduce."""
    sim = TargetNumpy(Sim(_hover_world()))
    sim.state["drone"]["position"] = np.array([0.0, 0.0, 5.0])
    sim.state["drone"]["velocity"] = np.array([0.5, 0.0, 0.0])
    sim.state["drone"]["t.throttle"] = 1.5 * 9.81
    for _ in range(n):
        sim.step(dt)
    return (np.asarray(sim.state["drone"]["position"]).ravel(),
            np.asarray(sim.reading("gps.position")).ravel())


def _step_entry(desc):
    return next(e for e in desc["entryPoints"] if e["method"] == "step")


# ---------------------------------------------------------------------------
# 1. Emission shape
# ---------------------------------------------------------------------------

def test_wasm_emission_shape(tmp_path: Path):
    res = TargetWasm(Sim(_hover_world()), tmp_path, class_name="Drone")

    for p in (res.kernels_c, res.kernels_h, res.abi_c, res.descriptor,
              res.js, res.build_sh):
        assert p.exists(), p

    desc = res.descriptor_obj
    assert desc["hosting"] == "threaded"
    assert desc["ambientDim"] == 13
    # state slots tile the ambient vector exactly, in order.
    off = 0
    for s in desc["stateSlots"]:
        assert s["off"] == off
        off += s["dim"]
    assert off == desc["ambientDim"]
    assert len(desc["initialState"]) == desc["ambientDim"]

    step = _step_entry(desc)
    assert step["shim"] == "drone_step"
    # every in/out slot is contiguous and the totals match.
    for key, total in (("in", "inSize"), ("out", "outSize")):
        off = 0
        for s in step[key]:
            assert s["off"] == off
            off += s["size"]
        assert off == step[total]
    # the step writes the manifold state back and returns the sensors.
    assert [s for s in step["out"] if s.get("writesState")][0]["size"] == 13
    assert {s["name"] for s in step["out"] if s["kind"] == "measurement"} == {
        "drone.g.gyro", "drone.g.accel", "drone.gps.position"}

    abi = res.abi_c.read_text()
    assert "int drone_step(const double* in, double* out)" in abi
    assert "EMSCRIPTEN_KEEPALIVE" in abi
    assert "k_drone_step(arg, res, iw, w, 0)" in abi      # internal kernel sym

    build = res.build_sh.read_text()
    assert "drone_kernels.c" in build and "drone_abi.c" in build
    assert "emcc" in build

    js = res.js.read_text()
    assert 'import createModule from "./drone.mjs"' in js
    assert "export class Sim" in js and "export async function load" in js
    # the descriptor is embedded so the JS needs no separate fetch.
    assert json.loads(json.dumps(desc))  # round-trippable
    assert '"ambientDim": 13' in js


# ---------------------------------------------------------------------------
# 2. C-ABI roundtrip (plain C compiler — no Eigen, no Emscripten)
# ---------------------------------------------------------------------------

_HARNESS = r"""
#include <stdio.h>
#include <string.h>
int drone_step(const double* in, double* out);
int main(void) {{
    double in[{in_size}]; double out[{out_size}];
    memset(in, 0, sizeof in);
    in[{pos}]   = 0.0; in[{pos}+1] = 0.0; in[{pos}+2] = 5.0;
    in[{quat}]  = 1.0;                       /* identity orientation */
    in[{vel}]   = 0.5; in[{vel}+1] = 0.0; in[{vel}+2] = 0.0;
    in[{ctrl}]  = 1.5 * 9.81;                /* throttle */
    double dt = 0.005, t = 0.0;
    for (int i = 0; i < 200; i++) {{
        in[{dt}] = dt; in[{tm}] = t;
        drone_step(in, out);
        memcpy(in, out, {amb} * sizeof(double));   /* threaded state back */
        t += dt;
    }}
    printf("pos %.17g %.17g %.17g\n", out[{pos}], out[{pos}+1], out[{pos}+2]);
    printf("gps %.17g %.17g %.17g\n", out[{gps}], out[{gps}+1], out[{gps}+2]);
    return 0;
}}
"""


def test_wasm_abi_native_roundtrip(tmp_path: Path):
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc is None:
        pytest.skip("no C compiler on PATH")

    res = TargetWasm(Sim(_hover_world()), tmp_path, class_name="Drone")
    desc = res.descriptor_obj
    slots = {s["name"]: s["off"] for s in desc["stateSlots"]}
    step = _step_entry(desc)
    in_of = {s["kind"]: s["off"] for s in step["in"]}
    gps = next(s["off"] for s in step["out"]
               if s["name"] == "drone.gps.position")

    harness = tmp_path / "harness.c"
    harness.write_text(_HARNESS.format(
        in_size=step["inSize"], out_size=step["outSize"],
        amb=desc["ambientDim"],
        pos=slots["drone.position"], quat=slots["drone.orientation"],
        vel=slots["drone.velocity"], ctrl=in_of["control"],
        dt=in_of["timestep"], tm=in_of["time"], gps=gps))

    binary = tmp_path / "harness"
    p = subprocess.run(
        [cc, "-O2", f"-I{tmp_path}", "-Wno-unused-parameter",
         "-Wno-unused-variable", str(res.kernels_c), str(res.abi_c),
         str(harness), "-lm", "-o", str(binary)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = {ln.split()[0]: [float(x) for x in ln.split()[1:]]
           for ln in p.stdout.strip().splitlines()}

    pos_ref, gps_ref = _numpy_reference()
    np.testing.assert_allclose(got["pos"], pos_ref, atol=1e-10)
    np.testing.assert_allclose(got["gps"], gps_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. JS runtime is valid ES
# ---------------------------------------------------------------------------

def test_wasm_js_is_valid_esm(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    res = TargetWasm(Sim(_hover_world()), tmp_path, class_name="Drone")
    mjs = tmp_path / "drone_check.mjs"
    mjs.write_text(res.js.read_text())
    p = subprocess.run(["node", "--check", str(mjs)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


# ---------------------------------------------------------------------------
# 4. Full Emscripten .wasm + JS Sim roundtrip
# ---------------------------------------------------------------------------

_NODE_HARNESS = r"""
import {{ load }} from "./drone_rt.mjs";
const rt = await load();
const sim = rt.sim();
sim.state[{pos}] = 0.0; sim.state[{pos}+2] = 5.0;
sim.state[{vel}] = 0.5;
for (let i = 0; i < 200; i++) sim.step(0.005, {{ "t.throttle": 1.5 * 9.81 }});
const pos = sim.slot("position");
const gps = sim.reading("gps.position");
console.log(JSON.stringify({{ pos: Array.from(pos), gps: Array.from(gps) }}));
"""


# A stub Emscripten module: a fixed linear-memory ArrayBuffer, a bump
# `_malloc`, and a `ccall` that fakes `step` (copy state through, integrate
# position by velocity*dt, mirror gps = position). It exercises the real JS
# `Runtime`/`Sim` plumbing — heap views, control packing, state copy-back,
# suffix lookup — with no Emscripten toolchain, deterministically.
_STUB_FACTORY = r"""
export default async function () {
  const buf = new ArrayBuffer(1 << 16);
  const HEAPF64 = new Float64Array(buf);
  let top = 16;
  return {
    HEAPF64,
    _malloc(n) { const p = top; top += n; return p; },
    _free() {},
    ccall(_shim, _ret, _types, [inPtr, outPtr]) {
      const L = globalThis.__LAYOUT;
      const i = inPtr / 8, o = outPtr / 8;
      for (let k = 0; k < L.amb; k++) HEAPF64[o + k] = HEAPF64[i + k];
      const dt = HEAPF64[i + L.dt];
      for (let k = 0; k < 3; k++)
        HEAPF64[o + L.pos + k] += HEAPF64[i + L.vel + k] * dt;
      for (let k = 0; k < 3; k++) HEAPF64[o + L.gps + k] = HEAPF64[o + L.pos + k];
      return 0;
    },
  };
}
"""

_STUB_HARNESS = r"""
import { load, descriptor } from "./drone_rt.mjs";
const sslot = (n) => descriptor.stateSlots.find((s) => s.name.endsWith(n)).off;
const step = descriptor.entryPoints.find((e) => e.method === "step");
const inOff = (k) => step.in.find((s) => s.kind === k).off;
globalThis.__LAYOUT = {
  amb: descriptor.ambientDim, pos: sslot("position"), vel: sslot("velocity"),
  dt: inOff("timestep"),
  gps: step.out.find((s) => s.name.endsWith("gps.position")).off,
};
const sim = (await load()).sim();
sim.state[sslot("position") + 2] = 5.0;     // start at z = 5
sim.state[sslot("velocity")] = 0.5;         // vx = 0.5
for (let n = 0; n < 200; n++) sim.step(0.005, { "t.throttle": 1.5 * 9.81 });
console.log(JSON.stringify({ pos: Array.from(sim.slot("position")),
                             gps: Array.from(sim.reading("gps.position")) }));
"""


def test_wasm_js_runtime_plumbing(tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    res = TargetWasm(Sim(_hover_world()), tmp_path, class_name="Drone")
    # `.mjs` copies so node loads them as ES modules; the runtime's internal
    # `./drone.mjs` factory import resolves to our stub in the same dir.
    (tmp_path / "drone.mjs").write_text(_STUB_FACTORY)
    (tmp_path / "drone_rt.mjs").write_text(res.js.read_text())
    (tmp_path / "harness.mjs").write_text(_STUB_HARNESS)

    p = subprocess.run(["node", str(tmp_path / "harness.mjs")], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])
    # vx=0.5 integrated for 200·0.005 s = 1 s ⇒ x advances 0.5; gps mirrors pos.
    np.testing.assert_allclose(got["pos"], [0.5, 0.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(got["gps"], [0.5, 0.0, 5.0], atol=1e-12)


def test_wasm_full_emscripten_roundtrip(tmp_path: Path):
    if shutil.which("emcc") is None or shutil.which("node") is None:
        pytest.skip("emcc + node required for the full WASM roundtrip")

    res = TargetWasm(Sim(_hover_world()), tmp_path, class_name="Drone")
    p = subprocess.run(["sh", str(res.build_sh)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "drone.wasm").exists()

    # node treats `.js` as CommonJS; the runtime is ESM, so import a `.mjs`
    # copy (its internal `./drone.mjs` factory import resolves in-dir).
    (tmp_path / "drone_rt.mjs").write_text(res.js.read_text())
    slots = {s["name"]: s["off"] for s in res.descriptor_obj["stateSlots"]}
    harness = tmp_path / "run.mjs"
    harness.write_text(_NODE_HARNESS.format(
        pos=slots["drone.position"], vel=slots["drone.velocity"]))

    p = subprocess.run(["node", str(harness)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])

    pos_ref, gps_ref = _numpy_reference()
    np.testing.assert_allclose(got["pos"], pos_ref, atol=1e-9)
    np.testing.assert_allclose(got["gps"], gps_ref, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. The Filter / Regulator views — EKF & LQR get the same lowering as Sim
# ---------------------------------------------------------------------------

def _ekf_world():
    """`_hover_world` with sensor noise declared, so the EKF's per-sensor R is
    nonsingular (a noiseless sensor bakes a zero R the first fold can't invert)."""
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("g", gyro_noise_sigma=0.005, accel_noise_sigma=0.05,
              gyro_bias_sigma=1e-4))
    c.add(PositionSensor("gps", position_noise_sigma=0.02))
    w = World(name="ekf_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


def _lqr_world():
    """A 3-thruster body: position + velocity controllable, attitude frozen."""
    c = Craft("drone")
    c.add(Mass("body", mass=1.5, moi=(0.05, 0.05, 0.08)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(Thruster("tx", force=(1.0, 0.0, 0.0)))
    c.add(Thruster("ty", force=(0.0, 1.0, 0.0)))
    w = World(name="lqr_world")
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


def _build_lqr():
    return LQR(_lqr_world(),
               x_ref={"drone": {"position": (0, 0, 10), "velocity": (0, 0, 0)}},
               u_ref={"t.throttle": 1.5 * 9.81},
               regulate=["drone.position", "drone.velocity"],
               Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3), dt=0.02)


def _emcc_node(tmp_path, res, base, harness):
    """Build the bundle with Emscripten and run `harness` under node, returning
    its last stdout line parsed as JSON. Skips without the toolchain."""
    if shutil.which("emcc") is None or shutil.which("node") is None:
        pytest.skip("emcc + node required for the full WASM roundtrip")
    p = subprocess.run(["sh", str(res.build_sh)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    # node loads `.js` as CommonJS; copy the ESM runtime to `.mjs` (its internal
    # `./<base>.mjs` factory import resolves to the emcc output in the same dir).
    (tmp_path / f"{base}_rt.mjs").write_text(res.js.read_text())
    (tmp_path / "run.mjs").write_text(harness)
    r = subprocess.run(["node", str(tmp_path / "run.mjs")], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_wasm_descriptor_carries_filter_and_regulator_state(tmp_path: Path):
    """The descriptor exposes the held state a stateful view needs to seed
    itself — the filter's `x`/`P` fields, the regulator's `x`/`x_ref` ports —
    and the JS emits the matching typed views."""
    ekf = TargetWasm(EKF(_ekf_world()), tmp_path / "ekf", class_name="E")
    fields = {f["name"]: f for f in ekf.descriptor_obj["stateFields"]}
    assert set(fields) == {"x", "P"}
    assert fields["x"]["kind"] == "state"
    assert fields["P"]["kind"] == "matrix"
    n = ekf.descriptor_obj["ambientDim"]
    t = ekf.descriptor_obj["tangentDim"]
    assert fields["x"]["shape"] == [n] and len(fields["x"]["init"]) == n
    assert fields["P"]["shape"] == [t, t] and len(fields["P"]["init"]) == t * t

    lqr = TargetWasm(_build_lqr(), tmp_path / "lqr", class_name="R")
    refs = {r["name"]: r for r in lqr.descriptor_obj["stateRefs"]}
    assert set(refs) == {"x", "x_ref"}
    assert refs["x_ref"]["dim"] == lqr.descriptor_obj["ambientDim"]

    for res in (ekf, lqr):
        js = res.js.read_text()
        assert "export class Filter" in js and "export class Regulator" in js
        assert "filter()" in js and "regulator()" in js


def test_wasm_filter_roundtrip(tmp_path: Path):
    """The `Filter` view (held `x`/`P`, `update` then `predict`) matches the
    numpy filter end-to-end through Emscripten."""
    u = {"t.throttle": 14.7}
    gyro, gps = [0.01, 0.02, 0.03], [0.05, -0.03, 5.1]

    ekf = TargetNumpy(EKF(_ekf_world()))
    for _ in range(5):
        ekf.update("g.gyro", gyro, u=u)
        ekf.update("gps.position", gps, u=u)
        ekf.predict(0.01, u=u)

    res = TargetWasm(EKF(_ekf_world()), tmp_path, class_name="DroneEkf")
    harness = """
import { load } from "./droneekf_rt.mjs";
const flt = (await load()).filter();
const u = { "t.throttle": 14.7 };
for (let i = 0; i < 5; i++) {
  flt.update("g.gyro", [0.01, 0.02, 0.03], { u });
  flt.update("gps.position", [0.05, -0.03, 5.1], { u });
  flt.predict(0.01, { u });
}
console.log(JSON.stringify({ x: Array.from(flt.x), P: Array.from(flt.P) }));
"""
    got = _emcc_node(tmp_path, res, "droneekf", harness)
    np.testing.assert_allclose(got["x"], ekf.x, atol=1e-9)
    # JS P is the kernel's flat (column-major) covariance.
    np.testing.assert_allclose(got["P"], ekf.P.ravel(order="F"), atol=1e-9)


def test_wasm_regulator_roundtrip(tmp_path: Path):
    """The `Regulator` view (`control(state)` over a held reference) matches
    the numpy regulator end-to-end through Emscripten."""
    reg = TargetNumpy(_build_lqr())
    xt = reg.x_ref.copy()
    xt[0] += 1.0; xt[1] += 2.0; xt[7] += 0.3        # off-setpoint pos + vel
    u_ref = reg.u(xt)

    res = TargetWasm(_build_lqr(), tmp_path, class_name="DroneLqr")
    harness = f"""
import {{ load, descriptor }} from "./dronelqr_rt.mjs";
const reg = (await load()).regulator();
const out = reg.control(Float64Array.from({json.dumps(xt.tolist())}));
console.log(JSON.stringify(descriptor.inputs.map((f) => out[f.name])));
"""
    got = _emcc_node(tmp_path, res, "dronelqr", harness)
    np.testing.assert_allclose(got, u_ref, atol=1e-9)
