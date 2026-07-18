// The live drive view: loads the compiled WASM sim for a spawned design and
// runs it at 60 Hz in a real spinning-Earth world. All state is re-expressed
// in the dive site's scene-ENU frame (meta.scene) for control and rendering;
// all control is client-side (web/mako/control.js) on the baked meta.mixer —
// manual wrench keys, or orientation/position PIDs (ori hold / auto).
//
// Loop structure mirrors web/quad/demo.js: fixed-timestep accumulator,
// FRAME = 0.02 s of control with DT = 0.004 s physics substeps, and pose
// interpolation between control frames for smooth rendering.

import { THREE, lerp, makeScene, fitToCanvas } from "../lib/render.js";
import { layout, makeCraftMesh } from "./catalog.js";
import { createController, makeSceneFrame, rpyOf } from "./control.js";
import { makeEnvironment } from "./environment.js";
import { createNavball } from "./navball.js";

const FRAME = 0.02, DT = 0.004, SUBSTEPS = Math.round(FRAME / DT);
// per-control-frame smoothing factor for the surface thrust gate (τ = 0.25 s)
const GATE_ALPHA = 1 - Math.exp(-FRAME / 0.25);

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

/** Box-Muller standard normal. */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Pack `{inputName: value}` over descriptor defaults into a flat vector
 *  (the generated runtime keeps its own packer private, so we mirror it). */
function makeControlPacker(descriptor) {
  const inputs = descriptor.inputs;
  const defaults = inputs.map((f) => f.default);
  return (u) => {
    const vec = [];
    inputs.forEach((f, i) => {
      const v = u[f.name] ?? defaults[i];
      if (f.dim === 1) vec.push(typeof v === "number" ? v : v[0]);
      else for (let k = 0; k < f.dim; k++) vec.push(v[k]);
    });
    return Float64Array.from(vec);
  };
}

// ---------------------------------------------------------------------------
// Noisy sim stepping — the generated `Sim` view is a noiseless oracle, so we
// call the `step` entry through the low-level Runtime with the noise slot
// filled with σ·N(0,1) draws. The kernels add the noise inputs VERBATIM (σ
// lives in the driver, not the kernel — verified in server/mako/test_smoke.py
// against numpy's NoiseDriver), so the draws must be pre-scaled by each
// noise field's descriptor σ.
// ---------------------------------------------------------------------------

function makeNoisySim(rt, descriptor) {
  const entry = descriptor.entryPoints.find((e) => e.method === "step");
  const packU = makeControlPacker(descriptor);
  const noiseSlot = entry.in.find((s) => s.kind === "noise");
  const noise = noiseSlot ? new Float64Array(noiseSlot.size) : null;
  // flat per-element σ, in descriptor noise-field order
  const sigma = noise ? new Float64Array(noiseSlot.size) : null;
  if (noise) {
    let off = 0;
    for (const f of descriptor.noise)
      for (let k = 0; k < f.dim; k++) sigma[off++] = f.sigma;
  }

  const sim = {
    t: 0,
    state: Float64Array.from(descriptor.initialState),
    outputs: {},
    slotMap: Object.fromEntries(descriptor.stateSlots.map(
      (s) => [s.name, s])),
    slot(name) {
      const s = this.slotMap[name];
      return this.state.slice(s.off, s.off + s.dim);
    },
    step(dt, u) {
      const args = {};
      for (const s of entry.in) {
        if (s.kind === "state") args[s.name] = this.state;
        else if (s.kind === "control") args[s.name] = packU(u);
        else if (s.kind === "timestep") args[s.name] = dt;
        else if (s.kind === "time") args[s.name] = this.t;
        else if (s.kind === "noise" && noise) {
          for (let i = 0; i < noise.length; i++)
            noise[i] = sigma[i] === 0 ? 0 : sigma[i] * randn();
          args[s.name] = noise;
        }
      }
      const out = rt.call("step", args);
      for (const s of entry.out) {
        if (s.writesState) this.state = out[s.name];
        else this.outputs[s.name] = out[s.name];
      }
      this.t += dt;
    },
  };
  return sim;
}

// ---------------------------------------------------------------------------
// The viewer
// ---------------------------------------------------------------------------

/** Boot the drive view for a compiled bundle. `base` is the bundle URL dir,
 *  `meta` the server's meta.json. `navballCanvas` hosts the artificial
 *  horizon; `setHud(data)` gets readout data; `onWarn(text)` fading
 *  warnings. Returns a controller for the page chrome. */
export async function startViewer(canvas, { design, base, meta, setHud,
                                            navballCanvas, onWarn }) {
  // --- load bundle -----------------------------------------------------------
  const simMod = await import(base + meta.files.sim);
  const simRt = await simMod.load();
  const sim = makeNoisySim(simRt, simMod.descriptor);
  const C = meta.craft;

  const sceneFrame = makeSceneFrame(meta.scene);
  const toRel = (state, t) => {
    const g = (n) => {
      const s = sim.slotMap[`${C}.${n}`];
      return state.slice(s.off, s.off + s.dim);
    };
    return sceneFrame.toScene(g("position"), g("orientation"),
                              g("velocity"), g("angular_velocity"), t);
  };

  const warn = onWarn ?? (() => {});
  const ctl = createController({
    meta,
    onWarn: (axis) => warn(`no actuator for ${axis} on this design`),
  });

  const limits = Object.fromEntries(meta.inputs.map((f) => [f.name, f]));
  const clampU = (u) => {
    for (const [k, v] of Object.entries(u)) {
      const lim = limits[k];
      if (lim) u[k] = Math.min(lim.max, Math.max(lim.min, v));
    }
    return u;
  };

  // --- scene ------------------------------------------------------------------
  const { renderer, scene, camera, sun } = makeScene(canvas, {
    fov: 55, shadowSpan: 30 });
  const env = makeEnvironment(THREE, scene, { meta });
  renderer.setClearColor(env.clearColor, 1);
  const SEABED = meta.seabed_z ?? -30;

  // scene → planet is an exact rigid transform (p = anchor + R_ps·p_scene),
  // so the HUD lat/long is the craft's true planet coordinate — it wraps
  // correctly if you really drive around the Earth.
  const scA = meta.scene.anchor_planet, scR = meta.scene.R_planet_from_scene;
  const latLonOf = (p) => {
    const px = scA[0] + scR[0][0] * p.x + scR[0][1] * p.y + scR[0][2] * p.z;
    const py = scA[1] + scR[1][0] * p.x + scR[1][1] * p.y + scR[1][2] * p.z;
    const pz = scA[2] + scR[2][0] * p.x + scR[2][1] * p.y + scR[2][2] * p.z;
    return { lat: Math.atan2(pz, Math.hypot(px, py)) * 180 / Math.PI,
             lon: Math.atan2(py, px) * 180 / Math.PI };
  };

  const craft = makeCraftMesh(THREE, design);
  scene.add(craft.group);

  // the auto-mode setpoint: a translucent axis triplet (x red, y green,
  // z blue) at the position setpoint, oriented like the attitude setpoint
  const spMarker = new THREE.Group();
  {
    const mk = (dx, dy, dz, color) => {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.03, Math.abs(dx)),
                              Math.max(0.03, Math.abs(dy)),
                              Math.max(0.03, Math.abs(dz))),
        new THREE.MeshBasicMaterial({ color, transparent: true,
                                      opacity: 0.55, depthWrite: false }));
      m.position.set(dx / 2, dy / 2, dz / 2);
      spMarker.add(m);
    };
    mk(1.4, 0, 0, 0xff5f5f); mk(0, 1.4, 0, 0x62d97c); mk(0, 0, 1.4, 0x5fa8ff);
    spMarker.visible = false;
    scene.add(spMarker);
  }

  // light payload: rides the ui module's payload port (frontend-only)
  const uiIdx = design.spine.findIndex((m) => m.type === "ui");
  let light = null;
  if (uiIdx >= 0)
    light = env.makeLight(THREE, craft.group, layout(design).centers[uiIdx]);

  // --- thruster bubbles --------------------------------------------------------
  // Pooled points in WORLD (scene) space, streaming out of the prop wash;
  // short-lived and faint — a hint of a trail, not a smoke column.
  const BUB_N = 500, BUB_LIFE = 0.55;
  const bubPos = new Float32Array(BUB_N * 3);
  const bubCol = new Float32Array(BUB_N * 4);
  const bubVel = new Float32Array(BUB_N * 3);
  const bubAge = new Float32Array(BUB_N).fill(BUB_LIFE);
  const bubGeo = new THREE.BufferGeometry();
  bubGeo.setAttribute("position", new THREE.BufferAttribute(bubPos, 3));
  bubGeo.setAttribute("color", new THREE.BufferAttribute(bubCol, 4));
  const bubCanvas = document.createElement("canvas");
  bubCanvas.width = bubCanvas.height = 32;
  {
    const g = bubCanvas.getContext("2d");
    const grad = g.createRadialGradient(16, 16, 2, 16, 16, 15);
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(0.6, "rgba(255,255,255,.55)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 32, 32);
  }
  const bubMat = new THREE.PointsMaterial({
    size: 0.045, map: new THREE.CanvasTexture(bubCanvas), vertexColors: true,
    transparent: true, depthWrite: false });
  const bubbles = new THREE.Points(bubGeo, bubMat);
  bubbles.frustumCulled = false;
  scene.add(bubbles);
  let bubHead = 0;
  const _bp = new THREE.Vector3(), _bd = new THREE.Vector3();
  for (const s of craft.spinners) s.debt = 0;

  function updateBubbles(dt) {
    for (const s of craft.spinners) {
      const name = `${C}.${s.input}`;
      const lim = limits[name];
      const u = uHeld[name] ?? 0;
      const frac = lim ? Math.abs(u) / lim.max : 0;
      if (frac < 0.04) { s.debt = 0; continue; }
      s.debt += dt * (6 + 55 * frac);
      if (s.debt < 1) continue;
      s.mesh.getWorldPosition(_bp);
      const sgn = Math.sign(u);
      _bd.set(-s.axis[0] * sgn, -s.axis[1] * sgn, -s.axis[2] * sgn)
        .applyQuaternion(craft.group.quaternion);
      const speed = 0.6 + 1.6 * frac;
      while (s.debt >= 1) {
        s.debt -= 1;
        const i = bubHead;
        bubHead = (bubHead + 1) % BUB_N;
        bubAge[i] = 0;
        bubPos[i * 3] = _bp.x + (Math.random() - 0.5) * 0.03;
        bubPos[i * 3 + 1] = _bp.y + (Math.random() - 0.5) * 0.03;
        bubPos[i * 3 + 2] = _bp.z + (Math.random() - 0.5) * 0.03;
        bubVel[i * 3] = _bd.x * speed + (Math.random() - 0.5) * 0.25;
        bubVel[i * 3 + 1] = _bd.y * speed + (Math.random() - 0.5) * 0.25;
        bubVel[i * 3 + 2] = _bd.z * speed + (Math.random() - 0.5) * 0.25;
      }
    }
    for (let i = 0; i < BUB_N; i++) {
      if (bubAge[i] >= BUB_LIFE) { bubCol[i * 4 + 3] = 0; continue; }
      bubAge[i] += dt;
      const k = Math.exp(-dt / 0.3);
      bubVel[i * 3] *= k;
      bubVel[i * 3 + 1] *= k;
      bubVel[i * 3 + 2] = bubVel[i * 3 + 2] * k + 0.5 * dt;
      bubPos[i * 3] += bubVel[i * 3] * dt;
      bubPos[i * 3 + 1] += bubVel[i * 3 + 1] * dt;
      bubPos[i * 3 + 2] += bubVel[i * 3 + 2] * dt;
      const a = Math.max(0, 1 - bubAge[i] / BUB_LIFE);
      bubCol[i * 4] = 0.72;
      bubCol[i * 4 + 1] = 0.86;
      bubCol[i * 4 + 2] = 0.95;
      bubCol[i * 4 + 3] = 0.42 * a;
    }
    bubGeo.attributes.position.needsUpdate = true;
    bubGeo.attributes.color.needsUpdate = true;
  }

  // --- seabed dust ------------------------------------------------------------
  // Brown, translucent plumes kicked up when a thruster washes the bottom or
  // the hull scrapes it. Unlike bubbles they don't rise: a small initial
  // shove away from the wash, then they hang and settle very slowly.
  const DUST_N = 600, DUST_LIFE = 3.2;
  const dustPos = new Float32Array(DUST_N * 3);
  const dustCol = new Float32Array(DUST_N * 4);
  const dustVel = new Float32Array(DUST_N * 3);
  const dustAge = new Float32Array(DUST_N).fill(DUST_LIFE);
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  dustGeo.setAttribute("color", new THREE.BufferAttribute(dustCol, 4));
  const dustMat = new THREE.PointsMaterial({
    size: 0.22, map: new THREE.CanvasTexture(bubCanvas), vertexColors: true,
    transparent: true, depthWrite: false });
  const dust = new THREE.Points(dustGeo, dustMat);
  dust.frustumCulled = false;
  scene.add(dust);
  let dustHead = 0, scrapeDebt = 0;

  function spawnDust(x, y, z, vx, vy, vz) {
    const i = dustHead;
    dustHead = (dustHead + 1) % DUST_N;
    dustAge[i] = 0;
    dustPos[i * 3] = x + (Math.random() - 0.5) * 0.25;
    dustPos[i * 3 + 1] = y + (Math.random() - 0.5) * 0.25;
    dustPos[i * 3 + 2] = z + Math.random() * 0.12;
    dustVel[i * 3] = vx + (Math.random() - 0.5) * 0.3;
    dustVel[i * 3 + 1] = vy + (Math.random() - 0.5) * 0.3;
    dustVel[i * 3 + 2] = vz + Math.random() * 0.1;
  }

  function updateDust(dt) {
    // thruster wash near the bottom stirs a plume at the bed below it
    for (const s of craft.spinners) {
      const name = `${C}.${s.input}`;
      const lim = limits[name];
      const u = uHeld[name] ?? 0;
      const frac = lim ? Math.abs(u) / lim.max : 0;
      if (frac < 0.12) { s.dustDebt = 0; continue; }
      s.mesh.getWorldPosition(_bp);
      const bedZ = env.bedHeight(_bp.x, _bp.y);
      const gap = _bp.z - bedZ;
      if (gap > 1.6 || gap < -0.5) { s.dustDebt = 0; continue; }
      const sgn = Math.sign(u) || 1;
      _bd.set(-s.axis[0] * sgn, -s.axis[1] * sgn, -s.axis[2] * sgn)
        .applyQuaternion(craft.group.quaternion);
      s.dustDebt = (s.dustDebt ?? 0)
        + dt * 30 * frac * (1 - Math.max(0, gap) / 1.6);
      const hl = Math.hypot(_bd.x, _bd.y);
      const hx = hl > 0.05 ? _bd.x / hl : Math.cos(Math.random() * 6.28);
      const hy = hl > 0.05 ? _bd.y / hl : Math.sin(Math.random() * 6.28);
      const push = 0.3 + 0.7 * frac;
      while (s.dustDebt >= 1) {
        s.dustDebt -= 1;
        spawnDust(_bp.x + _bd.x * Math.max(0.1, gap) * 0.6,
                  _bp.y + _bd.y * Math.max(0.1, gap) * 0.6,
                  bedZ + 0.05, hx * push, hy * push, 0.05);
      }
    }
    // hull scraping along the (flat, physics) floor
    const spd = Math.hypot(curRel.v[0], curRel.v[1], curRel.v[2]);
    if (craft.group.position.z - 0.25 < SEABED + 0.2 && spd > 0.5) {
      scrapeDebt += dt * 22 * Math.min(2.5, spd);
      while (scrapeDebt >= 1) {
        scrapeDebt -= 1;
        const px = craft.group.position.x + (Math.random() - 0.5) * 0.8;
        const py = craft.group.position.y + (Math.random() - 0.5) * 0.8;
        spawnDust(px, py, env.bedHeight(px, py) + 0.05,
                  curRel.v[0] * 0.3, curRel.v[1] * 0.3, 0.06);
      }
    } else scrapeDebt = 0;

    for (let i = 0; i < DUST_N; i++) {
      if (dustAge[i] >= DUST_LIFE) { dustCol[i * 4 + 3] = 0; continue; }
      dustAge[i] += dt;
      const k = Math.exp(-dt / 0.7);           // heavy water drag
      dustVel[i * 3] *= k;
      dustVel[i * 3 + 1] *= k;
      dustVel[i * 3 + 2] = dustVel[i * 3 + 2] * k - 0.02 * dt; // slow settle
      dustPos[i * 3] += dustVel[i * 3] * dt;
      dustPos[i * 3 + 1] += dustVel[i * 3 + 1] * dt;
      dustPos[i * 3 + 2] += dustVel[i * 3 + 2] * dt;
      if (dustVel[i * 3 + 2] < 0) {            // rest on the visual bed
        const floor = env.bedHeight(dustPos[i * 3], dustPos[i * 3 + 1]) + 0.03;
        if (dustPos[i * 3 + 2] < floor) {
          dustPos[i * 3 + 2] = floor;
          dustVel[i * 3 + 2] = 0;
        }
      }
      const a = Math.min(1, dustAge[i] * 3) * (1 - dustAge[i] / DUST_LIFE);
      dustCol[i * 4] = 0.55;
      dustCol[i * 4 + 1] = 0.47;
      dustCol[i * 4 + 2] = 0.35;
      dustCol[i * 4 + 3] = 0.28 * a;
    }
    dustGeo.attributes.position.needsUpdate = true;
    dustGeo.attributes.color.needsUpdate = true;
  }

  // --- keyboard → axis demands --------------------------------------------------
  // W/S pitch down/up · A/D yaw left/right · Q/E roll · ↑/↓ surge ·
  // →/← sway · space/shift heave. Codes, not chars (arrows/shift).
  const down = new Set();
  const CODES = new Set(["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
                         "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                         "Space", "ShiftLeft", "ShiftRight"]);
  const onKeyDown = (e) => {
    if (e.code === "KeyL") { toggleLight(); e.preventDefault(); return; }
    if (e.code === "KeyM") { api.cycleMode(); e.preventDefault(); return; }
    if (CODES.has(e.code)) { down.add(e.code); e.preventDefault(); }
  };
  const onKeyUp = (e) => down.delete(e.code);
  addEventListener("keydown", onKeyDown);
  addEventListener("keyup", onKeyUp);
  const held = (c) => (down.has(c) ? 1 : 0);
  // dev hook: /mako/?load=<hash>&hold=w^ pre-holds keys
  // (letters, "_"=space, "%"=shift, "^v<>"=arrows)
  const HOLDMAP = { _: "Space", "%": "ShiftLeft", "^": "ArrowUp",
                    v: "ArrowDown", "<": "ArrowLeft", ">": "ArrowRight" };
  for (const ch of new URLSearchParams(location.search).get("hold") ?? "")
    down.add(HOLDMAP[ch] ?? `Key${ch.toUpperCase()}`);

  const axisDemands = () => ({
    pitch: held("KeyW") - held("KeyS"),         // W = pitch DOWN (+θ)
    yaw: held("KeyA") - held("KeyD"),
    roll: held("KeyE") - held("KeyQ"),
    surge: held("ArrowUp") - held("ArrowDown"),
    sway: held("ArrowLeft") - held("ArrowRight"),   // +y is port
    heave: held("Space") - (held("ShiftLeft") || held("ShiftRight")),
  });

  function toggleLight() {
    if (!light) { warn("no light payload — add a UI module"); return; }
    api.lightOn = light.toggle();
  }

  // --- orbit camera ---------------------------------------------------------------
  const startRel = toRel(sim.state, 0);
  // orbit offset in the CRAFT frame (az = π → dead astern): the camera
  // follows the hull, so it turns, rolls and pitches with the mako — but it
  // rides a LOW-PASSED copy of the pose (world frame), so wave rocking and
  // control jitter don't shake the view. Drags stay instant (the offset is
  // applied after the filter).
  const cam = { az: Math.PI, el: 0.32, dist: 4.6 };
  const CAM_TAU_P = 0.1, CAM_TAU_Q = 0.3;      // position/orientation lags, s
  const camPos = new THREE.Vector3(startRel.p[0], startRel.p[1],
                                   startRel.p[2]);
  const camQ = new THREE.Quaternion(startRel.q[1], startRel.q[2],
                                    startRel.q[3], startRel.q[0]);
  let dragging = false, px = 0, py = 0;
  const onPointerDown = (e) => {
    dragging = true; px = e.clientX; py = e.clientY;
    canvas.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!dragging) return;
    cam.az -= (e.clientX - px) * 0.008;
    cam.el = Math.min(1.35, Math.max(-1.35, cam.el + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
  };
  const onPointerUp = () => { dragging = false; };
  const onWheel = (e) => {
    cam.dist = Math.min(16, Math.max(2.0, cam.dist * Math.exp(e.deltaY * 0.001)));
    e.preventDefault();
  };
  canvas.addEventListener("pointerdown", onPointerDown);
  addEventListener("pointermove", onPointerMove);
  addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  // --- navball --------------------------------------------------------------------
  const navball = navballCanvas ? createNavball(THREE, navballCanvas) : null;

  // --- loop -------------------------------------------------------------------------
  let curRel = startRel, prevRel = startRel;
  let uHeld = {};
  let acc = 0, last = performance.now(), raf = 0, disposed = false;

  function controlFrame() {
    const u = ctl.update(curRel, axisDemands(), FRAME);
    // thrusters only bite in the water: each prop's command fades to zero
    // over its last ~10 cm of submergence (η is the same wave the sim
    // runs). Smoothstep + a short low-pass on the gate — a hard edge
    // toggling as waves sweep the prop pumps the very oscillation that
    // put it at the surface. A breached prop spins free — no force, no
    // bubbles, no dust.
    for (const s of craft.spinners) {
      const name = `${C}.${s.input}`;
      s.mesh.getWorldPosition(_bp);
      const x = Math.min(1, Math.max(0,
        (env.surfaceHeight(_bp.x, _bp.y, sim.t) - _bp.z) / 0.10));
      const target = x * x * (3 - 2 * x);
      s.gate = (s.gate ?? 1) + (target - (s.gate ?? 1)) * GATE_ALPHA;
      if (s.gate < 0.999 && u[name]) u[name] *= s.gate;
    }
    uHeld = clampU(u);
    for (let i = 0; i < SUBSTEPS; i++) sim.step(DT, uHeld);
    prevRel = curRel;
    curRel = toRel(sim.state, sim.t);
  }

  const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
  const _camOff = new THREE.Vector3();
  const toQ = (q, out) => out.set(q[1], q[2], q[3], q[0]);

  function frame(now) {
    if (disposed) return;
    const rdt = Math.min((now - last) / 1000, 0.05); last = now; acc += rdt;
    while (acc >= FRAME) { controlFrame(); acc -= FRAME; }
    const a = Math.min(acc / FRAME, 1);

    // craft pose, interpolated in scene coordinates
    craft.group.position.set(lerp(prevRel.p[0], curRel.p[0], a),
                             lerp(prevRel.p[1], curRel.p[1], a),
                             lerp(prevRel.p[2], curRel.p[2], a));
    craft.group.quaternion.copy(
      toQ(prevRel.q, _qa).slerp(toQ(curRel.q, _qb), a));
    const tp = craft.group.position;

    // props: spin with the commanded throttle, signed
    for (const s of craft.spinners) {
      const name = `${C}.${s.input}`;
      const lim = limits[name];
      const u = uHeld[name] ?? 0;
      const frac = lim ? Math.abs(u) / lim.max : 0;
      s.mesh.rotation.z += (Math.sign(u) || 1) * (0.4 + 55 * frac) * rdt;
    }
    // fins track the live surface state (each sits in a hinge pivot)
    for (const f of craft.fins) {
      if (sim.slotMap[f.slot])
        f.pivot.rotation[f.rotAxis] = f.sign * sim.slot(f.slot)[0];
    }
    updateBubbles(rdt);
    updateDust(rdt);

    // auto-mode setpoint triplet
    const sp = ctl.setpoints;
    if (ctl.mode === "auto") {
      spMarker.visible = true;
      spMarker.position.set(...sp.pos);
      spMarker.quaternion.copy(toQ(sp.q, _qa));
    } else spMarker.visible = false;

    // orbit camera around the craft, offset held in the BODY frame of the
    // low-passed pose (small pose wiggles don't reach the camera)
    camPos.lerp(tp, 1 - Math.exp(-rdt / CAM_TAU_P));
    camQ.slerp(craft.group.quaternion, 1 - Math.exp(-rdt / CAM_TAU_Q));
    const ce = Math.cos(cam.el), se = Math.sin(cam.el);
    _camOff.set(cam.dist * ce * Math.cos(cam.az),
                cam.dist * ce * Math.sin(cam.az),
                cam.dist * se).applyQuaternion(camQ);
    camera.up.set(0, 0, 1).applyQuaternion(camQ);
    camera.position.set(camPos.x + _camOff.x, camPos.y + _camOff.y,
                        camPos.z + _camOff.z);
    camera.lookAt(camPos.x, camPos.y, camPos.z);
    sun.target.position.copy(tp); sun.target.updateMatrixWorld();
    env.update(rdt, sim.t, tp.x, tp.y, camera.position);

    if (navball)
      navball.update(curRel.q, ctl.mode === "manual" ? null : sp.q);

    if (setHud) {
      const v = curRel.v;
      const rpy = rpyOf(curRel.q);
      setHud({
        mode: ctl.mode,
        depth: -tp.z,
        ...latLonOf(tp),
        spd: Math.hypot(v[0], v[1], v[2]),
        hdg: ((-rpy.yaw * 180 / Math.PI) % 360 + 360) % 360,
        rpy,
        rpySp: ctl.mode === "manual"
          ? null : { roll: sp.roll, pitch: sp.pitch, yaw: sp.yaw },
        light: light ? (api.lightOn ?? false) : null,
      });
    }

    fitToCanvas(renderer, camera, canvas);
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  const api = {
    lightOn: false,
    get mode() { return ctl.mode; },
    setMode(m) {
      ctl.setMode(m);
      if (m !== "manual") ctl.syncSetpoints(curRel);
      return m;
    },
    cycleMode() {
      const order = ["manual", "ori", "auto"];
      return api.setMode(order[(order.indexOf(ctl.mode) + 1) % order.length]);
    },
    setVisibility(v) { renderer.setClearColor(env.setVisibility(v), 1); },
    toggleLight,
    get hasLight() { return !!light; },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      removeEventListener("keydown", onKeyDown);
      removeEventListener("keyup", onKeyUp);
      removeEventListener("pointermove", onPointerMove);
      removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("wheel", onWheel);
      navball?.dispose();
      bubGeo.dispose();
      bubMat.dispose();
      dustGeo.dispose();
      dustMat.dispose();
      env.dispose();
      renderer.dispose();
    },
  };
  return api;
}
