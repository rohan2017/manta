// Live airplane demo for the manta homepage.
//
// Runs the SAME jointed-airframe model as `examples/vehicles/airplane.py`,
// compiled to WebAssembly by manta's `TargetWasm` backend, stepping the real
// rigid-body + aero + contact physics in the browser. A custom Three.js scene
// reconstructs the airframe and poses it each frame from the live sim state —
// no rerun, no server. An autopilot flies a scripted circuit on a loop; the
// keyboard (W/A/S/D · Q/E · X/Z) adds live control on top.
//
// Frame convention matches manta's WorldFrame: x forward, y left, z up, metres.
// Quaternions are [w, x, y, z]; Three wants [x, y, z, w].

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { load, descriptor } from "./airplane.js";

// --- airframe geometry (mirrors airplane.py; m) ---------------------------
const D2R = Math.PI / 180;
const WING_INC = 10 * D2R;
const AIL_HINGE = 0.225, AIL_SPAN = 1.0;
const TAIL_X = -1.2, ELEV_X = -1.32, FIN_Z = 0.12;
const SURF_OFF = [-0.05, 0, 0];
const CG_Z = -0.18;
const GEAR = [[0.4, 0, -0.3], [-0.1, 0.25, -0.3], [-0.1, -0.25, -0.3]];

// --- control limits + servo (mirrors airplane.py) --------------------------
const MAX_AIL = 8 * D2R, MAX_ELEV = 20 * D2R, MAX_RUD = 20 * D2R;
const THR_MAX = 1.0, THR_RATE = 0.4, SERVO_KP = 20.0, AIL_TRIM = 0.13 * D2R;
const DT = 0.002, FRAME = 0.02, SUBSTEPS = Math.round(FRAME / DT);

// Scripted circuit: [t0, t1, {keys held}] — the autopilot baseline. Same
// routine as the Python example: take off, cruise, bank right, level, glide in.
const SCRIPT = [
  [0.5, 3.0, "x"],     // full throttle → takeoff roll + climb
  [9.0, 10.6, "z"],    // back to cruise power
  [12.5, 12.9, "e"],   // roll right into a bank
  [15.5, 15.8, "q"],   // roll back wings-level
  [18.0, 21.0, "z"],   // throttle to zero → glide in
];
const LOOP_T = 30.0;   // reset and replay the circuit after this many seconds

// --- palette (matches the site) --------------------------------------------
const C = {
  airframe: 0xc6d3d0, surface: 0xf5b461, fin: 0xa9bbb7,
  gear: 0x2a3133, ground: 0x0c1413, runway: 0x152021,
  centerline: 0x1c8478, marker: 0x143034, prop: 0x5eead4,
};

// State-slot offsets, read from the descriptor (single source of truth).
const SLOT = Object.fromEntries(
  descriptor.stateSlots.map((s) => [s.name.replace("plane.", ""), s.off]));

const SURFACES = ["ail_l", "ail_r", "elev", "rud"];
const lerp = (a, b, k) => a + (b - a) * k;

function quatWXYZtoThree(q, off) {
  // manta [w,x,y,z] at state offset `off` → THREE.Quaternion(x,y,z,w)
  return new THREE.Quaternion(q[off + 1], q[off + 2], q[off + 3], q[off]);
}

// A box given half-extents + center in its parent frame (rerun convention).
function box(parent, half, center, color, opts = {}) {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(2 * half[0], 2 * half[1], 2 * half[2]),
    new THREE.MeshStandardMaterial({
      color, metalness: 0.1, roughness: 0.65, ...opts,
    }));
  m.position.set(center[0], center[1], center[2]);
  m.castShadow = true; m.receiveShadow = true;
  parent.add(m);
  return m;
}

function buildAirframe() {
  const plane = new THREE.Group();

  box(plane, [0.95, 0.035, 0.035], [-0.45, 0, 0], C.airframe);        // fuselage
  // nose cone
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.05, 0.16, 16),
    new THREE.MeshStandardMaterial({ color: C.airframe, roughness: 0.6 }));
  nose.rotation.z = -Math.PI / 2; nose.position.set(0.58, 0, 0);
  plane.add(nose);

  // main wing, mounted at +10° incidence
  const wing = new THREE.Group();
  wing.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -WING_INC);
  box(wing, [0.15, 1.2, 0.008], [-0.075, 0, 0], C.airframe);
  plane.add(wing);

  box(plane, [0.075, 0.3, 0.006], [TAIL_X - 0.0375, 0, 0], C.airframe);  // tail
  box(plane, [0.075, 0.005, 0.12], [TAIL_X - 0.0375, 0, FIN_Z], C.fin);  // fin

  // control surfaces on hinge groups (deflect each frame)
  const surf = {};
  const mkHinge = (name, pos, axis, half) => {
    const g = new THREE.Group();
    g.position.set(pos[0], pos[1], pos[2]);
    box(g, half, SURF_OFF, C.surface);
    plane.add(g);
    surf[name] = { group: g, axis: new THREE.Vector3(...axis) };
  };
  mkHinge("ail_l", [-AIL_HINGE, AIL_SPAN, 0], [0, 1, 0], [0.05, 0.2, 0.004]);
  mkHinge("ail_r", [-AIL_HINGE, -AIL_SPAN, 0], [0, 1, 0], [0.05, 0.2, 0.004]);
  mkHinge("elev", [ELEV_X, 0, 0], [0, 1, 0], [0.04, 0.3, 0.004]);
  mkHinge("rud", [ELEV_X, 0, FIN_Z], [0, 0, 1], [0.04, 0.003, 0.1]);

  for (const g of GEAR) {
    const s = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 12, 10),
      new THREE.MeshStandardMaterial({ color: C.gear, roughness: 0.5 }));
    s.position.set(g[0], g[1], g[2]); plane.add(s);
  }

  // spinning propeller disc at the nose
  const prop = new THREE.Mesh(
    new THREE.RingGeometry(0.04, 0.34, 24),
    new THREE.MeshBasicMaterial({
      color: C.prop, transparent: true, opacity: 0.32,
      side: THREE.DoubleSide }));
  prop.rotation.y = Math.PI / 2; prop.position.set(0.5, 0, CG_Z);
  plane.add(prop);

  return { plane, wing, surf, prop };
}

function buildScene(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x080b0c, 0.006);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1200);
  camera.up.set(0, 0, 1);                       // z-up world

  scene.add(new THREE.HemisphereLight(0x9fb8c8, 0x0a1010, 0.9));
  const sun = new THREE.DirectionalLight(0xeaf6f2, 1.4);
  sun.position.set(40, 60, 120);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 400;
  const sc = sun.shadow.camera;
  sc.left = -60; sc.right = 60; sc.top = 60; sc.bottom = -60;
  scene.add(sun);

  // ground + runway + centerline dashes + distance markers (so motion reads)
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(2400, 2400),
    new THREE.MeshStandardMaterial({ color: C.ground, roughness: 1 }));
  ground.receiveShadow = true; scene.add(ground);
  const runway = new THREE.Mesh(
    new THREE.PlaneGeometry(260, 6),
    new THREE.MeshStandardMaterial({ color: C.runway, roughness: 1 }));
  runway.position.set(95, 0, 0.01); scene.add(runway);
  for (let x = -20; x < 200; x += 8) {
    const dash = new THREE.Mesh(
      new THREE.PlaneGeometry(3, 0.18),
      new THREE.MeshBasicMaterial({ color: C.centerline }));
    dash.position.set(x, 0, 0.02); scene.add(dash);
  }
  for (let x = 0; x < 600; x += 30) {
    for (const sy of [-14, 14]) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.6, 0.6),
        new THREE.MeshStandardMaterial({ color: C.marker, roughness: 1 }));
      m.position.set(x, sy, 0.3); m.castShadow = true; scene.add(m);
    }
  }

  const air = buildAirframe();
  scene.add(air.plane);
  return { renderer, scene, camera, air };
}

// Chase camera, exponentially smoothed toward 7.5 m behind the plane along
// its heading, 2.5 m up. `k` is a frame-rate-independent smoothing factor.
function chaseCamera(camera, pos, yaw, k) {
  const eye = new THREE.Vector3(
    pos[0] - 7.5 * Math.cos(yaw), pos[1] - 7.5 * Math.sin(yaw), pos[2] + 2.5);
  camera.position.lerp(eye, k);
  const look = new THREE.Vector3(
    pos[0] + 3 * Math.cos(yaw), pos[1] + 3 * Math.sin(yaw), pos[2] + 0.3);
  camera._look = camera._look ? camera._look.lerp(look, k) : look.clone();
  camera.lookAt(camera._look);
}

export async function start(canvas, hud) {
  const rt = await load();
  const sim = rt.sim();
  const { renderer, scene, camera, air } = buildScene(canvas);

  // Mode: "auto" (scripted circuit, loops) until the user clicks to take
  // over → "manual" (keyboard only, no auto-reset). Scrolling the demo out
  // of view (handled by the page) or reloading returns to "auto".
  let mode = "auto";
  const KEYS = new Set("wasdqexz".split(""));
  const down = new Set();

  canvas.addEventListener("pointerdown", () => { mode = "manual"; });
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (mode === "manual" && KEYS.has(k)) { down.add(k); e.preventDefault(); }
  });
  addEventListener("keyup", (e) => down.delete(e.key.toLowerCase()));

  let t = 0, throttle = 0, acc = 0, last = performance.now();
  let prev = Float64Array.from(sim.state);     // pose at the last frame edge

  function reset() {
    sim.state = Float64Array.from(descriptor.initialState);
    sim.t = 0; t = 0; throttle = 0;
    prev = Float64Array.from(sim.state);        // no interpolation across reset
  }
  // The page calls this when the demo scrolls out of view: hand control back
  // to the autopilot and start the circuit fresh.
  function toAuto() { if (mode !== "auto") { mode = "auto"; down.clear(); }
    reset(); }

  // In auto mode the script drives; in manual only the live keyboard does.
  const held = (k) => mode === "manual"
    ? (down.has(k) ? 1 : 0)
    : (SCRIPT.some(([a, b, key]) => key === k && t >= a && t < b) ? 1 : 0);

  function controlFrame() {
    if (mode === "auto" && t >= LOOP_T) reset();
    throttle = Math.min(Math.max(
      throttle + (held("x") - held("z")) * THR_RATE * FRAME, 0), THR_MAX);
    const ail = MAX_AIL * (held("q") - held("e")) - AIL_TRIM * throttle;
    const target = {
      ail_l: ail, ail_r: -ail,
      elev: MAX_ELEV * (held("s") - held("w")),
      rud: MAX_RUD * (held("d") - held("a")),
    };
    for (let i = 0; i < SUBSTEPS; i++) {
      const u = { "prop.throttle": throttle };
      for (const name of SURFACES) {
        const angle = sim.state[SLOT[`${name}.angle`]];
        u[`${name}.torque_cmd`] = SERVO_KP * (target[name] - angle);
      }
      sim.step(DT, u);
      t += DT;
    }
  }

  // Render the airframe at the interpolated pose between the two most recent
  // frame edges (`prev`→`curr`, fraction `a`) — this is what removes the
  // jitter: the eye sees continuous motion, not 50 Hz physics steps. Returns
  // the interpolated position + heading for the camera.
  function applyPose(curr, a) {
    const P = SLOT.position, O = SLOT.orientation;
    const pos = [lerp(prev[P], curr[P], a), lerp(prev[P + 1], curr[P + 1], a),
                 lerp(prev[P + 2], curr[P + 2], a)];
    air.plane.position.set(pos[0], pos[1], pos[2]);
    const q = quatWXYZtoThree(prev, O).slerp(quatWXYZtoThree(curr, O), a);
    air.plane.quaternion.copy(q);
    for (const name of SURFACES) {
      const off = SLOT[`${name}.angle`];
      air.surf[name].group.quaternion.setFromAxisAngle(
        air.surf[name].axis, lerp(prev[off], curr[off], a));
    }
    const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y),
                           1 - 2 * (q.y * q.y + q.z * q.z));
    return { pos, yaw };
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }

  function frame(now) {
    const rdt = Math.min((now - last) / 1000, 0.05); last = now;
    acc += rdt;
    // Fixed-timestep physics with leftover accumulator for interpolation.
    while (acc >= FRAME) {
      prev = Float64Array.from(sim.state);
      controlFrame();
      acc -= FRAME;
    }
    const a = Math.min(acc / FRAME, 1);

    const { pos, yaw } = applyPose(sim.state, a);
    air.prop.rotation.x += (0.3 + throttle * 2.2);            // spin ∝ throttle
    chaseCamera(camera, pos, yaw, 1 - Math.exp(-rdt / 0.10)); // dt-independent

    resize();
    renderer.render(scene, camera);

    if (hud) {
      const s = sim.state;
      const v = Math.hypot(s[SLOT.velocity], s[SLOT.velocity + 1],
                           s[SLOT.velocity + 2]);
      hud.textContent =
        `alt ${pos[2].toFixed(1)} m   spd ${v.toFixed(1)} m/s   ` +
        `thr ${(throttle * 100).toFixed(0)}%   ` +
        (mode === "manual" ? "MANUAL" : "AUTOPILOT");
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return { toAuto };
}
