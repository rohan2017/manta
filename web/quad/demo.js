// Live quadcopter for the homepage carousel.
//
// Runs the SAME model as web/demos.py:build_quad, compiled to WebAssembly by
// manta's TargetWasm, stepped live in the browser. The quad is open-loop
// unstable, so it's flown by manta's own LQR — the constant gain `K` solved
// for this exact model (see web/build_demos.py / the LQR transform) is baked
// in here and applied as u = u_ref − K·δx about a moving position setpoint.
// An autopilot tours a gentle box on a loop; click to take the controls.

import {
  THREE, C, lerp, quatWXYZ, box, wireBox, arrow, makeScene, ground, fitToCanvas,
} from "../lib/render.js";
import { load, descriptor } from "./quad.js";

// --- model constants (mirror web/demos.py) ---------------------------------
const ROTORS = ["front", "right", "back", "left"];
const ARM = 0.25, MAXT = 6.0, MASS = 1.0, G = 9.81;
const HOVER = MASS * G / 4 / MAXT;                 // 0.40875
const MOUNT = { front: [+ARM, 0, 0], right: [0, +ARM, 0],
                back: [-ARM, 0, 0], left: [0, -ARM, 0] };
const SPIN = { front: +1, right: -1, back: +1, left: -1 };

// LQR gain for this model: rows [front,right,back,left], columns the tangent
// error δx = [pos−setpoint(3), 2·vec(q)(3), velocity(3), angular_velocity(3)].
const K = [
  [-0.8507, 0, 1.6455, 0, -1.5927, 0.4078, -0.6761, 0, 0.6896, 0, -0.1808, 0.2872],
  [0, -0.8507, 1.6455, 1.5927, 0, -0.4078, 0, -0.6761, 0.6896, 0.1808, 0, -0.2872],
  [0.8507, 0, 1.6455, 0, 1.5927, 0.4078, 0.6761, 0, 0.6896, 0, 0.1808, 0.2872],
  [0, 0.8507, 1.6455, -1.5927, 0, -0.4078, 0, 0.6761, 0.6896, -0.1808, 0, -0.2872],
];
const THR_LIM = 1.5;                               // per-rotor clip (T/W head-room)

// --- timing ----------------------------------------------------------------
const FRAME = 0.02, DT = 0.004, SUBSTEPS = Math.round(FRAME / DT);
const SLOT = Object.fromEntries(
  descriptor.stateSlots.map((s) => [s.name.replace("quad.", ""), s.off]));

// Gentle autopilot tour: a loop of setpoints in a small box around hover.
const TOUR = [
  [1.4, 0, 2.4], [1.4, 1.4, 2.0], [-1.4, 1.4, 2.6],
  [-1.4, -1.4, 2.0], [1.4, -1.4, 2.6], [0, 0, 2.2],
];

function buildQuad(scene) {
  const root = new THREE.Group();
  // structural skeleton (stands in for the future 3D model)
  box(root, [0.16, 0.16, 0.07], [0, 0, 0], C.frame);          // hub
  const props = {};
  for (const r of ROTORS) {
    const m = MOUNT[r];
    // motor nacelle (the Thruster mount) + spinning prop disc
    box(root, [0.05, 0.05, 0.05], [m[0], m[1], 0.02], C.accent);
    const prop = new THREE.Mesh(
      new THREE.RingGeometry(0.04, 0.12, 28),
      new THREE.MeshBasicMaterial({ color: C.blue, transparent: true,
        opacity: 0.4, side: THREE.DoubleSide }));
    prop.position.set(m[0], m[1], 0.05);
    root.add(prop);
    props[r] = prop;
  }
  scene.add(root);

  // skeleton overlay: drag volume + per-rotor thrust arrows + gravity
  const skel = new THREE.Group();
  wireBox(skel, [0.62, 0.62, 0.16], [0, 0, 0], C.drag, { opacity: 0.7 });
  const thrust = {};
  for (const r of ROTORS) thrust[r] = arrow(skel, C.thrust, { scale: 0.16 });
  skel.visible = false;
  root.add(skel);

  return { root, props, skel, thrust };
}

// δx and the LQR law → four throttles, clipped.
function lqr(state, setpoint) {
  const P = SLOT.position, O = SLOT.orientation, V = SLOT.velocity, W = SLOT.angular_velocity;
  let qw = state[O], qx = state[O + 1], qy = state[O + 2], qz = state[O + 3];
  const s = qw < 0 ? -1 : 1;                                    // sign-normalise
  const dx = [
    state[P] - setpoint[0], state[P + 1] - setpoint[1], state[P + 2] - setpoint[2],
    2 * s * qx, 2 * s * qy, 2 * s * qz,
    state[V], state[V + 1], state[V + 2],
    state[W], state[W + 1], state[W + 2],
  ];
  return ROTORS.map((_, i) => {
    let u = HOVER;
    for (let j = 0; j < 12; j++) u -= K[i][j] * dx[j];
    return Math.min(Math.max(u, 0), THR_LIM);
  });
}

export async function start(canvas, ctx = {}) {
  const sim = (await load()).sim();
  const { renderer, scene, camera, sun } = makeScene(canvas, { fov: 50, shadowSpan: 16 });
  ground(scene, { size: 400, grid: 60, cells: 60 });
  const quad = buildQuad(scene);

  let interactive = false;
  const down = new Set();
  const KEYS = new Set("wasd".split("").concat(["shift", " "]));
  const onKeyDown = (e) => {
    const k = e.key === " " ? " " : e.key.toLowerCase();
    if (interactive && KEYS.has(k)) { down.add(k); e.preventDefault(); }
  };
  const onKeyUp = (e) => down.delete(e.key === " " ? " " : e.key.toLowerCase());
  addEventListener("keydown", onKeyDown); addEventListener("keyup", onKeyUp);

  const setpoint = [0, 0, 2.0];
  let tour = 0, throttle = ROTORS.map(() => HOVER);
  let t = 0, acc = 0, last = performance.now(), camAng = 0.6;
  let prev = Float64Array.from(sim.state);

  function controlFrame() {
    if (interactive) {
      const mv = [ (down.has("w") ? 1 : 0) - (down.has("s") ? 1 : 0),
                   (down.has("a") ? 1 : 0) - (down.has("d") ? 1 : 0),
                   (down.has(" ") ? 1 : 0) - (down.has("shift") ? 1 : 0) ];
      for (let i = 0; i < 3; i++) setpoint[i] += mv[i] * 1.4 * FRAME;
      setpoint[2] = Math.max(0.6, setpoint[2]);
    } else {
      // glide the setpoint toward the active tour waypoint; advance when close
      const tgt = TOUR[tour];
      let near = true;
      for (let i = 0; i < 3; i++) {
        const d = tgt[i] - setpoint[i];
        setpoint[i] += Math.max(-1.2 * FRAME, Math.min(1.2 * FRAME, d));
        if (Math.abs(tgt[i] - setpoint[i]) > 0.05) near = false;
      }
      if (near) tour = (tour + 1) % TOUR.length;
    }
    throttle = lqr(sim.state, setpoint);
    const u = {};
    for (let i = 0; i < ROTORS.length; i++) u[`${ROTORS[i]}.throttle`] = throttle[i];
    for (let i = 0; i < SUBSTEPS; i++) { sim.step(DT, u); t += DT; }
  }

  function applyPose(curr, a) {
    const P = SLOT.position, O = SLOT.orientation;
    const pos = [lerp(prev[P], curr[P], a), lerp(prev[P + 1], curr[P + 1], a),
                 lerp(prev[P + 2], curr[P + 2], a)];
    quad.root.position.set(pos[0], pos[1], pos[2]);
    quad.root.quaternion.copy(quatWXYZ(prev, O).slerp(quatWXYZ(curr, O), a));
    return pos;
  }

  function updateForces() {
    if (!quad.skel.visible) return;
    for (let i = 0; i < ROTORS.length; i++) {
      const r = ROTORS[i], m = MOUNT[r];
      quad.thrust[r].set([m[0], m[1], 0.05], [0, 0, throttle[i] * MAXT]);
    }
  }

  // snap the chase camera to its initial pose
  camera.position.set(3.2 * Math.cos(camAng), 3.2 * Math.sin(camAng), 3.5);
  camera.lookAt(0, 0, 2);

  let raf = 0;
  function frame(now) {
    const rdt = Math.min((now - last) / 1000, 0.05); last = now; acc += rdt;
    while (acc >= FRAME) { prev = Float64Array.from(sim.state); controlFrame(); acc -= FRAME; }
    const a = Math.min(acc / FRAME, 1);
    const pos = applyPose(sim.state, a);

    for (const r of ROTORS) quad.props[r].rotation.z += SPIN[r] * (0.4 + throttle[ROTORS.indexOf(r)] * 2.2);
    updateForces();

    // slow auto-orbit chase camera
    camAng += rdt * 0.18;
    const R = 3.2;
    const eye = new THREE.Vector3(pos[0] + R * Math.cos(camAng), pos[1] + R * Math.sin(camAng), pos[2] + 1.5);
    camera.position.lerp(eye, 1 - Math.exp(-rdt / 0.5));
    camera.lookAt(pos[0], pos[1], pos[2]);
    sun.target.position.set(pos[0], pos[1], pos[2]); sun.target.updateMatrixWorld();

    fitToCanvas(renderer, camera, canvas);
    renderer.render(scene, camera);
    if (ctx.setHud) {
      ctx.setHud(`alt ${pos[2].toFixed(1)} m   ${interactive ? "MANUAL" : "AUTOPILOT"}`);
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  return {
    dispose() {
      cancelAnimationFrame(raf);
      removeEventListener("keydown", onKeyDown); removeEventListener("keyup", onKeyUp);
      renderer.dispose();
    },
    setSkeleton(on) { quad.skel.visible = on; },
    setInteractive(on) { interactive = on; if (!on) down.clear(); },
  };
}
