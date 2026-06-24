// Live Cessna 172 for the homepage carousel.
//
// Runs the SAME airframe as examples/vehicles/airplane.py — a re-measured
// 172 on NACA aero surfaces and real control flaps — compiled to WebAssembly
// by manta's TargetWasm and stepped live. The real 172 takes ~1 km to roll
// off the runway, far too long for a loop, so the demo starts airborne in
// cruise and flies a gentle scripted circuit on trim (exactly how the Python
// example flies it: throttle + the occasional aileron). Click to take over.

import {
  THREE, C, D2R, lerp, quatWXYZ, yawOf, box, arrow, makeScene, ground, fitToCanvas,
} from "../lib/render.js";
import { load, descriptor } from "./airplane.js";

// --- geometry (mirrors airplane.py; metres) --------------------------------
const WING_QC_X = -2.44, WING_Z = 0.85, WING_SPAN = 11.0, WING_CHORD = 1.49;
const WING_INC = 1.5 * D2R, DIHEDRAL = 1.7 * D2R;
const AIL_Y = 3.9, AIL_SPAN = 1.8, AIL_CHORD = 0.46;
const HTAIL_QC_X = -7.05, HTAIL_Z = 0.25, HTAIL_SPAN = 3.4, HTAIL_CHORD = 0.6;
const VTAIL_QC_X = -6.79, VTAIL_Z = 0.9, VTAIL_CHORD = 0.95;
const GEAR = [[-1.10, 0, -1.23], [-2.66, 1.27, -1.23], [-2.66, -1.27, -1.23]];
const FUSELAGE = [
  { c: [-0.50, 0, -0.10], s: [1.00, 0.90, 0.95] },
  { c: [-2.30, 0, 0.10], s: [2.60, 1.05, 1.40] },
  { c: [-5.55, 0, 0.00], s: [3.90, 0.55, 0.75] },
];

// --- control throws (mirror airplane.py) -----------------------------------
const MAX_AIL = 12 * D2R, MAX_ELEV = 14 * D2R, MAX_RUD = 22 * D2R;
const ELEV_TRIM = -4.5 * D2R, AIL_TRIM = 0.28 * D2R, THR_RATE = 0.4;
const ELEV_SIGN = -1.0, RUD_SIGN = 1.0;
const THRUST = 1300.0;

const DT = 0.002, FRAME = 0.02, SUBSTEPS = Math.round(FRAME / DT);
const LOOP_T = 22.0;
const SURFACES = ["ail_l", "ail_r", "elev", "rud"];
const SLOT = Object.fromEntries(
  descriptor.stateSlots.map((s) => [s.name.replace("plane.", ""), s.off]));

// cruise start: airborne, level, ~44 m/s along +x at altitude.
const CRUISE_ALT = 140.0, CRUISE_SPD = 44.0, CRUISE_THR = 0.6;

// scripted circuit on top of cruise trim: [t0, t1, key]
const SCRIPT = [[2.5, 3.4, "q"], [6.0, 6.9, "e"], [12.0, 13.4, "d"], [16.0, 16.6, "a"]];

function buildPlane(scene) {
  const root = new THREE.Group();
  for (const b of FUSELAGE) box(root, b.s, b.c, C.frame);

  // main wing (one box at slight incidence), with dihedral hinted by tilting
  // each half. Drawn in a group so incidence is a single rotation.
  const wing = new THREE.Group();
  wing.position.set(WING_QC_X, 0, WING_Z);
  wing.rotation.y = -WING_INC;
  box(wing, [WING_CHORD, WING_SPAN, 0.05], [0, 0, 0], C.frame);
  root.add(wing);

  // control surfaces on hinge groups (deflect each frame)
  const surf = {};
  const hinge = (name, pos, axis, size) => {
    const g = new THREE.Group();
    g.position.set(pos[0], pos[1], pos[2]);
    box(g, size, [-size[0] / 2, 0, 0], C.surface);          // box sits aft of hinge
    root.add(g);
    surf[name] = { group: g, axis: new THREE.Vector3(...axis) };
  };
  hinge("ail_l", [WING_QC_X - WING_CHORD / 2, +AIL_Y, WING_Z], [0, 1, 0], [AIL_CHORD, AIL_SPAN, 0.04]);
  hinge("ail_r", [WING_QC_X - WING_CHORD / 2, -AIL_Y, WING_Z], [0, 1, 0], [AIL_CHORD, AIL_SPAN, 0.04]);
  hinge("elev", [HTAIL_QC_X, 0, HTAIL_Z], [0, 1, 0], [HTAIL_CHORD, HTAIL_SPAN, 0.05]);
  hinge("rud", [VTAIL_QC_X, 0, VTAIL_Z], [0, 0, 1], [VTAIL_CHORD, 0.05, 1.2]);

  for (const g of GEAR) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 10),
      new THREE.MeshStandardMaterial({ color: C.accent, roughness: 0.5 }));
    s.position.set(g[0], g[1], g[2]); s.castShadow = true; root.add(s);
  }

  // spinning propeller disc at the nose
  const prop = new THREE.Mesh(
    new THREE.RingGeometry(0.12, 0.95, 32),
    new THREE.MeshBasicMaterial({ color: C.blue, transparent: true, opacity: 0.32, side: THREE.DoubleSide }));
  prop.rotation.y = Math.PI / 2; prop.position.set(0.15, 0, 0);
  root.add(prop);

  // skeleton overlay: thrust at the prop (aero forces arrive with the
  // WASM force export); kept a child of root so it rides the airframe.
  const skel = new THREE.Group();
  const thrust = arrow(skel, C.thrust, { scale: 0.004 });
  skel.visible = false; root.add(skel);

  scene.add(root);
  return { root, wing, surf, prop, skel, thrust };
}

export async function start(canvas, ctx = {}) {
  const sim = (await load()).sim();
  const { renderer, scene, camera, sun } = makeScene(canvas, { fov: 55, shadowSpan: 120 });
  ground(scene, { z: 0, size: 8000, grid: 2000, cells: 100 });
  const plane = buildPlane(scene);

  function reset() {
    const s = Float64Array.from(descriptor.initialState);
    s[SLOT.position + 2] = CRUISE_ALT;
    s[SLOT.velocity] = CRUISE_SPD;
    sim.state = s; sim.t = 0; t = 0; throttle = CRUISE_THR;
    prev = Float64Array.from(sim.state);
  }

  let interactive = false, t = 0, throttle = CRUISE_THR;
  let acc = 0, last = performance.now(), prev;
  const down = new Set();
  const KEYS = new Set("wasdqexz".split(""));
  const onKeyDown = (e) => { const k = e.key.toLowerCase(); if (interactive && KEYS.has(k)) { down.add(k); e.preventDefault(); } };
  const onKeyUp = (e) => down.delete(e.key.toLowerCase());
  addEventListener("keydown", onKeyDown); addEventListener("keyup", onKeyUp);
  reset();

  const held = (k) => interactive ? (down.has(k) ? 1 : 0)
    : (SCRIPT.some(([a, b, key]) => key === k && t >= a && t < b) ? 1 : 0);

  function controlFrame() {
    if (!interactive && t >= LOOP_T) reset();
    throttle = Math.min(Math.max(throttle + (held("x") - held("z")) * THR_RATE * FRAME, 0), 1);
    const ail = MAX_AIL * (held("e") - held("q")) + AIL_TRIM * throttle;
    const target = {
      ail_l: +ail, ail_r: -ail,
      elev: ELEV_SIGN * MAX_ELEV * (held("s") - held("w")) + ELEV_TRIM,
      rud: RUD_SIGN * MAX_RUD * (held("d") - held("a")),
    };
    const u = { "prop.throttle": throttle };
    for (const n of SURFACES) u[`${n}.deflection_cmd`] = target[n];
    for (let i = 0; i < SUBSTEPS; i++) { sim.step(DT, u); t += DT; }
  }

  function applyPose(curr, a) {
    const P = SLOT.position, O = SLOT.orientation;
    const pos = [lerp(prev[P], curr[P], a), lerp(prev[P + 1], curr[P + 1], a),
                 lerp(prev[P + 2], curr[P + 2], a)];
    plane.root.position.set(pos[0], pos[1], pos[2]);
    const q = quatWXYZ(prev, O).slerp(quatWXYZ(curr, O), a);
    plane.root.quaternion.copy(q);
    for (const n of SURFACES) {
      const d = lerp(prev[SLOT[`${n}.deflection`]], curr[SLOT[`${n}.deflection`]], a);
      plane.surf[n].group.quaternion.setFromAxisAngle(plane.surf[n].axis, d);
    }
    return { pos, yaw: yawOf(q) };
  }

  // snap the chase camera to its pose so the airframe is framed from frame 0
  camera.position.set(-25, 7, CRUISE_ALT + 7);
  camera._look = new THREE.Vector3(8, 0, CRUISE_ALT);
  camera.lookAt(camera._look);

  let raf = 0;
  function frame(now) {
    const rdt = Math.min((now - last) / 1000, 0.05); last = now; acc += rdt;
    while (acc >= FRAME) { prev = Float64Array.from(sim.state); controlFrame(); acc -= FRAME; }
    const a = Math.min(acc / FRAME, 1);
    const { pos, yaw } = applyPose(sim.state, a);

    plane.prop.rotation.x += 0.6 + throttle * 2.4;
    if (plane.skel.visible) plane.thrust.set([0.15, 0, 0], [throttle * THRUST, 0, 0]);

    // chase camera: behind + slightly to the side (3/4 view), above
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const eye = new THREE.Vector3(
      pos[0] - 25 * cy + 7 * sy, pos[1] - 25 * sy - 7 * cy, pos[2] + 7);
    camera.position.lerp(eye, 1 - Math.exp(-rdt / 0.28));
    camera._look = (camera._look || new THREE.Vector3()).lerp(
      new THREE.Vector3(pos[0] + 8 * Math.cos(yaw), pos[1] + 8 * Math.sin(yaw), pos[2]),
      1 - Math.exp(-rdt / 0.28));
    camera.lookAt(camera._look);
    sun.target.position.set(pos[0], pos[1], pos[2]); sun.target.updateMatrixWorld();

    fitToCanvas(renderer, camera, canvas);
    renderer.render(scene, camera);
    if (ctx.setHud) {
      const v = Math.hypot(sim.state[SLOT.velocity], sim.state[SLOT.velocity + 1], sim.state[SLOT.velocity + 2]);
      ctx.setHud(`alt ${pos[2].toFixed(0)} m   spd ${v.toFixed(0)} m/s   ${interactive ? "MANUAL" : "AUTOPILOT"}`);
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
    setSkeleton(on) { plane.skel.visible = on; },
    setInteractive(on) { interactive = on; if (on) { down.clear(); } else { reset(); } },
  };
}
