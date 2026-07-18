// Marine life for the drive view: ~30 species of procedurally built fish,
// sharks, rays, eels, a lobster and three marine mammals, spawned in
// schools/pods around the craft with depth-stratified abundance.
//
// Data model: each SPECIES entry carries real-world-ish length, colours,
// body proportions, depth preference in the 0–30 m column, relative
// abundance weight, cruise speed and group size. Groups are recycled
// through the fog ring around the craft; which species spawns is drawn
// from the abundance weights, and WHERE it spawns vertically comes from
// its depth band — so surface fish live up top and bottom fish down deep.
//
// Models: one merged vertex-coloured geometry per species (body of
// revolution + fin quads, all primitives). Swimming bends the model along
// a SINGLE axis in the vertex shader — laterally (y) for fish/sharks/eels,
// vertically (z) for mammals, wing-flap for rays — driven by a per-group
// accumulated phase so tail-beat rate can track swim speed continuously.
//
// Coordinates: fish local frame is nose +x, y left, z up. Instance
// matrices hold member offsets from the GROUP centre; the InstancedMesh
// object itself carries the (float64) world position of the group.

const TARGET_GROUPS = 28;
const RECYCLE_D = 160, SPAWN_D0 = 55, SPAWN_D1 = 110;

// --- small colour helpers ----------------------------------------------------
const C = (hex) => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255,
                    (hex & 255) / 255];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t,
                          a[1] + (b[1] - a[1]) * t,
                          a[2] + (b[2] - a[2]) * t];
const hash2 = (a, b) => {
  const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

// paint(tt, th, zf): tt = 0 nose … 1 tail, th = angle around the section,
// zf = vertical −1 belly … +1 back. Returns [r,g,b].
const shade = (top, belly) => (tt, th, zf) =>
  mix(belly, top, 0.35 + 0.65 * Math.max(0, zf * 0.5 + 0.5));
const bars = (base, dark, n, w, topOnly = false) => (tt, th, zf) => {
  const c = base(tt, th, zf);
  if (topOnly && zf < -0.1) return c;
  const f = (tt * n) % 1;
  return f < w && tt > 0.08 && tt < 0.92 ? mix(c, dark, 0.75) : c;
};
const spots = (base, spot, d, scale = 9) => (tt, th, zf) => {
  const c = base(tt, th, zf);
  return hash2(Math.floor(tt * scale), Math.floor(th * 3.2)) < d
    ? mix(c, spot, 0.6) : c;
};

// --- the species table --------------------------------------------------------
// len m · girth [w,h] (half-extents as fraction of len) · depth: [mu,sigma]
// metres below surface, or "bed" (+clearance) · w = abundance weight ·
// spd = cruise m/s · grp = [min,max] group size · cls drives geometry+bend.
// Sizes/looks/habits from standard references (spot-checked 2026-07):
// e.g. treefish ≤41 cm yellow w/ 6 black bars + red lips, demersal;
// honeycomb rockfish ≤30 cm orange-brown honeycomb, the deepest band here;
// monkeyface prickleback eel-like ≤76 cm uniform dark olive, crevices.
export const SPECIES = [
  { id: "mako_shark", cls: "shark", len: 2.8, girth: [0.055, 0.075],
    depth: [6, 4], w: 1.0, spd: 1.6, grp: [1, 1], tail: "lunate",
    paint: shade(C(0x1b3d6e), C(0xe8eef2)) },
  { id: "leopard_shark", cls: "shark", len: 1.4, girth: [0.06, 0.075],
    depth: "bed", w: 5.0, spd: 0.5, grp: [3, 7], tail: "hetero",
    paint: bars(shade(C(0x8a7355), C(0xd8cfc0)), C(0x2e2620), 9, 0.42, true) },
  { id: "bat_ray", cls: "ray", len: 1.1, span: 1.5, depth: "bed",
    w: 4.0, spd: 0.6, grp: [1, 4],
    paint: shade(C(0x2f2a33), C(0xe8e4da)) },
  { id: "treefish", cls: "fish", len: 0.3, girth: [0.07, 0.14],
    depth: "bed", w: 2.5, spd: 0.25, grp: [1, 2], behav: "anchor",
    tail: "round", paint: (tt, th, zf) => {
      if (tt < 0.05) return C(0xc03a2e);              // the red lips
      return bars(shade(C(0xc8a832), C(0xd8c878)), C(0x1c1a12), 6, 0.4)(
        tt, th, zf);
    } },
  { id: "barred_sand_bass", cls: "fish", len: 0.45, girth: [0.065, 0.13],
    depth: "bed", w: 5.0, spd: 0.4, grp: [2, 6], tail: "round",
    paint: bars(shade(C(0x9a8f7a), C(0xd9d2c2)), C(0x4a4437), 7, 0.38, true) },
  { id: "halfmoon", cls: "fish", len: 0.26, girth: [0.06, 0.14],
    depth: [6, 3], w: 7.0, spd: 0.5, grp: [6, 14], tail: "round",
    paint: shade(C(0x5a6f7d), C(0x9fb0ba)) },
  { id: "zebra_perch", cls: "fish", len: 0.3, girth: [0.06, 0.14],
    depth: [5, 3], w: 4.0, spd: 0.5, grp: [4, 10], tail: "round",
    paint: bars(shade(C(0x6e7a52), C(0xb9bfa4)), C(0x30362a), 9, 0.3, true) },
  { id: "rainbow_surfperch", cls: "fish", len: 0.25, girth: [0.06, 0.14],
    depth: [9, 4], w: 3.5, spd: 0.4, grp: [4, 8], tail: "round",
    paint: (tt, th, zf) => {                          // orange/blue stripes
      if (zf < -0.55) return C(0xd8b070);
      const band = Math.floor((zf + 1) * 3.5) % 2;
      return band ? C(0x4a7fae) : C(0xd07038);
    } },
  { id: "walleye_surfperch", cls: "fish", len: 0.23, girth: [0.055, 0.13],
    depth: [4, 2.5], w: 5.0, spd: 0.5, grp: [8, 16], tail: "fork",
    paint: shade(C(0x9fb0bc), C(0xdde5ea)) },
  { id: "olive_rockfish", cls: "fish", len: 0.4, girth: [0.06, 0.13],
    depth: [12, 5], w: 5.0, spd: 0.5, grp: [3, 8], tail: "fork",
    paint: spots(shade(C(0x6b6a45), C(0xb8b48e)), C(0xcfc9a0), 0.22, 7) },
  { id: "kelp_greenling", cls: "fish", len: 0.35, girth: [0.06, 0.12],
    depth: "bed", w: 3.0, spd: 0.35, grp: [1, 2], tail: "round",
    paint: spots(shade(C(0x7a6a58), C(0xc4b6a2)), C(0x4a90b8), 0.3, 11) },
  { id: "sargo", cls: "fish", len: 0.4, girth: [0.06, 0.15],
    depth: [8, 4], w: 3.5, spd: 0.45, grp: [3, 7], tail: "fork",
    paint: (tt, th, zf) => {                          // one dark shoulder bar
      const base = shade(C(0xb8bec4), C(0xe6eaec))(tt, th, zf);
      return tt > 0.3 && tt < 0.38 ? mix(base, C(0x2c2c30), 0.8) : base;
    } },
  { id: "honeycomb_rockfish", cls: "fish", len: 0.22, girth: [0.065, 0.13],
    depth: "bed", w: 2.0, spd: 0.25, grp: [1, 3], tail: "round",
    paint: spots(shade(C(0xb06038), C(0xd8a684)), C(0xe8d8b8), 0.4, 13) },
  { id: "lionfish", cls: "fish", len: 0.3, girth: [0.07, 0.14],
    depth: "bed", w: 1.2, spd: 0.2, grp: [1, 1], tail: "round", fan: true,
    paint: bars(shade(C(0x9a3a2a), C(0xc9a08a)), C(0xf0e6da), 11, 0.45) },
  { id: "monkeyface_prickleback", cls: "eel", len: 0.55, girth: [0.035, 0.05],
    depth: "bed", w: 2.0, spd: 0.3, grp: [1, 1], behav: "anchor",
    paint: shade(C(0x4a4f3d), C(0x6e7258)) },
  { id: "bocaccio", cls: "fish", len: 0.6, girth: [0.06, 0.13],
    depth: [22, 5], w: 2.5, spd: 0.5, grp: [3, 7], tail: "fork",
    paint: shade(C(0x9a6a52), C(0xd0b09a)) },
  { id: "black_surfperch", cls: "fish", len: 0.28, girth: [0.06, 0.14],
    depth: [18, 6], w: 4.5, spd: 0.4, grp: [3, 8], tail: "round",
    paint: bars(shade(C(0x5c4a3a), C(0x8a7660)), C(0x3a2e24), 8, 0.35) },
  { id: "nurse_shark", cls: "shark", len: 2.4, girth: [0.065, 0.075],
    depth: "bed", w: 0.8, spd: 0.3, grp: [1, 2], tail: "round",
    paint: shade(C(0xa58a63), C(0xc9b795)) },
  { id: "ocean_sunfish", cls: "sunfish", len: 1.6, girth: [0.045, 0.28],
    depth: [4, 3], w: 1.2, spd: 0.4, grp: [1, 1],
    paint: spots(shade(C(0x8d959b), C(0xb8bec2)), C(0x6a7178), 0.3, 6) },
  { id: "great_white", cls: "shark", len: 4.6, girth: [0.06, 0.08],
    depth: [10, 6], w: 0.5, spd: 1.7, grp: [1, 1], tail: "lunate",
    paint: shade(C(0x6a7480), C(0xeef1f2)) },
  { id: "yellowfin_tuna", cls: "fish", len: 1.5, girth: [0.06, 0.1],
    depth: [6, 4], w: 1.8, spd: 3.0, grp: [4, 9], tail: "lunate",
    finC: C(0xe8c93a), paint: shade(C(0x23415e), C(0xdfe6ea)) },
  { id: "black_sea_bass", cls: "fish", len: 1.8, girth: [0.075, 0.15],
    depth: [20, 6], w: 1.0, spd: 0.5, grp: [1, 2], tail: "round",
    paint: spots(shade(C(0x3c4348), C(0x6d7478)), C(0x22262a), 0.25, 8) },
  { id: "kelp_bass", cls: "fish", len: 0.45, girth: [0.065, 0.13],
    depth: [10, 5], w: 8.0, spd: 0.45, grp: [2, 6], tail: "round",
    paint: spots(shade(C(0x79704f), C(0xc2b896)), C(0xd8d2b0), 0.35, 6) },
  { id: "giant_moray", cls: "eel", len: 2.2, girth: [0.035, 0.05],
    depth: "bed", w: 0.8, spd: 0.15, grp: [1, 1], behav: "anchor",
    paint: spots(shade(C(0x6b6b3f), C(0x8f8f5e)), C(0x3c3c22), 0.35, 14) },
  { id: "spiny_lobster", cls: "lobster", len: 0.4, depth: "bed",
    w: 3.0, spd: 0.08, grp: [2, 5], behav: "crawl", standH: 0.05,
    paint: shade(C(0x8a3a26), C(0xa5573c)) },
  { id: "rock_crab", cls: "crab", len: 0.15, depth: "bed",
    w: 3.0, spd: 0.1, grp: [2, 5], behav: "crawl", standH: 0.03,
    sideways: true, paint: shade(C(0x9a4a30), C(0xc07a52)) },
  { id: "barracuda", cls: "fish", len: 1.1, girth: [0.04, 0.06],
    depth: [5, 3], w: 2.5, spd: 1.1, grp: [3, 8], tail: "fork",
    paint: shade(C(0xaab6bd), C(0xe2e8ea)) },
  // mammals: pods of ≥3, vertical bend, periodic runs toward the surface
  { id: "harbor_seal", cls: "mammal", len: 1.6, girth: [0.095, 0.1],
    depth: [6, 4], w: 1.8, spd: 1.2, grp: [3, 5],
    paint: spots(shade(C(0x8b8b86), C(0xb6b4ac)), C(0x55554e), 0.3, 10) },
  { id: "sea_lion", cls: "mammal", len: 2.1, girth: [0.085, 0.09],
    depth: [6, 4], w: 2.2, spd: 1.6, grp: [3, 7],
    paint: shade(C(0x6a4f38), C(0x8a6c50)) },
  { id: "common_dolphin", cls: "mammal", len: 2.1, girth: [0.08, 0.085],
    depth: [5, 3], w: 1.8, spd: 2.5, grp: [5, 10], dolphin: true,
    paint: (tt, th, zf) => {                          // hourglass flank
      if (zf < -0.45) return C(0xe8e6de);
      if (zf < 0.35 && tt > 0.2 && tt < 0.62) return C(0xd9c9a2);
      return C(0x3b4652);
    } },
];
const W_TOTAL = SPECIES.reduce((s, x) => s + x.w, 0);

export function makeFauna(THREE, scene, { seabed, bedHeight }) {
  const dummy = new THREE.Object3D();
  dummy.rotation.order = "ZYX";                // yaw ∘ pitch
  const _col = new THREE.Color();
  const F = (x) => Number(x).toExponential(6);

  // --- geometry ---------------------------------------------------------------
  function out3() { return { pos: [], nrm: [], col: [] }; }
  function tri(o, a, b, c, col) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz,
        nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (const p of [a, b, c]) {
      o.pos.push(p[0], p[1], p[2]);
      o.nrm.push(nx, ny, nz);
      o.col.push(col[0], col[1], col[2]);
    }
  }
  function quad(o, a, b, c, d, col) { tri(o, a, b, c, col); tri(o, a, c, d, col); }

  // body of revolution along +x (nose at +len/2), elliptic cross-section,
  // per-vertex paint. Returns peduncle info for the tail builder.
  function body(o, spec, prof) {
    const L = spec.len, [gw, gh] = spec.girth;
    const NR = 22, RAD = 9;
    const ring = (t) => {
      const p = prof(t);
      return { x: (0.5 - t) * L, rw: p * gw * L, rh: p * gh * L };
    };
    for (let i = 0; i < NR; i++) {
      const t0 = i / NR, t1 = (i + 1) / NR;
      const r0 = ring(t0), r1 = ring(t1);
      for (let a = 0; a < RAD; a++) {
        const th0 = a / RAD * Math.PI * 2, th1 = (a + 1) / RAD * Math.PI * 2;
        const c = spec.paint((t0 + t1) / 2, (th0 + th1) / 2,
                             Math.cos((th0 + th1) / 2));
        const P = (r, th) => [r.x, r.rw * Math.sin(th), r.rh * Math.cos(th)];
        quad(o, P(r0, th0), P(r1, th0), P(r1, th1), P(r0, th1), c);
      }
    }
    return ring(1);
  }
  const stdProf = (nose = 0.28, noseP = 0.6, tailR = 0.2) => (t) =>
    t < nose ? Math.pow(t / nose, noseP)
             : 1 - (1 - tailR) * Math.pow((t - nose) / (1 - nose), 1.4);

  function caudal(o, spec, ped, finC) {
    const L = spec.len, x0 = ped.x, h = ped.rh;
    const add = (up, lo, sweep) => {
      tri(o, [x0, 0, h * 0.5], [x0 - sweep, 0, up], [x0 - sweep * 0.45, 0, 0],
          finC);
      tri(o, [x0, 0, -h * 0.5], [x0 - sweep * 0.45, 0, 0],
          [x0 - sweep, 0, -lo], finC);
    };
    if (spec.tail === "lunate") add(L * 0.16, L * 0.16, L * 0.13);
    else if (spec.tail === "hetero") add(L * 0.17, L * 0.09, L * 0.15);
    else if (spec.tail === "fork") add(L * 0.11, L * 0.11, L * 0.12);
    else quad(o, [x0, 0, h], [x0 - L * 0.11, 0, h * 0.9],
              [x0 - L * 0.11, 0, -h * 0.9], [x0, 0, -h], finC);
  }

  function fins(o, spec, finC) {
    const L = spec.len, [gw, gh] = spec.girth;
    // dorsal
    const dx = L * 0.08, dh = gh * L * (spec.cls === "shark" ? 1.5 : 1.1);
    tri(o, [dx + L * 0.09, 0, gh * L * 0.85], [dx - L * 0.02, 0, gh * L + dh],
        [dx - L * 0.07, 0, gh * L * 0.85], finC);
    // pectorals (lionfish: big fans)
    const n = spec.fan ? 3 : 1;
    for (let s = -1; s <= 1; s += 2)
      for (let k = 0; k < n; k++) {
        const px = L * 0.18, pl = L * (spec.fan ? 0.3 : 0.14);
        const ang = (spec.fan ? -0.5 + k * 0.5 : -0.25);
        quad(o, [px, s * gw * L * 0.8, 0],
             [px - pl, s * (gw * L + pl * 0.8), pl * Math.sin(ang)],
             [px - pl * 1.25, s * (gw * L + pl * 0.55),
              pl * Math.sin(ang) - pl * 0.18],
             [px - pl * 0.3, s * gw * L * 0.8, -pl * 0.12], finC);
      }
  }

  function buildGeo(spec) {
    const o = out3();
    const finC = spec.finC ?? mix(spec.paint(0.5, 0, 0.6), [0, 0, 0], 0.25);
    if (spec.cls === "ray") {
      const sp = spec.span / 2, L = spec.len;
      const NY = 8;
      for (const [zoff, cf] of [[0.012, spec.paint(0.3, 0, 1)],
                                [-0.012, spec.paint(0.3, 0, -1)]]) {
        for (let i = -NY; i < NY; i++) {
          const u0 = i / NY, u1 = (i + 1) / NY;
          const ch = (u) => L * (1 - Math.pow(Math.abs(u), 1.25));
          const dome = (u) => zoff + 0.07 * L * (1 - Math.abs(u));
          const P = (u, v) => [ch(u) * (0.45 - v), u * sp, dome(u)];
          quad(o, P(u0, 0), P(u0, 1), P(u1, 1), P(u1, 0), cf);
        }
      }
      // whip tail
      quad(o, [-L * 0.5, 0.01, 0], [-L * 1.15, 0.004, 0.01],
           [-L * 1.15, -0.004, 0.01], [-L * 0.5, -0.01, 0], finC);
    } else if (spec.cls === "sunfish") {
      const ped = body(o, spec, stdProf(0.42, 0.85, 0.62));
      const L = spec.len, gh = spec.girth[1];
      // clavus (the truncated "tail fin")
      quad(o, [ped.x, 0, ped.rh * 1.4], [ped.x - L * 0.09, 0, ped.rh],
           [ped.x - L * 0.09, 0, -ped.rh], [ped.x, 0, -ped.rh * 1.4], finC);
      // the huge dorsal + anal blades
      tri(o, [L * 0.06, 0, gh * L * 0.8], [-L * 0.12, 0, gh * L + L * 0.5],
          [-L * 0.16, 0, gh * L * 0.7], finC);
      tri(o, [L * 0.06, 0, -gh * L * 0.8], [-L * 0.16, 0, -gh * L * 0.7],
          [-L * 0.12, 0, -(gh * L + L * 0.5)], finC);
    } else if (spec.cls === "lobster") {
      const L = spec.len;
      spec.girth = [0.11, 0.1];
      const ped = body(o, { ...spec, len: L * 0.62 }, stdProf(0.1, 0.8, 0.5));
      // tail fan
      quad(o, [ped.x, 0, 0.01], [ped.x - L * 0.16, L * 0.09, 0],
           [ped.x - L * 0.2, 0, 0], [ped.x - L * 0.16, -L * 0.09, 0],
           spec.paint(0.9, 0, 0));
      // antennae + legs (thin quads read fine at this scale)
      const aC = mix(spec.paint(0.1, 0, 0), [0, 0, 0], 0.2);
      for (let s = -1; s <= 1; s += 2) {
        quad(o, [L * 0.3, s * 0.02, 0.02], [L * 0.95, s * 0.16, L * 0.22],
             [L * 0.95, s * 0.17, L * 0.22], [L * 0.3, s * 0.03, 0.02], aC);
        for (let k = 0; k < 3; k++)
          quad(o, [L * (0.15 - k * 0.09), s * 0.03, -0.01],
               [L * (0.12 - k * 0.09), s * 0.11, -L * 0.11],
               [L * (0.13 - k * 0.09), s * 0.12, -L * 0.11],
               [L * (0.16 - k * 0.09), s * 0.04, -0.01], aC);
      }
    } else if (spec.cls === "crab") {
      const L = spec.len;
      // wide flat carapace (crabs are broader than long)
      spec.girth = [0.5, 0.18];
      body(o, { ...spec, len: L * 0.7 }, stdProf(0.5, 1.0, 0.85));
      const legC = mix(spec.paint(0.5, 0, 0), [0, 0, 0], 0.15);
      for (let s = -1; s <= 1; s += 2) {
        for (let k = 0; k < 4; k++) {            // 4 walking legs per side
          const lx = L * (0.18 - k * 0.13);
          quad(o, [lx, s * L * 0.3, 0],
               [lx - L * 0.05, s * L * 0.62, -L * 0.06],
               [lx - L * 0.03, s * L * 0.66, -L * 0.14],
               [lx + L * 0.02, s * L * 0.32, -L * 0.02], legC);
        }
        // claw held forward
        quad(o, [L * 0.3, s * L * 0.16, 0], [L * 0.52, s * L * 0.26, -0.01],
             [L * 0.56, s * L * 0.18, -0.015],
             [L * 0.34, s * L * 0.1, -0.005], legC);
      }
    } else if (spec.cls === "mammal") {
      const ped = body(o, spec,
        stdProf(spec.dolphin ? 0.2 : 0.3, spec.dolphin ? 1.5 : 0.8, 0.16));
      const L = spec.len;
      // horizontal fluke (mammals) / rear flippers (pinnipeds)
      quad(o, [ped.x, 0, 0], [ped.x - L * 0.11, L * 0.14, 0],
           [ped.x - L * 0.15, 0, 0], [ped.x - L * 0.11, -L * 0.14, 0], finC);
      if (spec.dolphin)                        // falcate dorsal
        tri(o, [L * 0.1, 0, spec.girth[1] * L * 0.9],
            [-L * 0.04, 0, spec.girth[1] * L + L * 0.1],
            [-L * 0.08, 0, spec.girth[1] * L * 0.85], finC);
      for (let s = -1; s <= 1; s += 2)         // fore flippers
        quad(o, [L * 0.22, s * spec.girth[0] * L * 0.8, -0.02],
             [L * 0.1, s * (spec.girth[0] * L + L * 0.13), -L * 0.06],
             [L * 0.05, s * (spec.girth[0] * L + L * 0.1), -L * 0.07],
             [L * 0.14, s * spec.girth[0] * L * 0.8, -0.03], finC);
    } else {
      // fish / shark / eel
      const prof = spec.cls === "eel"
        ? (t) => Math.min(1, t / 0.07) * (1 - 0.7 * Math.pow(Math.max(0,
            (t - 0.5) / 0.5), 1.3))
        : stdProf(spec.cls === "shark" ? 0.35 : 0.28,
                  spec.cls === "shark" ? 0.7 : 0.6);
      const ped = body(o, spec, prof);
      caudal(o, spec, ped, finC);
      if (spec.cls !== "eel") fins(o, spec, finC);
      if (spec.cls === "shark")                // second dorsal
        tri(o, [-spec.len * 0.28, 0, spec.girth[1] * spec.len * 0.55],
            [-spec.len * 0.34, 0, spec.girth[1] * spec.len * 0.95],
            [-spec.len * 0.36, 0, spec.girth[1] * spec.len * 0.5], finC);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(o.pos), 3));
    g.setAttribute("normal",
      new THREE.BufferAttribute(new Float32Array(o.nrm), 3));
    g.setAttribute("color",
      new THREE.BufferAttribute(new Float32Array(o.col), 3));
    return g;
  }
  const geoCache = new Map();
  const geoFor = (spec) => {
    if (!geoCache.has(spec.id)) geoCache.set(spec.id, buildGeo(spec));
    return geoCache.get(spec.id);
  };

  // --- swim material ------------------------------------------------------------
  // Bend along ONE axis, weight growing toward the tail (or the wingtips
  // for rays). uPhase is accumulated CPU-side per group so the beat rate
  // follows swim speed without phase jumps.
  function swimChunk(spec) {
    const L = F(spec.len);
    if (spec.cls === "ray")
      return `
        float wf_ = pow(min(1.0, abs(position.y) / ${F(spec.span * 0.5)}),
                        1.6);
        transformed.z += ${F(spec.span * 0.14)} * wf_
          * sin(uPhase + float(gl_InstanceID) * 2.399);`;
    const amp = { eel: 0.14, shark: 0.09, mammal: 0.07, sunfish: 0.02,
                  lobster: 0.0, crab: 0.0 }[spec.cls] ?? 0.09;
    if (amp === 0) return "";
    // travelling wave: near-zero amplitude at the nose, growing ~tt² to
    // the tail, with enough phase lag along the body that the tail FLICKS
    // while the head barely moves — a rigid in-phase sway reads as the
    // whole fish pivoting, which looks broken
    const k = { eel: 7.0, shark: 3.6, mammal: 2.6 }[spec.cls] ?? 4.2;
    const axis = spec.cls === "mammal" ? "z" : "y";
    return `
      float tt_ = clamp(0.5 - position.x / ${L}, 0.0, 1.0);
      transformed.${axis} += ${F(amp * spec.len)}
        * (0.02 + 0.98 * pow(tt_, 2.2))
        * sin(uPhase + float(gl_InstanceID) * 2.399 - tt_ * ${F(k)});`;
  }
  function makeSwimMat(spec) {
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true,
      roughness: 0.55, metalness: 0.2, side: THREE.DoubleSide });
    const uPhase = { value: Math.random() * 10 };
    mat.userData.uPhase = uPhase;
    const chunk = swimChunk(spec);
    if (chunk)
      mat.onBeforeCompile = (sh) => {
        sh.uniforms.uPhase = uPhase;
        sh.vertexShader = "uniform float uPhase;\n" + sh.vertexShader
          .replace("#include <begin_vertex>",
                   `#include <begin_vertex>\n{${chunk}\n}`);
      };
    return mat;
  }

  // --- groups ---------------------------------------------------------------
  const groups = [];

  function sampleSpecies() {
    let r = Math.random() * W_TOTAL;
    for (const s of SPECIES) { r -= s.w; if (r <= 0) return s; }
    return SPECIES[SPECIES.length - 1];
  }
  const gauss = () => {
    let u = 0; while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * Math.random());
  };
  function depthFor(spec, x, y) {
    if (spec.depth === "bed")
      return bedHeight(x, y) + 0.3 + Math.random() * (spec.cls === "ray"
        ? 1.2 : 0.8) + spec.len * 0.2;
    const z = -(spec.depth[0] + gauss() * spec.depth[1]);
    return Math.min(-0.6 - spec.len * 0.3, Math.max(seabed + 1, z));
  }

  function spawnGroup(cx, cy, spec = sampleSpecies(), ringR = null) {
    const az = Math.random() * Math.PI * 2;
    const d = ringR ?? (SPAWN_D0 + Math.random() * (SPAWN_D1 - SPAWN_D0));
    const gx = cx + Math.cos(az) * d, gy = cy + Math.sin(az) * d;
    const gz = depthFor(spec, gx, gy);
    const n = spec.grp[0]
      + Math.floor(Math.random() * (spec.grp[1] - spec.grp[0] + 1));
    const mesh = new THREE.InstancedMesh(geoFor(spec), makeSwimMat(spec), n);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const spread = spec.len * 1.3 + n * 0.28;
    const members = [];
    for (let i = 0; i < n; i++) {
      members.push({
        sx: (Math.random() - 0.5) * spread,       // slot in the formation
        sy: (Math.random() - 0.5) * spread,
        sz: (Math.random() - 0.5) * spread * 0.4,
        x: 0, y: 0, z: 0,                          // offsets from centre
        vx: 0, vy: 0, vz: 0,
        yaw: Math.random() * Math.PI * 2, pitch: 0,
        sc: 0.85 + Math.random() * 0.3,
      });
      members[i].x = members[i].sx; members[i].y = members[i].sy;
      members[i].z = members[i].sz;
      _col.setScalar(0.9 + Math.random() * 0.2);
      mesh.setColorAt(i, _col);
    }
    const g = {
      spec, mesh, members,
      cx: gx, cy: gy, cz: gz,
      heading: Math.random() * Math.PI * 2,
      turnT: 0, breathT: 8 + Math.random() * 20, breathing: false,
      behav: spec.behav ?? (spec.cls === "mammal" ? "mammal"
        : spec.depth === "bed" ? "bottom" : "cruise"),
    };
    groups.push(g);
    return g;
  }

  function disposeGroup(g) {
    scene.remove(g.mesh);
    g.mesh.material.dispose();
    g.mesh.dispose();                            // instance buffers
  }

  function updateGroup(g, dt, t, cx, cy) {
    const s = g.spec;
    // --- group centre -------------------------------------------------------
    let spd = s.spd;
    if (g.behav === "anchor") spd = s.spd * 0.15;
    if (g.behav === "crawl") spd = s.spd;        // already walking pace
    if (g.behav === "mammal") {
      g.breathT -= dt;
      if (g.breathT < 0) {
        g.breathing = !g.breathing;
        g.breathT = g.breathing ? 5 + Math.random() * 4
                                : 15 + Math.random() * 25;
      }
    }
    g.turnT -= dt;
    if (g.turnT < 0) {
      g.turnT = 4 + Math.random() * 8;
      g.targetHeading = g.heading + (Math.random() - 0.5) * 2.2;
      // drift loosely toward the craft so groups wander into view
      if (Math.hypot(g.cx - cx, g.cy - cy) > 55)
        g.targetHeading = Math.atan2(cy - g.cy, cx - g.cx)
          + (Math.random() - 0.5) * 1.4;
    }
    let dh = ((g.targetHeading ?? g.heading) - g.heading + Math.PI * 3)
      % (Math.PI * 2) - Math.PI;
    g.heading += Math.max(-0.5 * dt, Math.min(0.5 * dt, dh));
    // centre moves at the same cruise speed the members swim, so nobody
    // systematically outruns the formation
    g.cx += Math.cos(g.heading) * spd * dt;
    g.cy += Math.sin(g.heading) * spd * dt;
    // vertical: bed-followers hug the floor, mammals run for air sometimes
    let tz = g.cz;
    if (g.behav === "crawl")
      tz = bedHeight(g.cx, g.cy) + (s.standH ?? 0.04);
    else if (g.behav === "bottom" || g.behav === "anchor")
      tz = bedHeight(g.cx, g.cy) + 0.3 + s.len * 0.25;
    else if (g.behav === "mammal")
      tz = g.breathing ? -0.5 : -(s.depth[0] || 5);
    g.cz += (tz - g.cz) * Math.min(1, dt * (g.behav === "mammal" ? 0.5 : 0.3));
    g.cz = Math.min(-0.4, Math.max(seabed + 0.3, g.cz));

    // --- members ----------------------------------------------------------------
    // Everyone swims the GROUP's direction at cruise speed, with a gentle
    // capped correction toward their (slowly swirling) formation slot.
    // Never chase the slot point directly: at the slot the error direction
    // flips every frame and the fish spins in place at the yaw slew limit.
    const agile = Math.min(1, dt * 1.2);
    const hx = Math.cos(g.heading), hy = Math.sin(g.heading);
    const corrMax = Math.max(0.06, spd * 0.6);
    let avgSpd = 0;
    for (const [i, m] of g.members.entries()) {
      const rot = (t * 0.025 + i * 0.7);         // slots slowly circulate
      const txp = m.sx * Math.cos(rot) - m.sy * Math.sin(rot);
      const typ = m.sx * Math.sin(rot) + m.sy * Math.cos(rot);
      const dx = txp - m.x, dy = typ - m.y, dz = m.sz - m.z;
      const cl = (v) => Math.max(-corrMax, Math.min(corrMax, v));
      let dvx, dvy, dvz;
      if (g.behav === "anchor" || g.behav === "crawl") {
        // station-keeping with a deadband so nothing dances on its spot
        const dist = Math.hypot(dx, dy);
        const go = dist > Math.max(0.15, s.len * 0.4) ? spd : 0;
        dvx = dist > 1e-4 ? dx / dist * go : 0;
        dvy = dist > 1e-4 ? dy / dist * go : 0;
        dvz = 0;
      } else {
        dvx = hx * spd + cl(dx * 0.4);
        dvy = hy * spd + cl(dy * 0.4);
        dvz = cl(dz * 0.35);
      }
      m.vx += (dvx - m.vx) * agile;
      m.vy += (dvy - m.vy) * agile;
      m.vz += (dvz - m.vz) * agile;
      m.x += m.vx * dt;
      m.y += m.vy * dt;
      m.z += m.vz * dt;
      const wz = g.cz + m.z;
      if (g.behav === "crawl") {
        // feet ON the seabed: z is the local floor, always
        m.z = bedHeight(g.cx + m.x, g.cy + m.y) + (s.standH ?? 0.04) - g.cz;
        m.vz = 0;
      } else {
        const floor = bedHeight(g.cx + m.x, g.cy + m.y) + 0.2 + s.len * 0.15;
        if (wz < floor) m.z = floor - g.cz;
        if (wz > -0.4) m.z = -0.4 - g.cz;
      }
      const sp = Math.hypot(m.vx, m.vy, m.vz);
      avgSpd += sp;
      if (g.behav === "anchor") {
        // hover in place, just sway the heading
        m.yaw += Math.sin(t * 0.6 + i * 2.1) * 0.09 * dt;
      } else if (sp > Math.max(0.04, spd * 0.15)) {
        // only steer while actually swimming — coast through lulls so a
        // parked fish never twitches its heading
        const wantYaw = Math.atan2(m.vy, m.vx)
          + (s.sideways ? Math.PI / 2 : 0);
        let dyaw = (wantYaw - m.yaw + Math.PI * 3) % (Math.PI * 2) - Math.PI;
        m.yaw += Math.max(-1.8 * dt, Math.min(1.8 * dt, dyaw));
        if (g.behav !== "crawl")
          m.pitch += (-Math.asin(Math.max(-0.8, Math.min(0.8, m.vz
            / (sp + 0.05)))) * 0.6 - m.pitch) * agile;
      }
      dummy.position.set(m.x, m.y, m.z);
      dummy.rotation.set(0, m.pitch, m.yaw);
      dummy.scale.setScalar(m.sc);
      dummy.updateMatrix();
      g.mesh.setMatrixAt(i, dummy.matrix);
    }
    g.mesh.instanceMatrix.needsUpdate = true;
    g.mesh.position.set(g.cx, g.cy, g.cz);

    // tail beat follows actual speed — big animals beat SLOW (f ~ v/L),
    // and anchored eels keep a sinuous idle
    avgSpd = avgSpd / g.members.length + spd * 0.2;
    const f = s.cls === "eel" ? 0.5 + 0.4 * avgSpd / s.len
      : Math.max(0.3, Math.min(3.0, 0.35 + 0.5 * avgSpd / s.len));
    g.mesh.material.userData.uPhase.value += Math.PI * 2 * f * dt;
  }

  // --- public API -------------------------------------------------------------
  let booted = false;
  return {
    update(dt, t, cx, cy) {
      if (!booted) {
        // initial fill: populate the whole bubble, including close by —
        // there's no pop-in at load time, the world is just starting
        booted = true;
        while (groups.length < TARGET_GROUPS)
          spawnGroup(cx, cy, sampleSpecies(),
                     20 + Math.random() * (SPAWN_D1 - 20));
      }
      for (let i = groups.length - 1; i >= 0; i--) {
        const g = groups[i];
        if (Math.hypot(g.cx - cx, g.cy - cy) > RECYCLE_D) {
          disposeGroup(g);
          groups.splice(i, 1);
          continue;
        }
        updateGroup(g, dt, t, cx, cy);
      }
      if (groups.length < TARGET_GROUPS) spawnGroup(cx, cy);
    },
    /** debug: one group of every species in a ring at `r` around (x,y). */
    _parade(x, y, r = 14) {
      for (const [i, s] of SPECIES.entries()) {
        const az = i / SPECIES.length * Math.PI * 2;
        const g = spawnGroup(x - Math.cos(az) * r, y - Math.sin(az) * r, s, 0);
        g.cx = x + Math.cos(az) * r;
        g.cy = y + Math.sin(az) * r;
        g.mesh.position.set(g.cx, g.cy, g.cz);
      }
    },
    dispose() {
      for (const g of groups) disposeGroup(g);
      groups.length = 0;
      for (const g of geoCache.values()) g.dispose();
      geoCache.clear();
    },
  };
}
