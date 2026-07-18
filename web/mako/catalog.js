// Module catalog for the "Build a Mako" editor + viewer — transcribed from
// the design document `MakoModules.pdf` (2026-07-07) and mirroring
// server/mako/catalog.py (the physics side).
//
// Every module is a segment of ONE spine of ~20 cm-diameter cylinders:
// `kind` is "nose" (front cap), "rear" (stern cap) or "spine" (middle).
// There are no side/top mount points. Each module is neutrally buoyant, so
// mass is DERIVED (fairing volume × flooded fraction), not an option.
//
// This file owns everything the BROWSER needs: palette metadata, the 2D
// side-view silhouette (editor + palette icon, in the document's dark-hull
// style), option schemas, placement math, and the procedural 3D mesh for
// the live viewer (props registered for spin, fins for live deflection).
//
// Conventions: metres, x forward (nose at +x), z up. The 2D editor is a
// side view; local SVG frames are y-down (up = −y).

export const HULL_R = 0.1;        // hull radius (m) — 20 cm diameter
const RHO = 1025;

// ---------------------------------------------------------------------------
// Option schema helpers
// ---------------------------------------------------------------------------

const opt = (key, label, min, max, def, step, unit = "") =>
  ({ key, label, min, max, def, step, unit });

const THRUST_OPTS = [
  opt("max_thrust", "Max thrust", 50, 500, 250, 5, "N"),
  opt("noise", "Thrust noise σ", 0, 10, 1, 0.25, "N"),
];
const GPS_OPT = opt("gps_noise", "GPS noise σ", 0.02, 2, 0.1, 0.01, "m");

/** Fill missing/out-of-range option values from the schema (client-side
 *  mirror of the server clamp — the server re-clamps authoritatively). */
export function cleanOptions(type, options = {}) {
  const out = {};
  for (const o of MODULES[type].options) {
    let v = Number(options[o.key]);
    if (!Number.isFinite(v)) v = o.def;
    out[o.key] = Math.min(o.max, Math.max(o.min, v));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shared 2D fragments (local frame: origin = module centre, y-down).
// ---------------------------------------------------------------------------

const R = HULL_R;

const barrel = (h, cls = "mk-hull") =>
  `<rect x="${-h}" y="${-R}" width="${2 * h}" height="${2 * R}" class="${cls}"/>`;

const collar = (x, w = 0.025) =>
  `<rect x="${x - w / 2}" y="${-R}" width="${w}" height="${2 * R}" class="mk-dark"/>`;

// The shared GPS antenna: short mast + cap on the hull top (y-down: up=−y).
const antenna2d = (x) => `
  <path d="M ${x - 0.014} ${-R} L ${x - 0.024} ${-R - 0.045}
           L ${x + 0.024} ${-R - 0.045} L ${x + 0.014} ${-R} Z" class="mk-dark"/>
  <rect x="${x - 0.032}" y="${-R - 0.062}" width="0.064" height="0.02"
        rx="0.008" class="mk-dark"/>`;

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

export const MODULES = {
  // --- noses -----------------------------------------------------------------
  nose_dvl: {
    kind: "nose", group: "Noses", label: "DVL nose",
    blurb: "Nose cone with a downward DVL in the underside.",
    length: 0.3, fill: 0.85,
    options: [opt("dvl_noise", "DVL noise σ", 0.005, 0.2, 0.02, 0.005, "m/s")],
    shape2d: (o, ctx = {}) => `
      <path d="M -0.15 ${-R} L 0.02 ${-R} C 0.11 ${-R} 0.15 -0.045 0.15 0
               C 0.15 0.05 0.11 ${R} 0.05 ${R}
               L 0.03 ${R} L -0.15 ${R} Z" class="mk-hull"/>
      ${collar(-0.13)}
      <path d="M -0.02 ${R} L 0.05 ${R} L 0.04 ${R + 0.014} L -0.01 ${R + 0.014} Z"
            class="mk-dark"/>`,
  },

  nose_camera: {
    kind: "nose", group: "Noses", label: "Camera nose",
    blurb: "Stereo cameras looking forward (cameras not simulated yet).",
    length: 0.2, fill: 0.85,
    options: [],
    shape2d: (o, ctx = {}) => `
      <path d="M -0.1 ${-R} L 0.0 ${-R} C 0.07 ${-R} 0.1 -0.05 0.1 0
               C 0.1 0.05 0.07 ${R} 0.0 ${R} L -0.1 ${R} Z" class="mk-hull"/>
      ${collar(-0.08)}
      <circle cx="0.055" cy="-0.048" r="0.02" class="mk-glass"/>
      <circle cx="0.055" cy="0.048" r="0.02" class="mk-glass"/>`,
  },

  nose_sonar: {
    kind: "nose", group: "Noses", label: "Sonar nose",
    blurb: "Forward sonar array (sonar not in manta yet).",
    length: 0.5, fill: 0.8,
    options: [],
    shape2d: (o, ctx = {}) => `
      <path d="M -0.25 ${-R} L 0.13 ${-R} C 0.22 ${-R} 0.25 -0.05 0.25 -0.02
               L 0.25 0.02 C 0.25 0.04 0.23 0.05 0.2 0.05
               L 0.02 0.05 C 0.0 0.05 -0.01 0.06 -0.02 ${R}
               L -0.25 ${R} Z" class="mk-hull"/>
      ${collar(-0.23)}
      <path d="M 0.18 0.05 L 0.1 0.13 L 0.05 0.09 L 0.13 0.03 Z" class="mk-dark"/>
      <path d="M -0.05 ${R} A 0.055 0.055 0 0 1 0.02 0.052 L -0.05 0.052 Z"
            class="mk-mid"/>
      ${["-0.1", "-0.06", "-0.02", "0.02", "0.06"].map((x) =>
        `<line x1="${x}" y1="-0.07" x2="${+x + 0.012}" y2="-0.03" class="mk-slit"/>`).join("")}`,
  },

  // --- plain spine modules ------------------------------------------------------
  brain: {
    kind: "spine", group: "Guidance", label: "Brain",
    blurb: "Flight computer: GPS antenna on top + depth sensor.",
    length: 0.2, fill: 1.0,
    options: [GPS_OPT,
              opt("depth_noise", "Depth noise σ", 100, 5000, 500, 50, "Pa")],
    shape2d: (o, ctx = {}) => `
      ${barrel(0.1)}
      ${antenna2d(0)}
      <path d="M -0.045 ${R} A 0.05 0.028 0 0 0 0.045 ${R} Z" class="mk-dark"/>`,
  },

  ins: {
    kind: "spine", group: "Guidance", label: "INS",
    blurb: "Navigation-grade IMU for dead-reckoning.",
    length: 0.3, fill: 1.0,
    options: [
      opt("gyro_noise", "Gyro noise σ", 1e-4, 0.01, 5e-4, 1e-4, "rad/s"),
      opt("gyro_bias", "Gyro bias walk σ", 0, 1e-3, 2e-5, 1e-5, "rad/s"),
      opt("accel_noise", "Accel noise σ", 0.005, 0.5, 0.02, 0.005, "m/s²"),
    ],
    shape2d: (o, ctx = {}) => `
      ${barrel(0.15)}
      <rect x="-0.09" y="-0.075" width="0.035" height="0.012" rx="0.006" class="mk-slit"/>
      <rect x="-0.09" y="0.063" width="0.035" height="0.012" rx="0.006" class="mk-slit"/>`,
  },

  ui: {
    kind: "spine", group: "Structure", label: "UI",
    blurb: "Interface segment with a payload point underneath (payloads later).",
    length: 0.15, fill: 1.0,
    options: [],
    shape2d: (o, ctx = {}) => `
      ${barrel(0.075)}
      <rect x="-0.04" y="${R}" width="0.08" height="0.022" class="mk-pay"/>
      <rect x="-0.02" y="${-R - 0.014}" width="0.014" height="0.014" class="mk-dark"/>
      <rect x="0.005" y="${-R - 0.018}" width="0.012" height="0.018" class="mk-dark"/>`,
  },

  battery: {
    kind: "spine", group: "Structure", label: "Battery",
    blurb: "Energy segment — mass and drag for now, battery model later.",
    length: 0.35, fill: 1.0,
    options: [],
    shape2d: (o, ctx = {}) => `
      ${barrel(0.175, "mk-hull")}
      <rect x="-0.155" y="${-R}" width="0.29" height="${2 * R}" rx="0.02" class="mk-cell"/>
      ${collar(-0.16, 0.03)}${collar(0.155, 0.03)}
      <rect x="-0.145" y="${-R - 0.02}" width="0.016" height="0.02" class="mk-dark"/>`,
  },

  side_scan: {
    kind: "spine", group: "Structure", label: "Side-scan sonar",
    blurb: "Twin side-looking sonars (sonar not in manta yet).",
    length: 0.55, fill: 1.0,
    options: [],
    shape2d: (o, ctx = {}) => `
      ${barrel(0.275)}
      <rect x="-0.19" y="-0.035" width="0.38" height="0.07" rx="0.035" class="mk-mid"/>
      <rect x="-0.175" y="-0.02" width="0.35" height="0.04" rx="0.02" class="mk-window"/>
      ${collar(-0.255)}${collar(0.255)}`,
  },

  // --- transverse thrusters --------------------------------------------------------
  thrust_vert: {
    kind: "spine", group: "Propulsion", label: "Vertical thruster",
    blurb: "Through-hull heave tunnel. Fore/aft placement buys pitch.",
    length: 0.3, fill: 0.7,
    options: THRUST_OPTS,
    // the tunnel reads as the waisted profile, exactly as in the document
    shape2d: (o, ctx = {}) => `
      <path d="M -0.15 ${-R} L -0.09 ${-R} C -0.03 ${-R} -0.05 -0.055 0 -0.055
               C 0.05 -0.055 0.03 ${-R} 0.09 ${-R} L 0.15 ${-R}
               L 0.15 ${R} L 0.09 ${R} C 0.03 ${R} 0.05 0.055 0 0.055
               C -0.05 0.055 -0.03 ${R} -0.09 ${R} L -0.15 ${R} Z" class="mk-hull"/>
      ${collar(-0.115)}${collar(0.115)}
      <circle cx="0" cy="0" r="0.008" class="mk-dark"/>`,
  },

  thrust_horiz: {
    kind: "spine", group: "Propulsion", label: "Horizontal thruster",
    blurb: "Through-hull sway tunnel. Fore/aft placement buys yaw.",
    length: 0.3, fill: 0.7,
    options: THRUST_OPTS,
    // sway tunnel seen face-on: bore + 3-blade prop
    shape2d: (o, ctx = {}) => `
      ${barrel(0.15)}
      ${collar(-0.115)}${collar(0.115)}
      <circle cx="0" cy="0" r="0.078" class="mk-bore"/>
      ${[90, 210, 330].map((a) => `
        <path d="M 0 0 L ${0.07 * Math.cos((a - 24) * Math.PI / 180)} ${0.07 * Math.sin((a - 24) * Math.PI / 180)}
                 A 0.07 0.07 0 0 1 ${0.07 * Math.cos((a + 24) * Math.PI / 180)} ${0.07 * Math.sin((a + 24) * Math.PI / 180)} Z"
              class="mk-blade"/>`).join("")}
      <circle cx="0" cy="0" r="0.022" class="mk-mid"/>`,
  },

  agility_front: {
    kind: "spine", group: "Propulsion", label: "Front agility cluster",
    blurb: "4 pods: heave/roll pair + surge props canted 25° inward.",
    length: 0.25, fill: 0.8,
    options: THRUST_OPTS,
    shape2d: (o, ctx = {}) => `
      ${barrel(0.125)}
      <rect x="-0.02" y="${-R - 0.014}" width="0.05" height="0.014" rx="0.006" class="mk-dark"/>
      <circle cx="0.075" cy="-0.035" r="0.052" class="mk-pod"/>
      <circle cx="0.075" cy="-0.035" r="0.036" class="mk-blade"/>
      <circle cx="0.075" cy="-0.035" r="0.012" class="mk-mid"/>
      <path d="M -0.115 -0.055 A 0.055 0.055 0 0 1 -0.005 -0.055
               L -0.005 0.055 A 0.055 0.055 0 0 1 -0.115 0.055 Z" class="mk-pod"/>
      <ellipse cx="-0.06" cy="-0.055" rx="0.05" ry="0.016" class="mk-dark"/>`,
  },

  agility_rear: {
    kind: "spine", group: "Propulsion", label: "Rear agility cluster",
    blurb: "4 pods: heave/roll pair + surge props canted 25° outward.",
    length: 0.25, fill: 0.8,
    options: THRUST_OPTS,
    shape2d: (o, ctx = {}) => `
      ${barrel(0.125)}
      <rect x="-0.03" y="${-R - 0.014}" width="0.05" height="0.014" rx="0.006" class="mk-dark"/>
      <circle cx="-0.075" cy="-0.035" r="0.052" class="mk-pod"/>
      <circle cx="-0.075" cy="-0.035" r="0.036" class="mk-blade"/>
      <circle cx="-0.075" cy="-0.035" r="0.012" class="mk-mid"/>
      <path d="M 0.005 -0.055 A 0.055 0.055 0 0 1 0.115 -0.055
               L 0.115 0.055 A 0.055 0.055 0 0 1 0.005 0.055 Z" class="mk-pod"/>
      <ellipse cx="0.06" cy="-0.055" rx="0.05" ry="0.016" class="mk-dark"/>`,
  },

  // --- sterns ----------------------------------------------------------------------
  rear_thrust: {
    kind: "rear", group: "Sterns", label: "Rear thruster",
    blurb: "Main prop in a duct + GPS antenna.",
    length: 0.4, fill: 0.7,
    options: [...THRUST_OPTS, GPS_OPT],
    shape2d: (o, ctx = {}) => `
      <path d="M 0.2 ${-R} L 0.02 ${-R} C -0.06 ${-R} -0.1 -0.062 -0.13 -0.058
               L -0.13 0.058 C -0.1 0.062 -0.06 ${R} 0.02 ${R} L 0.2 ${R} Z"
            class="mk-hull"/>
      ${collar(0.17)}
      <path d="M -0.13 -0.075 L -0.2 -0.082 L -0.2 0.082 L -0.13 0.075 Z"
            class="mk-dark"/>
      ${antenna2d(0.13)}`,
  },

  fin_control: {
    kind: "rear", group: "Sterns", label: "Fin control",
    blurb: "Main prop + four actuating fins (+ tail) + GPS antenna.",
    length: 0.4, fill: 0.65,
    options: [...THRUST_OPTS, GPS_OPT,
              opt("fin_area", "Fin area (each)", 0.005, 0.03, 0.011, 0.001, "m²"),
              opt("max_deflection", "Max deflection", 10, 40, 25, 1, "°")],
    shape2d: (o, ctx = {}) => `
      <path d="M 0.2 ${-R} L 0.05 ${-R} C -0.08 ${-R} -0.14 -0.05 -0.17 -0.045
               L -0.17 0.045 C -0.14 0.05 -0.08 ${R} 0.05 ${R} L 0.2 ${R} Z"
            class="mk-hull"/>
      ${collar(0.17)}
      <path d="M -0.05 -0.07 L -0.1 -0.21 L -0.16 -0.21 L -0.16 -0.055 Z" class="mk-fin"/>
      <path d="M -0.05 0.07 L -0.1 0.21 L -0.16 0.21 L -0.16 0.055 Z" class="mk-fin"/>
      <ellipse cx="-0.11" cy="0" rx="0.055" ry="0.014" class="mk-fin"/>
      <ellipse cx="-0.185" cy="0" rx="0.012" ry="0.07" class="mk-blade"/>
      ${antenna2d(0.12)}`,
  },
};

export const GROUPS = ["Noses", "Guidance", "Structure", "Propulsion", "Sterns"];

// ---------------------------------------------------------------------------
// Placement + budget — mirrors server/mako/catalog.py.
// ---------------------------------------------------------------------------

/** Layout of a design: module centres (x, spine centred on 0, nose at +x,
 *  design order = front→back). */
export function layout(design) {
  const lengths = design.spine.map((m) => MODULES[m.type].length);
  const total = lengths.reduce((a, b) => a + b, 0);
  let x = total / 2;
  const centers = lengths.map((L) => { const c = x - L / 2; x -= L; return c; });
  return { centers, lengths, total };
}

/** Derived mass (each module is neutrally buoyant: fairing volume × fill ×
 *  ρ) — display only; the server owns the physics. */
export function massBudget(design) {
  let mass = 0;
  for (const m of design.spine) {
    const t = MODULES[m.type];
    mass += RHO * Math.PI * HULL_R * HULL_R * t.length * t.fill;
  }
  return { mass, netBuoyancyN: 0 };
}

// ---------------------------------------------------------------------------
// 3D meshes — procedural stand-ins for the real CAD, matching the document's
// silhouettes. Each builder gets THREE, cleaned options, materials and
// `ctx = {index}` (the spine index, for input/state names), and returns a
// Group in LOCAL frame (origin = module centre). Registered animations ride
// on group.userData: `spinners` [{mesh, input}] pulse with throttle and
// `fins` [{pivot, rotAxis, sign, slot}] deflect with the live sim state.
// ---------------------------------------------------------------------------

function lathe(THREE, profile, mat, segments = 28) {
  const pts = profile.map(([x, r]) => new THREE.Vector2(Math.max(r, 1e-4), x));
  const m = new THREE.Mesh(new THREE.LatheGeometry(pts, segments), mat);
  m.rotation.z = -Math.PI / 2;   // lathe axis y → x
  m.castShadow = true;
  return m;
}

function cyl(THREE, r, len, mat, axis = "x") {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 24), mat);
  if (axis === "x") m.rotation.z = Math.PI / 2;
  if (axis === "z") m.rotation.x = Math.PI / 2;
  m.castShadow = true;
  return m;
}

function ring(THREE, rIn, rOut, mat) {
  const m = new THREE.Mesh(new THREE.RingGeometry(rIn, rOut, 20), mat.clone());
  m.material.side = THREE.DoubleSide;
  return m;
}

/** An open thruster duct along local +z: outer + inner cylinder walls
 *  (optionally tapered, NO end caps) with a 3-blade prop on a hub inside.
 *  Returns { group, prop } — spin the prop about its local z. */
function thrusterDuct(T, M, { rOut, len, taper = 1.0, wall = 0.15 } = {}) {
  const g = new T.Group();
  const rIn = rOut * (1 - wall);
  const mk = (r) => {
    const geo = new T.CylinderGeometry(r * taper, r, len, 24, 1, true);
    const m = new T.Mesh(geo, M.duct);
    m.rotation.x = Math.PI / 2;          // cylinder y-axis → local z
    m.castShadow = true;
    return m;
  };
  g.add(mk(rOut), mk(rIn));
  // thin annular rims closing the wall thickness (the bore stays open)
  for (const s of [1, -1]) {
    const rr = s > 0 ? [rIn * taper, rOut * taper] : [rIn, rOut];
    const rim = ring(T, rr[0], rr[1], M.dark);
    rim.position.z = s * len / 2;
    g.add(rim);
  }
  // prop: hub + 3 pitched blades, mid-duct
  const prop = new T.Group();
  const hub = new T.Mesh(new T.CylinderGeometry(rIn * 0.22, rIn * 0.22,
                                                len * 0.35, 10), M.mid);
  hub.rotation.x = Math.PI / 2;
  prop.add(hub);
  for (let i = 0; i < 3; i++) {
    const blade = new T.Mesh(
      new T.BoxGeometry(rIn * 0.32, rIn * 0.86, len * 0.16), M.prop);
    const holder = new T.Group();
    blade.position.y = rIn * 0.52;
    blade.rotation.y = 0.6;              // blade pitch
    holder.add(blade);
    holder.rotation.z = (i * 2 * Math.PI) / 3;
    prop.add(holder);
  }
  g.add(prop);
  return { group: g, prop };
}

function boxMesh(THREE, sx, sy, sz, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  m.castShadow = true;
  return m;
}

/** The shared GPS antenna (mast + cap) at local x, on the hull top. */
function antenna3d(T, M, x) {
  const g = new T.Group();
  const mast = new T.Mesh(new T.CylinderGeometry(0.01, 0.017, 0.045, 12), M.dark);
  mast.rotation.x = Math.PI / 2;
  mast.position.set(x, 0, HULL_R + 0.022);
  const cap = new T.Mesh(new T.CylinderGeometry(0.032, 0.032, 0.018, 16), M.dark);
  cap.rotation.x = Math.PI / 2;
  cap.position.set(x, 0, HULL_R + 0.055);
  g.add(mast, cap);
  return g;
}

/** Shared materials. */
export function makeMaterials(THREE) {
  const std = (color, extra = {}) => new THREE.MeshStandardMaterial(
    { color, metalness: 0.25, roughness: 0.5, ...extra });
  return {
    hull: std(0x454c54),
    dark: std(0x2b3036),
    mid: std(0x5a636c),
    fin: std(0x383f46),
    glass: std(0x10141a, { metalness: 0.1, roughness: 0.2 }),
    pay: std(0x27b463),
    prop: std(0xb8c6cc, { metalness: 0.4, roughness: 0.35 }),
    duct: std(0x22272c, { side: 2 /* THREE.DoubleSide */, roughness: 0.7 }),
  };
}

const MESH = {
  nose_dvl: (T, o, M) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.15, R], [0.05, R * 0.97], [0.11, 0.07], [0.15, 0.0]], M.hull));
    const dvl = boxMesh(T, 0.09, 0.06, 0.025, M.dark);
    dvl.position.set(0.02, 0, -R + 0.002);
    g.add(dvl);
    return g;
  },
  nose_camera: (T, o, M) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.1, R], [0.02, R * 0.98], [0.07, 0.07], [0.1, 0.0]], M.hull));
    for (const s of [1, -1]) {
      const lens = cyl(T, 0.02, 0.015, M.glass, "x");
      lens.position.set(0.078, 0, s * 0.048);
      const lens2 = lens.clone();
      lens2.position.set(0.078, s * 0.048, 0);
      g.add(lens, lens2);
    }
    return g;
  },
  nose_sonar: (T, o, M) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.25, R], [0.1, R * 0.95], [0.2, 0.06], [0.25, 0.0]], M.hull));
    const xd = boxMesh(T, 0.07, 0.06, 0.07, M.dark);
    xd.position.set(0.12, 0, -0.1);
    xd.rotation.y = 0.6;
    const dome = new T.Mesh(new T.SphereGeometry(0.05, 16, 10), M.mid);
    dome.scale.set(1, 1, 0.6);
    dome.position.set(-0.02, 0, -R + 0.01);
    g.add(xd, dome);
    return g;
  },
  brain: (T, o, M) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.2, M.hull), antenna3d(T, M, 0));
    const dome = new T.Mesh(new T.SphereGeometry(0.045, 16, 10), M.dark);
    dome.scale.set(1, 1, 0.5);
    dome.position.set(0, 0, -R + 0.005);
    g.add(dome);
    return g;
  },
  ins: (T, o, M) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.3, M.hull));
    return g;
  },
  ui: (T, o, M) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.15, M.hull));
    const pad = boxMesh(T, 0.08, 0.06, 0.02, M.pay);
    pad.position.set(0, 0, -R - 0.005);
    g.add(pad);
    return g;
  },
  battery: (T, o, M) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.35, M.hull));
    for (const x of [-0.16, 0.155]) {
      const c = cyl(T, R + 0.004, 0.03, M.dark);
      c.position.x = x;
      g.add(c);
    }
    return g;
  },
  side_scan: (T, o, M) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.55, M.hull));
    for (const s of [1, -1]) {
      const slab = boxMesh(T, 0.4, 0.024, 0.09, M.mid);
      slab.position.set(0, s * (R - 0.005), 0);
      g.add(slab);
    }
    return g;
  },
  thrust_vert: (T, o, M, ctx) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.15, R], [-0.09, R], [0, 0.062], [0.09, R], [0.15, R]], M.hull));
    const duct = thrusterDuct(T, M, { rOut: 0.062, len: 0.2 });
    g.add(duct.group);                       // tunnel bore along z already
    g.userData.spinners = [{ mesh: duct.prop, axis: [0, 0, 1],
                             input: `thrust_vert${ctx.index}_prop.throttle` }];
    return g;
  },
  thrust_horiz: (T, o, M, ctx) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.3, M.hull));
    const duct = thrusterDuct(T, M, { rOut: 0.08, len: 0.21 });
    duct.group.rotation.x = Math.PI / 2;     // bore along y
    g.add(duct.group);
    g.userData.spinners = [{ mesh: duct.prop, axis: [0, 1, 0],
                             input: `thrust_horiz${ctx.index}_prop.throttle` }];
    return g;
  },
  rear_thrust: (T, o, M, ctx) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.13, 0.058], [-0.02, 0.088], [0.06, R], [0.2, R]], M.hull));
    const duct = thrusterDuct(T, M, { rOut: 0.09, len: 0.1, taper: 0.91 });
    duct.group.rotation.y = -Math.PI / 2;    // bore along x, narrow end aft
    duct.group.position.x = -0.165;
    g.add(duct.group, antenna3d(T, M, 0.13));
    g.userData.spinners = [{ mesh: duct.prop, axis: [1, 0, 0],
                             input: `rear_thrust${ctx.index}_prop.throttle` }];
    return g;
  },
  fin_control: (T, o, M, ctx) => {
    const g = new T.Group();
    g.add(lathe(T, [[-0.17, 0.045], [-0.05, 0.08], [0.06, R], [0.2, R]], M.hull));
    const duct = thrusterDuct(T, M, { rOut: 0.068, len: 0.085, taper: 0.9 });
    duct.group.rotation.y = -Math.PI / 2;    // narrow end aft
    duct.group.position.x = -0.19;
    g.add(duct.group, antenna3d(T, M, 0.12));
    g.userData.spinners = [{ mesh: duct.prop, axis: [1, 0, 0],
                             input: `fin_control${ctx.index}_prop.throttle` }];

    // Four identical + fins, each in a pivot Group at its hinge so the
    // viewer can deflect it live from the sim's `.deflection` state.
    const span = 0.13, chord = Math.max(0.07, Math.sqrt(o.fin_area));
    const fins = [];
    const mk = (name, pos, vertical) => {
      const pivot = new T.Group();
      pivot.position.set(-0.12, pos[0], pos[1]);
      const fin = vertical
        ? boxMesh(T, chord, 0.012, span, M.fin)
        : boxMesh(T, chord, span, 0.012, M.fin);
      // grow outward from the hull, swept slightly back
      if (vertical) fin.position.z = Math.sign(pos[1]) * span / 2;
      else fin.position.y = Math.sign(pos[0]) * span / 2;
      fin.position.x = -0.02;
      pivot.add(fin);
      g.add(pivot);
      fins.push({ pivot, rotAxis: vertical ? "z" : "y", sign: 1,
                  slot: `mako.fin_control${ctx.index}_fin${name}.deflection` });
    };
    mk("t", [0, +0.055], true);
    mk("b", [0, -0.055], true);
    mk("l", [+0.055, 0], false);
    mk("r", [-0.055, 0], false);
    g.userData.fins = fins;
    return g;
  },
};

// Agility clusters share a builder: per side, a sponson plate carrying a
// dome-capped vertical pod and a 25°-canted surge pod. As in the document,
// the canted pods sit at the FRONT corners of the front module (vertical
// pods behind them) and at the REAR corners of the rear module — staggered
// in x and pushed clear of the hull so nothing interpenetrates.
function agilityMesh(cantSign, type) {
  const cantedX = -cantSign * 0.05;   // front module: forward; rear: aft
  const vertX = cantSign * 0.06;
  return (T, o, M, ctx) => {
    const g = new T.Group();
    g.add(cyl(T, R, 0.25, M.hull));
    const spinners = [];
    for (const [tag, side] of [["l", 1], ["r", -1]]) {
      const sponson = boxMesh(T, 0.21, 0.045, 0.09, M.hull);
      sponson.position.set(0, side * 0.115, 0);
      g.add(sponson);

      // vertical pod: an open duct with the prop visible in the bore
      const vDuct = thrusterDuct(T, M, { rOut: 0.042, len: 0.1 });
      vDuct.group.position.set(vertX, side * 0.165, 0);
      g.add(vDuct.group);
      spinners.push({ mesh: vDuct.prop, axis: [0, 0, 1],
                      input: `${type}${ctx.index}_v${tag}.throttle` });

      // canted surge pod: open duct along +x inside a pivot group, yawed
      // by the cant angle (matches the physics axis exactly)
      const pod = new T.Group();
      const fDuct = thrusterDuct(T, M, { rOut: 0.04, len: 0.12, taper: 0.92 });
      fDuct.group.rotation.y = Math.PI / 2;  // bore along +x, taper forward
      pod.add(fDuct.group);
      pod.position.set(cantedX, side * 0.19, 0);
      pod.rotation.z = cantSign * side * (25 * Math.PI / 180);
      g.add(pod);
      spinners.push({ mesh: fDuct.prop,
                      axis: [Math.cos(25 * Math.PI / 180),
                             cantSign * side * Math.sin(25 * Math.PI / 180), 0],
                      input: `${type}${ctx.index}_f${tag}.throttle` });
    }
    g.userData.spinners = spinners;
    return g;
  };
}
MESH.agility_front = agilityMesh(-1, "agility_front");
MESH.agility_rear = agilityMesh(+1, "agility_rear");

/** Build the whole craft as a Three.js Group in craft frame. Returns
 *  { group, spinners, fins }: spinners pulse with commanded throttle
 *  ([{mesh, input}], input part-local), fins deflect with the live state
 *  ([{pivot, rotAxis, sign, slot}], slot a full state-slot name). */
export function makeCraftMesh(THREE, design) {
  const M = makeMaterials(THREE);
  const { centers } = layout(design);
  const group = new THREE.Group();
  const spinners = [], fins = [];

  design.spine.forEach((m, i) => {
    const opts = cleanOptions(m.type, m.options);
    const g = MESH[m.type](THREE, opts, M, { index: i });
    g.position.x = centers[i];
    group.add(g);
    spinners.push(...(g.userData.spinners || []));
    fins.push(...(g.userData.fins || []));
  });
  return { group, spinners, fins };
}

// ---------------------------------------------------------------------------
// Starter design — the document's natural fit: full 6-DOF authority and
// every sensor path (DVL, GPS ×2, depth, INS).
// ---------------------------------------------------------------------------

export function starterDesign() {
  return {
    version: 2,
    name: "mako one",
    spine: [
      { type: "nose_dvl", options: {} },
      { type: "agility_front", options: {} },
      { type: "brain", options: {} },
      { type: "ins", options: {} },
      { type: "battery", options: {} },
      { type: "agility_rear", options: {} },
      { type: "fin_control", options: {} },
    ],
    spawn: { x: 0, y: 0, depth: 2, heading_deg: 0,
             wave_amp: 0.25, wave_len: 18, visibility: 55 },
  };
}
