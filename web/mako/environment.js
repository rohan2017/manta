// The drive view's ocean environment — everything that is NOT the craft:
// the waving sea surface (the exact η the physics uses, from meta.waves),
// an ENDLESS procedurally generated seabed (chunked: relief + rocks +
// coral reefs + kelp clusters + sea-grass meadows stream in around the
// craft and unload far away, all behind the fog), a wandering fish school,
// the visibility (murk) control and the frontend-only light payload.
// All primitives, no assets.
//
// Coordinates are scene-ENU (z up, surface ≈ 0, seabed at meta.seabed_z);
// the viewer positions the craft in the same frame. Chunk geometry is baked
// in CHUNK-LOCAL coordinates with the group positioned at the chunk origin,
// so float32 vertex buffers stay small no matter how far you drive.

import { makeFauna } from "./fauna.js";

// deterministic PRNG so the reef looks the same every dive
function makeRand(seed) {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

// cheap smooth 2D value noise (few octaves, good enough for relief)
function makeNoise2(rand) {
  const N = 64, grid = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) grid[i] = rand() * 2 - 1;
  const at = (ix, iy) => grid[((iy % N + N) % N) * N + ((ix % N + N) % N)];
  const sm = (t) => t * t * (3 - 2 * t);
  return (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = sm(x - ix), fy = sm(y - iy);
    const a = at(ix, iy), b = at(ix + 1, iy);
    const c = at(ix, iy + 1), d = at(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
  };
}

export function makeEnvironment(THREE, scene, { meta }) {
  const rand = makeRand(20260714);
  const noise = makeNoise2(rand);
  const SEABED = meta.seabed_z ?? -30;
  const dummy = new THREE.Object3D();
  const uTime = { value: 0 };                  // shared by the sway shaders
  const F = (x) => Number(x).toExponential(6); // number → GLSL literal

  // --- the wave surface (matches the sim exactly) ---------------------------
  // η = A·cos(k·ξ − ω·t) with ξ = p_planet·dir. The scene co-rotates with
  // the planet, so in scene coords: ξ = dirScene·p + phase0 (time-invariant
  // form). dirScene = R_psᵀ·dir, phase0 = dir·anchor.
  const wv = meta.waves ?? { amplitude: 0, wavelength: 20, k: 0.3, omega: 1 };
  const sc = meta.scene;
  const Rps = sc.R_planet_from_scene;
  const dirP = wv.dir_planet ?? [1, 0, 0];
  const dirScene = [0, 1, 2].map((j) =>
    Rps[0][j] * dirP[0] + Rps[1][j] * dirP[1] + Rps[2][j] * dirP[2]);
  const phase0 = dirP[0] * sc.anchor_planet[0]
    + dirP[1] * sc.anchor_planet[1] + dirP[2] * sc.anchor_planet[2];
  const surfaceHeight = (x, y, t) =>
    wv.amplitude * Math.cos(
      wv.k * (dirScene[0] * x + dirScene[1] * y + phase0) - wv.omega * t);

  const SURF_SPAN = 320, SURF_SEG = 72;
  const surfGeo = new THREE.PlaneGeometry(SURF_SPAN, SURF_SPAN,
                                          SURF_SEG, SURF_SEG);
  const surfMat = new THREE.MeshStandardMaterial({
    color: 0x2f7fc2, transparent: true, opacity: 0.5, vertexColors: true,
    roughness: 0.3, metalness: 0.12, side: THREE.DoubleSide });
  const surface = new THREE.Mesh(surfGeo, surfMat);
  scene.add(surface);
  const surfPos = surfGeo.attributes.position;
  const surfCol = new Float32Array(surfPos.count * 3).fill(1);
  surfGeo.setAttribute("color", new THREE.BufferAttribute(surfCol, 3));

  function updateSurface(t, cx, cy) {
    // follow the craft snapped to the vertex grid so the crests don't swim
    const step = SURF_SPAN / SURF_SEG;
    const ox = Math.round(cx / step) * step, oy = Math.round(cy / step) * step;
    surface.position.set(ox, oy, 0);
    for (let i = 0; i < surfPos.count; i++) {
      const wx = ox + surfPos.getX(i), wy = oy + surfPos.getY(i);
      // + small ambient ripples (visual only, NOT in surfaceHeight — the
      // physics η stays pure) so a flat sea doesn't read as glass
      const rip = 0.075 * Math.sin(0.52 * wx + 1.15 * t)
                        * Math.sin(0.6 * wy - 0.95 * t)
                + 0.04 * Math.sin(0.33 * (wx + wy) + 0.7 * t);
      surfPos.setZ(i, surfaceHeight(wx, wy, t) + rip);
      // travelling brightness shimmer so the water reads as textured,
      // not glass (multiplies the base colour)
      const sh = 1 + 0.08 * Math.sin(0.41 * wx - 0.9 * t)
                        * Math.sin(0.47 * wy + 0.75 * t)
               + 0.04 * Math.sin(0.9 * (wx - wy) + 1.3 * t);
      surfCol[i * 3] = surfCol[i * 3 + 1] = surfCol[i * 3 + 2] = sh;
    }
    surfPos.needsUpdate = true;
    surfGeo.attributes.color.needsUpdate = true;
    surfGeo.computeVertexNormals();
  }

  // --- sun glints on the surface (visible from above) -----------------------
  const glintCv = document.createElement("canvas");
  glintCv.width = glintCv.height = 16;
  {
    const g = glintCv.getContext("2d");
    const gr = g.createRadialGradient(8, 8, 1, 8, 8, 7.5);
    gr.addColorStop(0, "rgba(255,255,255,1)");
    gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, 16, 16);
  }
  const SPK_N = 420;
  const spkPos = new Float32Array(SPK_N * 3);
  const spkCol = new Float32Array(SPK_N * 4);
  const spkOff = new Float32Array(SPK_N * 2);
  const spkPh = new Float32Array(SPK_N);
  const spkGeo = new THREE.BufferGeometry();
  spkGeo.setAttribute("position", new THREE.BufferAttribute(spkPos, 3));
  spkGeo.setAttribute("color", new THREE.BufferAttribute(spkCol, 4));
  const sparkles = new THREE.Points(spkGeo, new THREE.PointsMaterial({
    size: 0.25, map: new THREE.CanvasTexture(glintCv), vertexColors: true,
    transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending }));
  sparkles.frustumCulled = false;
  sparkles.visible = false;
  scene.add(sparkles);
  let spkBx = 0, spkBy = 0;
  function scatterSparkles(bx, by) {
    spkBx = bx; spkBy = by;
    sparkles.position.set(bx, by, 0);
    for (let i = 0; i < SPK_N; i++) {
      spkOff[i * 2] = (Math.random() - 0.5) * 240;
      spkOff[i * 2 + 1] = (Math.random() - 0.5) * 240;
      spkPh[i] = Math.random() * Math.PI * 2;
    }
  }
  scatterSparkles(0, 0);
  function updateSparkles(t, cam) {
    if (!sparkles.visible) return;
    if (Math.hypot(cam.x - spkBx, cam.y - spkBy) > 100)
      scatterSparkles(cam.x, cam.y);
    for (let i = 0; i < SPK_N; i++) {
      const wx = spkBx + spkOff[i * 2], wy = spkBy + spkOff[i * 2 + 1];
      spkPos[i * 3] = spkOff[i * 2];
      spkPos[i * 3 + 1] = spkOff[i * 2 + 1];
      spkPos[i * 3 + 2] = surfaceHeight(wx, wy, t) + 0.06;
      // sharp twinkle: mostly dark, brief bright glints
      const tw = Math.sin(t * (1.3 + (i % 7) * 0.37) + spkPh[i]);
      const a = Math.max(0, tw) ** 6;
      spkCol[i * 4] = 1; spkCol[i * 4 + 1] = 0.98; spkCol[i * 4 + 2] = 0.9;
      spkCol[i * 4 + 3] = a;
    }
    spkGeo.attributes.position.needsUpdate = true;
    spkGeo.attributes.color.needsUpdate = true;
  }

  // --- marine snow ------------------------------------------------------------
  // Pixel-scale motes hanging in the water column around the camera — the
  // parallax "starfield" that sells motion. Fixed pixel size (no
  // attenuation), slow drift, fade-in on (re)spawn; a mote drifting out of
  // range respawns somewhere in the sphere around the camera.
  // dedicated sprite: mostly-solid dot (the glint sprite's soft gradient
  // sampled at a few pixels is nearly invisible)
  const snowCv = document.createElement("canvas");
  snowCv.width = snowCv.height = 16;
  {
    const g = snowCv.getContext("2d");
    const gr = g.createRadialGradient(8, 8, 4, 8, 8, 7.5);
    gr.addColorStop(0, "rgba(255,255,255,1)");
    gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, 16, 16);
  }
  const SNOW_N = 800, SNOW_R = 42;
  const snowPos = new Float32Array(SNOW_N * 3);   // offsets from snowB
  const snowCol = new Float32Array(SNOW_N * 4);
  const snowVel = new Float32Array(SNOW_N * 3);
  const snowAge = new Float32Array(SNOW_N);
  const snowGeo = new THREE.BufferGeometry();
  snowGeo.setAttribute("position", new THREE.BufferAttribute(snowPos, 3));
  snowGeo.setAttribute("color", new THREE.BufferAttribute(snowCol, 4));
  // size is in DEVICE pixels with sizeAttenuation off (the page renders at
  // devicePixelRatio ≤ 2) — 4 ≈ a 2-px CSS speck
  const snow = new THREE.Points(snowGeo, new THREE.PointsMaterial({
    size: 4, sizeAttenuation: false, map: new THREE.CanvasTexture(snowCv),
    vertexColors: true, transparent: true, depthWrite: false }));
  snow.frustumCulled = false;
  scene.add(snow);
  let snowBx = 0, snowBy = 0, snowInit = false;
  function respawnMote(i, cx, cy, cz, prime) {
    // uniform-ish in a sphere around the camera, clamped to the water column
    const az = Math.random() * Math.PI * 2, el = Math.random() * 2 - 1;
    const r = SNOW_R * Math.cbrt(Math.random());
    const h = r * Math.sqrt(1 - el * el);
    snowPos[i * 3] = cx - snowBx + Math.cos(az) * h;
    snowPos[i * 3 + 1] = cy - snowBy + Math.sin(az) * h;
    snowPos[i * 3 + 2] = Math.min(-0.3, Math.max(SEABED + 0.2, cz + r * el));
    for (let k = 0; k < 3; k++)
      snowVel[i * 3 + k] = (Math.random() - 0.5) * 0.04;
    // prime: initial fill starts at random ages so the first frame isn't
    // a synchronized fade-in
    snowAge[i] = prime ? Math.random() * 2 : 0;
  }
  function updateSnow(dt, cam) {
    if (!snowInit) {
      snowInit = true;
      snowBx = cam.x; snowBy = cam.y;
      snow.position.set(snowBx, snowBy, 0);
      for (let i = 0; i < SNOW_N; i++)
        respawnMote(i, cam.x, cam.y, cam.z, true);
    }
    // rebase the float32 offsets long before they lose precision
    if (Math.hypot(cam.x - snowBx, cam.y - snowBy) > 300) {
      const dx = cam.x - snowBx, dy = cam.y - snowBy;
      snowBx = cam.x; snowBy = cam.y;
      snow.position.set(snowBx, snowBy, 0);
      for (let i = 0; i < SNOW_N; i++) {
        snowPos[i * 3] -= dx;
        snowPos[i * 3 + 1] -= dy;
      }
    }
    for (let i = 0; i < SNOW_N; i++) {
      const wx = snowBx + snowPos[i * 3], wy = snowBy + snowPos[i * 3 + 1];
      const dxc = wx - cam.x, dyc = wy - cam.y,
            dzc = snowPos[i * 3 + 2] - cam.z;
      if (dxc * dxc + dyc * dyc + dzc * dzc > SNOW_R * SNOW_R * 1.21)
        respawnMote(i, cam.x, cam.y, cam.z, false);
      snowAge[i] += dt;
      snowPos[i * 3] += snowVel[i * 3] * dt;
      snowPos[i * 3 + 1] += snowVel[i * 3 + 1] * dt;
      snowPos[i * 3 + 2] += snowVel[i * 3 + 2] * dt;
      const a = Math.min(1, snowAge[i] / 1.4);    // fade in, then hold
      snowCol[i * 4] = 0.8;
      snowCol[i * 4 + 1] = 0.88;
      snowCol[i * 4 + 2] = 0.94;
      snowCol[i * 4 + 3] = 0.1 * a;
    }
    snowGeo.attributes.position.needsUpdate = true;
    snowGeo.attributes.color.needsUpdate = true;
  }

  // --- seabed relief ---------------------------------------------------------
  // Visual only — the physics floor is flat at SEABED (colliders). Keep the
  // relief mostly DIPS (−1.6 m) with small bumps (+0.15 m) so a hull
  // resting on the contact plane never visually sinks into a mound.
  const bedHeight = (x, y) => {
    const n = 0.7 * noise(x * 0.045, y * 0.045) + 0.3 * noise(x * 0.16, y * 0.16);
    return SEABED + (n < 0 ? n * 1.6 : n * 0.15);
  };

  // ==========================================================================
  // Chunked, endless world
  // ==========================================================================
  // The seabed and everything on it stream in as CHUNK×CHUNK tiles keyed by
  // integer chunk coordinates. Content is seeded per chunk, with the seed
  // wrapping at the planet's circumference — so circumnavigating brings you
  // back to the same chunks (the flat scene frame is the approximation; the
  // physics really does run on the round planet). Load/unload happens well
  // beyond the fog's sight range, so nothing pops in view.
  const CHUNK = 64, BED_SEG = 16;
  const R_LOAD = 230, R_UNLOAD = 268;
  const RPL = Math.hypot(sc.anchor_planet[0], sc.anchor_planet[1],
                         sc.anchor_planet[2]);
  const NWRAP = Math.max(1, Math.round(2 * Math.PI * RPL / CHUNK));
  function chunkSeed(i, j) {
    const wi = ((i % NWRAP) + NWRAP) % NWRAP;
    const wj = ((j % NWRAP) + NWRAP) % NWRAP;
    let h = (Math.imul(wi, 374761393) + Math.imul(wj, 668265263)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return ((h >>> 0) % 2147483646) + 1;
  }

  // --- shared geometry pools -------------------------------------------------
  const NI = (g) => (g.index ? g.toNonIndexed() : g);

  // rocks: convex polyhedra with consistent radial jitter per shared corner
  // (duplicated soup verts get the same factor → no tears, stays convex-ish),
  // flat facets from non-indexed normals
  function makeRockGeo(detail) {
    const geo = NI(rand() < 0.5
      ? new THREE.IcosahedronGeometry(1, detail)
      : new THREE.DodecahedronGeometry(1, detail));
    const p = geo.attributes.position;
    const fac = new Map();
    for (let i = 0; i < p.count; i++) {
      const key = `${p.getX(i)},${p.getY(i)},${p.getZ(i)}`;
      if (!fac.has(key)) fac.set(key, 0.74 + rand() * 0.48);
      const s = fac.get(key);
      p.setXYZ(i, p.getX(i) * s, p.getY(i) * s, p.getZ(i) * s);
    }
    geo.computeVertexNormals();
    return geo;
  }
  const rockLo = [], rockHi = [];
  for (let i = 0; i < 6; i++) rockLo.push(makeRockGeo(0));
  for (let i = 0; i < 6; i++) rockHi.push(makeRockGeo(1));

  // coral prims (all non-indexed, unit-ish sized, y-up where cylindrical)
  const prims = {
    fan: (() => {                              // standing half-disc
      const g = NI(new THREE.CircleGeometry(0.5, 9, 0, Math.PI));
      g.rotateX(Math.PI / 2);
      return g;
    })(),
    dome: (() => {                             // brain-coral hemisphere
      const g = new THREE.SphereGeometry(0.5, 12, 8, 0, Math.PI * 2,
                                         0, Math.PI / 2);
      const p = g.attributes.position;
      const fac = new Map();
      for (let i = 0; i < p.count; i++) {
        const key = `${p.getX(i)},${p.getY(i)},${p.getZ(i)}`;
        if (!fac.has(key)) fac.set(key, 0.92 + rand() * 0.16);
        const s = fac.get(key);
        p.setXYZ(i, p.getX(i) * s, p.getY(i) * s, p.getZ(i) * s);
      }
      g.rotateX(Math.PI / 2);                  // pole → +z, base ring at z=0
      g.scale(1, 1, 0.62);
      g.computeVertexNormals();
      return NI(g);
    })(),
    finger: NI(new THREE.CylinderGeometry(0.085, 0.13, 1, 7)),
    cap: (() => {
      const g = new THREE.SphereGeometry(0.085, 7, 4, 0, Math.PI * 2,
                                         0, Math.PI / 2);
      g.rotateX(Math.PI / 2);
      return NI(g);
    })(),
    branch: NI(new THREE.CylinderGeometry(0.035, 0.06, 1, 5)),
    tip: NI(new THREE.SphereGeometry(0.07, 6, 5)),
    leaf: (() => {                             // kelp leaf, base at y=0
      const g = new THREE.PlaneGeometry(0.14, 0.75, 1, 2);
      g.translate(0, 0.375, 0);
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++)
        p.setZ(i, 0.08 * Math.sin(Math.PI * p.getY(i) / 0.75));
      g.computeVertexNormals();
      return NI(g);
    })(),
    nodule: NI(new THREE.OctahedronGeometry(0.05, 0)),
  };

  // grass blade: unit height along +z, tapered, slightly bowed; the sway
  // shader bends it by (local z)² so the base stays rooted
  const bladeGeo = (() => {
    const g = new THREE.PlaneGeometry(0.055, 1, 1, 3);
    g.translate(0, 0.5, 0);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i);
      p.setX(i, p.getX(i) * (1 - 0.75 * y));
      p.setZ(i, 0.18 * y * y);
    }
    g.rotateX(Math.PI / 2);
    g.computeVertexNormals();
    return g;
  })();

  // --- shared materials ------------------------------------------------------
  const bedMat = new THREE.MeshStandardMaterial({ vertexColors: true,
                                                  roughness: 1.0 });
  const rockMat = new THREE.MeshStandardMaterial({ vertexColors: true,
                                                   roughness: 0.95 });
  const coralMat = new THREE.MeshStandardMaterial({ vertexColors: true,
    roughness: 0.85, side: THREE.DoubleSide });
  // kelp sways in the vertex shader: the water's own orbital displacement
  // A·e^{k·z} (dies off with depth exactly like the physics) + a gentle
  // ambient current, both scaled by aBend.x = fraction along the stalk so
  // the holdfast stays rooted. aBend.y = per-plant phase.
  const kelpMat = new THREE.MeshStandardMaterial({ vertexColors: true,
    roughness: 0.9, side: THREE.DoubleSide });
  kelpMat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime;
    sh.vertexShader = "attribute vec2 aBend;\nuniform float uTime;\n"
      + sh.vertexShader.replace("#include <begin_vertex>", `
        #include <begin_vertex>
        {
          float f_ = aBend.x;
          vec4 wp_ = modelMatrix * vec4(position, 1.0);
          float orb_ = ${F(wv.amplitude)}
            * exp(min(0.0, ${F(wv.k)} * wp_.z));
          float ph_ = ${F(wv.k)} * (${F(dirScene[0])} * wp_.x
            + ${F(dirScene[1])} * wp_.y + ${F(phase0)})
            - ${F(wv.omega)} * uTime;
          transformed.xy +=
            vec2(${F(dirScene[0])}, ${F(dirScene[1])})
              * (sin(ph_ + f_ * 0.8) * orb_ * f_)
            + vec2(cos(aBend.y), sin(aBend.y))
              * (sin(uTime * 0.45 + aBend.y * 7.0)
                 * (0.04 + 0.10 * f_ * f_));
        }`);
  };
  const grassMat = new THREE.MeshStandardMaterial({ color: 0xffffff,
    roughness: 0.95, side: THREE.DoubleSide });
  grassMat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime;
    sh.vertexShader = "uniform float uTime;\n"
      + sh.vertexShader.replace("#include <begin_vertex>", `
        #include <begin_vertex>
        {
          float f_ = clamp(position.z, 0.0, 1.0);
          float ph_ = instanceMatrix[3].x * 1.7 + instanceMatrix[3].y * 2.3;
          transformed.xy += vec2(sin(uTime * 0.6 + ph_),
                                 cos(uTime * 0.53 + ph_ * 1.3))
            * (f_ * f_ * 0.05);
        }`);
  };

  // --- merge helpers ---------------------------------------------------------
  const _m3 = new THREE.Matrix3();
  const _v = new THREE.Vector3(), _n = new THREE.Vector3();
  const _q = new THREE.Quaternion(), _Y = new THREE.Vector3(0, 1, 0);
  function pushGeo(out, geo, m, color, vary = 0, rnd = null, bend = null) {
    const p = geo.attributes.position, nr = geo.attributes.normal;
    _m3.getNormalMatrix(m);
    let f = 1;
    for (let i = 0; i < p.count; i++) {
      if (vary && rnd && i % 3 === 0) f = 1 - vary * (0.5 - rnd());
      _v.fromBufferAttribute(p, i).applyMatrix4(m);
      _n.fromBufferAttribute(nr, i).applyMatrix3(_m3).normalize();
      out.pos.push(_v.x, _v.y, _v.z);
      out.nrm.push(_n.x, _n.y, _n.z);
      out.col.push(Math.min(1, color.r * f), Math.min(1, color.g * f),
                   Math.min(1, color.b * f));
      if (out.bend) out.bend.push(bend ? bend[0] : 0, bend ? bend[1] : 0);
    }
  }
  function bake(out, material, { shadow = false, sphere = 0 } = {}) {
    if (!out.pos.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(out.pos), 3));
    g.setAttribute("normal",
      new THREE.BufferAttribute(new Float32Array(out.nrm), 3));
    g.setAttribute("color",
      new THREE.BufferAttribute(new Float32Array(out.col), 3));
    if (out.bend)
      g.setAttribute("aBend",
        new THREE.BufferAttribute(new Float32Array(out.bend), 2));
    g.computeBoundingSphere();
    if (sphere) g.boundingSphere.radius += sphere;
    const mesh = new THREE.Mesh(g, material);
    mesh.castShadow = shadow;
    return mesh;
  }

  // --- coral builders --------------------------------------------------------
  // Deliberate simple types: fan, brain, columnar, staghorn. Per-coral
  // scale 60–150 % and a colour from a fixed coral palette; per-face tint
  // jitter breaks up the flat look.
  const CORAL_COLORS = [0xd96f52, 0xe8955e, 0xc95f7d, 0x9d6bb5, 0xe3c184,
                        0x5fb3a1].map((c) => new THREE.Color(c));
  function addFan(out, rnd, x, y, z, s, color) {
    const n = 1 + Math.floor(rnd() * 3);
    for (let i = 0; i < n; i++) {
      dummy.position.set(x + (rnd() - 0.5) * 0.35 * s,
                         y + (rnd() - 0.5) * 0.35 * s, z);
      dummy.rotation.set((rnd() - 0.5) * 0.25, (rnd() - 0.5) * 0.25,
                         rnd() * Math.PI * 2);
      const k = s * (0.8 + rnd() * 0.7);
      dummy.scale.set(k, k, k);
      dummy.updateMatrix();
      pushGeo(out, prims.fan, dummy.matrix, color, 0.25, rnd);
    }
  }
  function addBrain(out, rnd, x, y, z, s, color) {
    dummy.position.set(x, y, z - 0.03);
    dummy.rotation.set(0, 0, rnd() * Math.PI * 2);
    const k = s * (0.7 + rnd() * 0.5);
    dummy.scale.set(k * (0.9 + rnd() * 0.3), k * (0.9 + rnd() * 0.3), k);
    dummy.updateMatrix();
    pushGeo(out, prims.dome, dummy.matrix, color, 0.3, rnd);
  }
  function addColumnar(out, rnd, x, y, z, s, color) {
    const n = 3 + Math.floor(rnd() * 4);
    for (let i = 0; i < n; i++) {
      const px = x + (rnd() - 0.5) * 0.7 * s, py = y + (rnd() - 0.5) * 0.7 * s;
      const h = s * (0.35 + rnd() * 0.65), w = s * (0.8 + rnd() * 0.5);
      dummy.position.set(px, py, z + h / 2);
      dummy.rotation.set(Math.PI / 2, 0, 0);   // cylinder y → z
      dummy.scale.set(w, h, w);
      dummy.updateMatrix();
      pushGeo(out, prims.finger, dummy.matrix, color, 0.2, rnd);
      dummy.position.set(px, py, z + h);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(w, w, w * 0.8);
      dummy.updateMatrix();
      pushGeo(out, prims.cap, dummy.matrix, color, 0.2, rnd);
    }
  }
  function addStag(out, rnd, x, y, z, s, color) {
    const nb = 4 + Math.floor(rnd() * 4);
    for (let i = 0; i < nb; i++) {
      const az = rnd() * Math.PI * 2;
      const tilt = i === 0 ? 0.08 : 0.35 + rnd() * 0.55;
      _v.set(Math.sin(tilt) * Math.cos(az), Math.sin(tilt) * Math.sin(az),
             Math.cos(tilt));
      const L = s * (0.4 + rnd() * 0.6);
      _q.setFromUnitVectors(_Y, _v);
      dummy.rotation.setFromQuaternion(_q);
      dummy.position.set(x + _v.x * L / 2, y + _v.y * L / 2, z + _v.z * L / 2);
      dummy.scale.set(s, L, s);
      dummy.updateMatrix();
      pushGeo(out, prims.branch, dummy.matrix, color, 0.2, rnd);
      dummy.position.set(x + _v.x * L, y + _v.y * L, z + _v.z * L);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      pushGeo(out, prims.tip, dummy.matrix, color, 0.2, rnd);
    }
  }
  const CORAL_KINDS = [addFan, addBrain, addColumnar, addStag];

  // --- sessile extras: sea urchins + abalone ----------------------------------
  // Built once as colourless prims, merged into the chunk coral mesh with a
  // per-individual colour (purple/red urchins, brick-red abalone shells).
  const urchinGeo = (() => {
    const out = { pos: [], nrm: [], col: [] };
    const white = new THREE.Color(1, 1, 1);
    dummy.position.set(0, 0, 0);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.setScalar(1);
    dummy.updateMatrix();
    pushGeo(out, NI(new THREE.IcosahedronGeometry(0.5, 0)), dummy.matrix,
            white);
    const spike = NI(new THREE.ConeGeometry(0.055, 1, 3, 1, true));
    spike.translate(0, 0.5, 0);                // base at origin, along +y
    for (let i = 0; i < 18; i++) {             // golden-angle spiral of spikes
      const az = i * 2.399963;
      const el = Math.acos(1 - 2 * ((i + 0.5) / 18));
      _v.set(Math.sin(el) * Math.cos(az), Math.sin(el) * Math.sin(az),
             Math.cos(el));
      if (_v.z < -0.25) continue;              // none into the ground
      _q.setFromUnitVectors(_Y, _v);
      dummy.rotation.setFromQuaternion(_q);
      dummy.position.set(_v.x * 0.3, _v.y * 0.3, _v.z * 0.3);
      dummy.scale.set(1, 0.85 + (i % 4) * 0.1, 1);
      dummy.updateMatrix();
      pushGeo(out, spike, dummy.matrix, white);
    }
    spike.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(out.pos), 3));
    g.setAttribute("normal",
      new THREE.BufferAttribute(new Float32Array(out.nrm), 3));
    return g;
  })();
  const abaloneGeo = (() => {                  // low oval dome shell
    const g = new THREE.SphereGeometry(0.5, 8, 4, 0, Math.PI * 2,
                                       0, Math.PI / 2);
    g.rotateX(Math.PI / 2);                    // dome up, base at z = 0
    g.scale(1, 0.72, 0.4);
    g.computeVertexNormals();
    return NI(g);
  })();
  const URCHIN_COLS = [new THREE.Color(0x5a2d6e),   // purple urchin
                       new THREE.Color(0x8a3025)];  // red urchin
  const ABALONE_COL = new THREE.Color(0x7c4a35);
  function addUrchins(out, rnd, cx0, cy0, ox, oy, n, r0, r1) {
    for (let i = 0; i < n; i++) {
      const az = rnd() * Math.PI * 2, rr = r0 + rnd() * (r1 - r0);
      const x = cx0 + Math.cos(az) * rr, y = cy0 + Math.sin(az) * rr;
      const s = 0.08 + rnd() * 0.06;
      dummy.position.set(x, y, bedHeight(ox + x, oy + y) + s * 0.3);
      dummy.rotation.set(0, 0, rnd() * 6.28);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      pushGeo(out, urchinGeo, dummy.matrix,
              URCHIN_COLS[rnd() < 0.3 ? 1 : 0], 0.2, rnd);
    }
  }

  // --- kelp builder ----------------------------------------------------------
  // A proper plant: a tapered stem tube riding a mostly-upward random walk
  // (gentle spiral/curve), leaves + pneumatocyst nodules off the stem. Sway
  // is GPU-side via aBend = (height fraction, plant phase).
  function addKelp(out, rnd, bx, by, bz, h) {
    const stemCol = new THREE.Color(0x5a6b35);
    const leafCol = new THREE.Color(0x3f7034);
    const nodCol = new THREE.Color(0x93a84e);
    const phase = rnd() * Math.PI * 2;
    const nSeg = Math.max(6, Math.round(h / 1.05));
    const step = h / nSeg;
    let lx = 0, ly = 0;
    const pts = [[bx, by, bz]];
    for (let i = 1; i <= nSeg; i++) {
      lx = Math.max(-0.3, Math.min(0.3, lx + (rnd() - 0.5) * 0.16));
      ly = Math.max(-0.3, Math.min(0.3, ly + (rnd() - 0.5) * 0.16));
      const p = pts[i - 1];
      pts.push([p[0] + lx * step, p[1] + ly * step, p[2] + step]);
    }
    const RAD = 5, ca = [], sa = [];
    for (let a = 0; a <= RAD; a++) {
      ca.push(Math.cos(a / RAD * Math.PI * 2));
      sa.push(Math.sin(a / RAD * Math.PI * 2));
    }
    for (let i = 0; i < nSeg; i++) {
      const p0 = pts[i], p1 = pts[i + 1];
      const r0 = 0.02 + 0.06 * (1 - i / nSeg);
      const r1 = 0.02 + 0.06 * (1 - (i + 1) / nSeg);
      const f0 = i / nSeg, f1 = (i + 1) / nSeg;
      const shade = 0.85 + rnd() * 0.3;
      const cr = Math.min(1, stemCol.r * shade);
      const cg = Math.min(1, stemCol.g * shade);
      const cb = Math.min(1, stemCol.b * shade);
      for (let a = 0; a < RAD; a++) {
        const nx0 = ca[a] + ca[a + 1], ny0 = sa[a] + sa[a + 1];
        const nl = Math.hypot(nx0, ny0) || 1;
        const nx = nx0 / nl, ny = ny0 / nl;
        const emit = (px, py, pz, f) => {
          out.pos.push(px, py, pz);
          out.nrm.push(nx, ny, 0);
          out.col.push(cr, cg, cb);
          out.bend.push(f, phase);
        };
        const x00 = p0[0] + ca[a] * r0, y00 = p0[1] + sa[a] * r0;
        const x01 = p0[0] + ca[a + 1] * r0, y01 = p0[1] + sa[a + 1] * r0;
        const x10 = p1[0] + ca[a] * r1, y10 = p1[1] + sa[a] * r1;
        const x11 = p1[0] + ca[a + 1] * r1, y11 = p1[1] + sa[a + 1] * r1;
        emit(x00, y00, p0[2], f0); emit(x10, y10, p1[2], f1);
        emit(x11, y11, p1[2], f1);
        emit(x00, y00, p0[2], f0); emit(x11, y11, p1[2], f1);
        emit(x01, y01, p0[2], f0);
      }
    }
    // leaves + nodules from ~20 % height up (usually a pair per node, on
    // opposite sides — bare stems read as wires)
    for (let i = Math.max(2, Math.ceil(nSeg * 0.2)); i < nSeg; i++) {
      if (rnd() > 0.8) continue;
      const f = i / nSeg, p = pts[i];
      const az0 = rnd() * Math.PI * 2;
      const nLeaf = rnd() < 0.65 ? 2 : 1;
      for (let l = 0; l < nLeaf; l++) {
        const az = az0 + l * Math.PI + (rnd() - 0.5) * 0.6;
        const ox = Math.cos(az), oy = Math.sin(az);
        dummy.position.set(p[0] + ox * 0.06, p[1] + oy * 0.06, p[2]);
        dummy.rotation.set(0, 0, 0);
        const ns = 0.7 + rnd() * 0.6;
        dummy.scale.set(ns, ns, ns * 1.5);
        dummy.updateMatrix();
        pushGeo(out, prims.nodule, dummy.matrix, nodCol, 0.15, rnd,
                [f, phase]);
        _v.set(ox * 0.5, oy * 0.5, 1).normalize(); // leaf leans up-and-out
        _q.setFromUnitVectors(_Y, _v);
        dummy.rotation.setFromQuaternion(_q);
        dummy.position.set(p[0] + ox * 0.08, p[1] + oy * 0.08, p[2]);
        const ls = 0.9 + rnd() * 0.8;
        dummy.scale.set(ls, ls, ls);
        dummy.updateMatrix();
        pushGeo(out, prims.leaf, dummy.matrix, leafCol, 0.2, rnd,
                [f, phase]);
      }
    }
  }

  // --- chunk builder ---------------------------------------------------------
  const GRASS_GREENS = [0x3d7a45, 0x4d8a3f, 0x35704f, 0x6b8f3a]
    .map((c) => new THREE.Color(c));
  const _c1 = new THREE.Color(), _cA = new THREE.Color(0x28414d),
        _cB = new THREE.Color(0x1e2f37), _cC = new THREE.Color(0x3a463e);
  // regional tones (very low-frequency noise, ~200 m scale): olive-sand
  // flats and deeper teal basins. Kept dark, close to the fog colour —
  // a bright bed under distance fog reads as a camera-following pool.
  const _cD = new THREE.Color(0x40473a), _cE = new THREE.Color(0x1f3c46);

  function buildChunk(ci, cj) {
    const rnd = makeRand(chunkSeed(ci, cj));
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd() + 1e-9))
      * Math.cos(Math.PI * 2 * rnd());
    const ox = ci * CHUNK, oy = cj * CHUNK;
    const group = new THREE.Group();
    group.position.set(ox, oy, 0);
    const geos = [];

    // bed tile (local coords 0..CHUNK; edges shared exactly with neighbours)
    const bedGeo = new THREE.PlaneGeometry(CHUNK, CHUNK, BED_SEG, BED_SEG);
    bedGeo.translate(CHUNK / 2, CHUNK / 2, 0);
    {
      const p = bedGeo.attributes.position;
      const colors = new Float32Array(p.count * 3);
      // palette sits close to the fog color on purpose: a bright bed under
      // distance fog reads as a camera-following "spotlight" pool
      for (let i = 0; i < p.count; i++) {
        const wx = ox + p.getX(i), wy = oy + p.getY(i);
        const z = bedHeight(wx, wy);
        p.setZ(i, z - SEABED);
        const m = Math.min(1, Math.max(0, (z - SEABED + 1.6) / 1.75));
        _c1.copy(_cA).lerp(_cB, 1 - m)
          .lerp(_cC, 0.5 * Math.max(0, noise(wx * 0.02 + 9, wy * 0.02)))
          .lerp(_cD, 0.45 * Math.max(0, noise(wx * 0.006 + 40, wy * 0.006)))
          .lerp(_cE, 0.45 * Math.max(0, noise(wx * 0.005 - 23,
                                              wy * 0.005 + 61)));
        colors.set([_c1.r, _c1.g, _c1.b], i * 3);
      }
      bedGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      bedGeo.computeVertexNormals();
    }
    const bed = new THREE.Mesh(bedGeo, bedMat);
    bed.position.z = SEABED;
    bed.receiveShadow = true;
    group.add(bed);
    geos.push(bedGeo);

    // rocks: lots of pebbles, few boulders (density falls off with size)
    const rockOut = { pos: [], nrm: [], col: [] };
    const rockCol = new THREE.Color(0x46525a);
    const anchors = [];
    const nRocks = 42 + Math.floor(rnd() * 26);
    for (let i = 0; i < nRocks; i++) {
      const s = 0.14 + 2.6 * rnd() ** 4;
      const x = rnd() * CHUNK, y = rnd() * CHUNK;
      dummy.position.set(x, y, bedHeight(ox + x, oy + y) + s * 0.2);
      dummy.rotation.set(rnd() * 3.14, rnd() * 3.14, rnd() * 3.14);
      dummy.scale.set(s * (0.75 + rnd() * 0.5), s * (0.75 + rnd() * 0.5),
                      s * (0.6 + rnd() * 0.55));
      dummy.updateMatrix();
      const pool = s < 0.5 ? rockLo : rockHi;
      pushGeo(rockOut, pool[Math.floor(rnd() * pool.length)], dummy.matrix,
              rockCol, 0.35, rnd);
      if (s > 1.15 && anchors.length < 3) anchors.push([x, y, s]);
    }
    const rocks = bake(rockOut, rockMat, { shadow: true });
    if (rocks) { group.add(rocks); geos.push(rocks.geometry); }

    // coral: mini reefs ringing the big rocks, 2 palette colours per reef
    const coralOut = { pos: [], nrm: [], col: [] };
    for (const [ax, ay, as] of anchors) {
      const reefCols = [
        CORAL_COLORS[Math.floor(rnd() * CORAL_COLORS.length)],
        CORAL_COLORS[Math.floor(rnd() * CORAL_COLORS.length)]];
      const n = 5 + Math.floor(rnd() * 5);
      for (let i = 0; i < n; i++) {
        const az = rnd() * Math.PI * 2;
        const rr = as * 0.9 + 0.4 + rnd() * 2.2;
        const x = ax + Math.cos(az) * rr, y = ay + Math.sin(az) * rr;
        const z = bedHeight(ox + x, oy + y);
        const s = 0.6 + rnd() * 0.9;           // 60 %..150 %
        const color = reefCols[Math.floor(rnd() * reefCols.length)];
        CORAL_KINDS[Math.floor(rnd() * CORAL_KINDS.length)](
          coralOut, rnd, x, y, z, s, color);
      }
    }
    // urchin colonies + abalone hug the big rocks; the odd loose urchin
    // barren sits out on the open bed
    for (const [ax, ay, as] of anchors) {
      if (rnd() < 0.65)
        addUrchins(coralOut, rnd, ax, ay, ox, oy,
                   3 + Math.floor(rnd() * 5), as * 0.8 + 0.2, as * 0.8 + 1.6);
      if (rnd() < 0.5) {
        const n = 1 + Math.floor(rnd() * 2);
        for (let i = 0; i < n; i++) {
          const az = rnd() * Math.PI * 2, rr = as * 0.75 + 0.15 + rnd() * 0.8;
          const x = ax + Math.cos(az) * rr, y = ay + Math.sin(az) * rr;
          const s = 0.14 + rnd() * 0.1;
          dummy.position.set(x, y, bedHeight(ox + x, oy + y) + 0.02);
          dummy.rotation.set((rnd() - 0.5) * 0.3, (rnd() - 0.5) * 0.3,
                             rnd() * 6.28);
          dummy.scale.setScalar(s);
          dummy.updateMatrix();
          pushGeo(coralOut, abaloneGeo, dummy.matrix, ABALONE_COL, 0.25, rnd);
        }
      }
    }
    if (rnd() < 0.22)
      addUrchins(coralOut, rnd, rnd() * CHUNK, rnd() * CHUNK, ox, oy,
                 4 + Math.floor(rnd() * 6), 0, 2.5);
    const coral = bake(coralOut, coralMat, { shadow: true });
    if (coral) { group.add(coral); geos.push(coral.geometry); }

    // kelp: clustered around a point, big height variation, ≥3 m tall,
    // tops capped 1 m below the surface
    if (rnd() < 0.3) {
      const kelpOut = { pos: [], nrm: [], col: [], bend: [] };
      const kx = 8 + rnd() * (CHUNK - 16), ky = 8 + rnd() * (CHUNK - 16);
      const n = 24 + Math.floor(rnd() * 13);
      for (let i = 0; i < n; i++) {
        const px = kx + (rnd() - 0.5) * 16, py = ky + (rnd() - 0.5) * 16;
        const bz = bedHeight(ox + px, oy + py);
        const hMax = Math.max(3.2, -1 - bz);
        const h = 3 + rnd() ** 1.3 * (hMax - 3);
        addKelp(kelpOut, rnd, px, py, bz - 0.15, h);
      }
      const kelp = bake(kelpOut, kelpMat, { sphere: wv.amplitude + 0.8 });
      if (kelp) { group.add(kelp); geos.push(kelp.geometry); }
    }

    // sea grass: big blob-shaped meadows; blades share the patch direction
    // (small jitter), heights ~N(0.5, 0.2²) clamped to [0.15, 1] m
    let grassInst = null;
    const nPatch = (rnd() < 0.75 ? 1 : 0) + (rnd() < 0.3 ? 1 : 0);
    if (nPatch) {
      const items = [];
      for (let pI = 0; pI < nPatch; pI++) {
        const pcx = rnd() * CHUNK, pcy = rnd() * CHUNK;
        const dir0 = rnd() * Math.PI * 2;      // shared blade yaw
        const leanAz = rnd() * Math.PI * 2;    // shared lean direction
        const blobPh = rnd() * Math.PI * 2;
        const nB = 380 + Math.floor(rnd() * 220);
        for (let i = 0; i < nB; i++) {
          const th = rnd() * Math.PI * 2;
          const rMax = 9 * (0.72 + 0.28 * Math.sin(3 * th + blobPh));
          const rr = Math.sqrt(rnd()) * rMax;
          const h = Math.min(1, Math.max(0.15, 0.5 + gauss() * 0.2));
          items.push({
            x: pcx + Math.cos(th) * rr, y: pcy + Math.sin(th) * rr, h,
            yaw: dir0 + (rnd() - 0.5) * 0.5,
            lean: 0.1 + rnd() * 0.18, leanAz,
            color: GRASS_GREENS[Math.floor(rnd() * GRASS_GREENS.length)]
              .clone().multiplyScalar(0.85 + rnd() * 0.3),
          });
        }
      }
      grassInst = new THREE.InstancedMesh(bladeGeo, grassMat, items.length);
      for (const [i, it] of items.entries()) {
        dummy.position.set(it.x, it.y, bedHeight(ox + it.x, oy + it.y));
        dummy.rotation.set(it.lean * Math.cos(it.leanAz),
                           it.lean * Math.sin(it.leanAz), it.yaw);
        dummy.scale.set(0.8 + 0.5 * it.h, 1, it.h);
        dummy.updateMatrix();
        grassInst.setMatrixAt(i, dummy.matrix);
        grassInst.setColorAt(i, it.color);
      }
      grassInst.computeBoundingSphere();
      group.add(grassInst);
    }

    scene.add(group);
    return { group, geos, inst: grassInst, ij: [ci, cj] };
  }

  // --- chunk manager ---------------------------------------------------------
  const chunks = new Map();
  let queue = [];                              // [dist, i, j], nearest first
  const key = (i, j) => i + "," + j;
  function disposeChunk(rec) {
    scene.remove(rec.group);
    for (const g of rec.geos) g.dispose();
    rec.inst?.dispose();
  }
  function ensureChunks(cx, cy) {
    const ci0 = Math.floor(cx / CHUNK), cj0 = Math.floor(cy / CHUNK);
    const RC = Math.ceil(R_LOAD / CHUNK) + 1;
    queue = [];
    for (let i = ci0 - RC; i <= ci0 + RC; i++)
      for (let j = cj0 - RC; j <= cj0 + RC; j++) {
        const d = Math.hypot((i + 0.5) * CHUNK - cx, (j + 0.5) * CHUNK - cy);
        if (d < R_LOAD && !chunks.has(key(i, j))) queue.push([d, i, j]);
      }
    queue.sort((a, b) => a[0] - b[0]);
    for (const [k, rec] of chunks) {
      const d = Math.hypot((rec.ij[0] + 0.5) * CHUNK - cx,
                           (rec.ij[1] + 0.5) * CHUNK - cy);
      if (d > R_UNLOAD) { disposeChunk(rec); chunks.delete(k); }
    }
  }
  function processQueue(budgetMs) {
    const t0 = performance.now();
    while (queue.length && performance.now() - t0 < budgetMs) {
      const [, i, j] = queue.shift();
      if (!chunks.has(key(i, j))) chunks.set(key(i, j), buildChunk(i, j));
    }
  }
  let chunkClock = 1e9, booted = false;        // force ensure on first update

  // --- marine life (web/mako/fauna.js) -----------------------------------------
  // ~30 species in schools/pods, depth-stratified, recycled through the fog
  // ring — see fauna.js for the species table and the swim-bend shader.
  const fauna = makeFauna(THREE, scene, { seabed: SEABED, bedHeight });

  // --- sky dome (for above-surface camera angles) -----------------------------
  // An equirect-gradient sphere centred on the camera (fog-exempt so the
  // underwater FogExp2 never eats it), with a sun glow and a few far
  // clouds riding the shell. Hidden underwater — there the fog + clear
  // color ARE the backdrop.
  const sky = new THREE.Group();
  {
    const W = 16, H = 512;
    const cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    const g = cv.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0.0, "#2f6ea8");         // zenith
    grad.addColorStop(0.42, "#9cc6e2");
    grad.addColorStop(0.495, "#e8f4fb");       // horizon band
    grad.addColorStop(0.505, "#3d77a6");
    // sea shades tuned to blend with the wave plane's fogged far edge
    // (the plane only spans 320 m — the dome takes over beyond it, and
    // above water the fog fades the plane toward AIR_FOG_COLOR)
    grad.addColorStop(0.53, "#2c608c");
    grad.addColorStop(0.65, "#1b4463");
    grad.addColorStop(1.0, "#11334c");
    g.fillStyle = grad; g.fillRect(0, 0, W, H);
    const tex = new THREE.CanvasTexture(cv);
    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(1500, 32, 24),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide,
                                    fog: false, depthWrite: false }));
    dome.rotation.x = Math.PI / 2;             // poles → z (z up)
    dome.renderOrder = -100;
    sky.add(dome);

    // sun: core + halo, in the same direction as the scene's key light
    const sunDir = new THREE.Vector3(30, 50, 90).normalize();
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(42, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xfff8dc, fog: false }));
    core.position.copy(sunDir).multiplyScalar(1420);
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(110, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xfff3c0, transparent: true,
                                    opacity: 0.22, fog: false,
                                    depthWrite: false }));
    halo.position.copy(core.position);
    sky.add(core, halo);

    // clouds: flattened white spheres pinned to the shell (they ride the
    // camera like the dome, i.e. they read as "at infinity")
    const cloudMat = new THREE.MeshBasicMaterial({ color: 0xf4f8fb,
      transparent: true, opacity: 0.88, fog: false, depthWrite: false });
    const cloudGeo = new THREE.SphereGeometry(1, 10, 7);
    for (let i = 0; i < 9; i++) {
      const az = rand() * Math.PI * 2, el = 0.12 + rand() * 0.4;
      const d = 1250;
      const base = new THREE.Vector3(
        Math.cos(az) * Math.cos(el) * d, Math.sin(az) * Math.cos(el) * d,
        Math.sin(el) * d);
      for (let p = 0; p < 2 + Math.floor(rand() * 3); p++) {
        const puff = new THREE.Mesh(cloudGeo, cloudMat);
        puff.position.copy(base);
        puff.position.x += (rand() - 0.5) * 160;
        puff.position.y += (rand() - 0.5) * 160;
        puff.position.z += (rand() - 0.5) * 24;
        const s = 55 + rand() * 90;
        puff.scale.set(s, s * (0.75 + rand() * 0.4), s * 0.32);
        sky.add(puff);
      }
    }
    sky.visible = false;
    scene.add(sky);
  }

  // --- visibility (murk) -------------------------------------------------------
  // v ∈ [0,1]: 0 = pea soup, 1 = tropical clear. The clear end is capped so
  // sight range stays inside the chunk load radius — chunks stream in and
  // out entirely hidden by the fog.
  let visibility = 0.55;
  const fog = new THREE.FogExp2(0x0d3049, 0.024);
  scene.fog = fog;
  let densityUnder = fog.density;
  const murkColor = new THREE.Color(0x0d3049);
  // Above water the fog plays "deep water": it fades distant SUBMERGED
  // geometry (bed, kelp beyond the wave plane) toward a sea blue, so the
  // world doesn't show crisply through the surface at grazing angles. The
  // sky dome, sun and clouds are fog-exempt and stay crisp.
  const AIR_DENSITY = 0.0105;
  const AIR_FOG_COLOR = new THREE.Color(0x2b6390);
  function applyVisibility() {
    densityUnder = 0.055 - visibility * 0.044;
    murkColor.copy(new THREE.Color(0x081f30)
      .lerp(new THREE.Color(0x11486e), visibility));
    if (!sky.visible) {
      fog.density = densityUnder;
      fog.color.copy(murkColor);
    }
    return murkColor;
  }
  let clearColor = applyVisibility();

  // --- light payload (frontend-only) ------------------------------------------
  // A lamp on the `ui` module's payload port: a spotlight + a soft additive
  // beam cone. Returns null if the design has no ui module.
  function makeLight(THREE2, craftGroup, xLocal) {
    const g = new THREE.Group();
    const housing = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.045, 0.07, 12),
      new THREE.MeshStandardMaterial({ color: 0x20262c, roughness: 0.6 }));
    housing.rotation.z = Math.PI / 2;
    const lens = new THREE.Mesh(
      new THREE.SphereGeometry(0.028, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0xfff2c9,
        emissive: 0x000000, roughness: 0.3 }));
    lens.position.x = 0.04;
    const spot = new THREE.SpotLight(0xffeeb8, 0, 40, 0.5, 0.55, 1.2);
    spot.position.set(0.05, 0, 0);
    spot.target.position.set(8, 0, -1.5);
    const beam = new THREE.Mesh(
      new THREE.ConeGeometry(3.4, 12, 16, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xffedb0, transparent: true,
        opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false,
        side: THREE.DoubleSide }));
    beam.rotation.z = Math.PI / 2;             // cone axis → +x, apex at lamp
    beam.position.set(6.05, 0, -1.0);
    beam.rotation.y = 0.12;
    g.add(housing, lens, spot, spot.target, beam);
    g.position.set(xLocal, 0, -0.13);          // under the payload port
    craftGroup.add(g);
    let on = false;
    return {
      get on() { return on; },
      toggle() {
        on = !on;
        spot.intensity = on ? 60 : 0;
        beam.material.opacity = on ? 0.10 : 0;
        lens.material.emissive.setHex(on ? 0xffe9a8 : 0x000000);
        return on;
      },
    };
  }

  return {
    surfaceHeight,
    bedHeight,
    makeLight,
    get clearColor() { return clearColor; },
    setVisibility(v) {
      visibility = Math.min(1, Math.max(0, v));
      clearColor = applyVisibility();
      return clearColor;
    },
    /** `cam` = the camera's world position (THREE.Vector3) — decides the
     *  underwater-vs-sky presentation on surface crossings. */
    update(dt, t, cx, cy, cam) {
      uTime.value = t;
      updateSurface(t, cx, cy);
      fauna.update(dt, t, cx, cy);

      // stream chunks: first update builds the near field synchronously,
      // then the manager re-scans every 0.5 s and builds on a frame budget
      chunkClock += dt;
      if (chunkClock > 0.5) {
        chunkClock = 0;
        ensureChunks(cx, cy);
      }
      if (!booted) {
        booted = true;
        while (queue.length && queue[0][0] < 150) {
          const [, i, j] = queue.shift();
          chunks.set(key(i, j), buildChunk(i, j));
        }
      }
      processQueue(6);

      if (cam) {
        // hysteresis band wider than the ambient ripples, so a camera
        // skimming the surface doesn't strobe between presentations
        const h = surfaceHeight(cam.x, cam.y, t);
        const above = cam.z > h + (sky.visible ? -0.08 : 0.14);
        if (above !== sky.visible) {
          sky.visible = above;
          fog.density = above ? AIR_DENSITY : densityUnder;
          fog.color.copy(above ? AIR_FOG_COLOR : murkColor);
          // from above, the sea should read mostly solid (the fog is in
          // "air" mode, so transparency would show a crisp seabed)
          surfMat.opacity = above ? 0.92 : 0.5;
          sparkles.visible = above;
        }
        if (above) updateSparkles(t, cam);
        snow.visible = !above;
        if (!above) updateSnow(dt, cam);
        // dome rides the camera in x/y but stays sea-anchored in z, so
        // its equator (the painted horizon) sits at sea level
        sky.position.set(cam.x, cam.y, 0);
      }
    },
    dispose() {
      for (const rec of chunks.values()) disposeChunk(rec);
      chunks.clear();
      queue = [];
      fauna.dispose();
      for (const g of [...rockLo, ...rockHi, ...Object.values(prims),
                       urchinGeo, abaloneGeo, bladeGeo, surfGeo, spkGeo,
                       snowGeo])
        g.dispose();
      for (const m of [bedMat, rockMat, coralMat, kelpMat, grassMat,
                       surfMat, sparkles.material, snow.material])
        m.dispose();
    },
  };
}
