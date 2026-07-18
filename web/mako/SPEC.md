# Build a Mako — client/server contract

The "Build a Mako" page (`/mako/`) is a drag-and-drop submarine builder. The
client assembles a **design** (modules + options + spawn pose) and POSTs it to
the compile server, which builds the manta `World` and lowers ONE transform —
the `Sim` — to WebAssembly with `TargetWasm`, returning a cached bundle the
browser drives at 60 Hz. All control (manual mixing, orientation/position
PIDs) runs client-side on the baked `meta.mixer` tables. This file is the
contract between `web/mako/*.js` (client) and `server/mako/*.py` (server).

## Coordinate/world conventions

A **real spinning Earth** (`Earth` + `SeaWaves`, per
`examples/vehicles/submarine.py`) with the dive site anchored at the north
pole via a `Scene`:

* WorldFrame is INERTIAL with the planet at the true origin; Earth spins at
  the sidereal rate about +z. The client re-expresses all state in the
  scene's local ENU coordinates using `meta.scene` (the JS port of
  `Scene.relative`): `p_scene = R_ws(t)ᵀ·(p_world − o(t))`, `R_ws(t) =
  R_axis(ωt)·R_ps`, `o(t) = R_axis(ωt)·anchor + center`. At the pole, local
  up is the spin axis, so gravity stays along scene −z at all t.
* Scene frame: x/y horizontal, z up. Sea surface at scene **z ≈ 0** (waving:
  η = A·cos(k·ξ − ω_w·t), deep-water dispersion — parameters in
  `meta.waves`), seabed at scene **z = −30**.
* Fluid: Earth's ISA atmosphere over its `Ocean` (hydrostatic pressure for
  the Barometer, wave orbital velocities, seawater viscosity), blended over
  `surface_smoothing=0.15` m. Earth's sea-level-sphere obstacle is OFF.
* Contact: one point `Collider` per spine module on the hull axis
  (stiffness 2e4, damping 800, friction 250) against a world half-space at
  scene z = −30 + `HULL_R` — a point on the axis against the raised plane
  is exactly a hull-radius sphere touching the true seabed.
* Spawn: `scene.at_rest(...)` — the craft co-rotates with the planet (its
  IMU gyro reads Ω on local up).
* Craft name: **`mako`**. Craft origin: centre of the spine, z = 0 on the
  hull axis. Spine modules chain along x, **nose at +x**.

## design.json (client → server)

```json
{
  "version": 2,
  "name": "my mako",
  "spine": [
    {"type": "nose_dvl", "options": {"dvl_noise": 0.02}},
    {"type": "agility_front", "options": {}},
    {"type": "brain", "options": {}},
    {"type": "ins", "options": {}},
    {"type": "battery", "options": {}},
    {"type": "agility_rear", "options": {}},
    {"type": "fin_control", "options": {"max_thrust": 250}}
  ],
  "spawn": {"x": 0, "y": 0, "depth": 2.0, "heading_deg": 0,
            "wave_amp": 0.25, "wave_len": 18, "visibility": 55}
}
```

* `spine` is ordered **front (+x) to back (−x)**. The first element must be
  a `nose`-kind module, the last a `rear`-kind, everything between
  `spine`-kind. There are NO mount points — the craft is one stack.
* Unknown option keys are ignored; values are clamped server-side to the
  catalog ranges; missing options take catalog defaults.
* `spawn.depth` ≥ 0 (metres below surface; position z = −depth),
  clamped to [0.5, 25]. x/y clamped to ±200. `heading_deg` → yaw about z
  (0 = +x).
* `spawn.wave_amp` (m, [0, 1.5], default 0.25) and `spawn.wave_len` (m,
  [6, 80], default 18) set the sea state — they bake into the compiled
  world (SeaWaves surface + orbital velocities), so they are part of the
  canonical design and the cache hash. `wave_amp: 0` = flat sea (the
  wave field is skipped entirely). `spawn.visibility` (%, [0, 100]) is
  frontend-only murk: it seeds the drive-view slider and is STRIPPED
  from the canonical design, so changing it never recompiles.

## Module catalog (from `MakoModules.pdf`, 2026-07-07)

Every module is a ~20 cm-diameter spine segment. Part names are
`"<type><spineIndex>"`, sub-parts suffixed (`_prop`, `_gps`, `_dvl`, `_imu`,
`_depth`, `_fint/b/l/r`, `_vl/vr/fl/fr`), so readings are e.g.
`mako.ins3_imu.gyro`, inputs e.g. `mako.fin_control6_prop.throttle`.

Shared physics (server `catalog.py:_body`):

* **Near-neutral buoyancy** per module: mass is DERIVED — fairing volume
  × flooded fraction `fill` × ρ — and the buoy volume is mass/ρ ×
  `BUOY_RESERVE` (1.012): every module carries the same ~1.2 % reserve, so
  an unpowered mako creeps to the surface at ~0.08 m/s (dead-ship
  failsafe). The trim-residual warning tolerates exactly this designed
  float-up. Mass is NOT a user option. CoM 1 cm below the buoy line
  (slight passive
  roll/pitch stability). Roll MoI 0.8·m·r² (shell, not solid).
* **Wetted-surface drag only**: axial = cylinder-skin friction (Cf 0.008;
  + a 0.25-cd face on nose/rear caps); crossflow = D·L at cd 1.1, split
  4-way around the rim (y,z = ±0.09) so the V_LIN linear floor damps roll.
* **Thrusters**: reversible, unit force axis (throttle in N, clamp
  ±max_thrust), prop counter-torque K_CT = 0.015 N·m/N signed by spin
  (agility pairs counter-rotate; a lone stern prop heels a little).

| type (kind) | L (m) | fill | parts / options |
|---|---|---|---|
| `nose_dvl` (nose) | 0.30 | 0.85 | down-looking `VelocitySensor` (`dvl_noise` [0.005,0.2] 0.02) |
| `nose_camera` (nose) | 0.20 | 0.85 | structure only (stereo cameras cosmetic until manta cameras have targets) |
| `nose_sonar` (nose) | 0.50 | 0.80 | structure only (sonar not in manta yet); extra transducer drag |
| `brain` (spine) | 0.20 | 1.0 | water-gated `SurfaceGPS` on top mast (`gps_noise` [0.02,2] 0.1) + `Barometer` (`depth_noise` [100,5000] 500) |
| `ins` (spine) | 0.30 | 1.0 | `IMU` (`gyro_noise` [1e-4,0.01] 5e-4, `gyro_bias` [0,1e-3] 2e-5, `accel_noise` [0.005,0.5] 0.02) |
| `ui` (spine) | 0.15 | 1.0 | structure only; green payload port (payloads later) |
| `battery` (spine) | 0.35 | 1.0 | structure only (battery model later) |
| `side_scan` (spine) | 0.55 | 1.0 | structure only (sonar not in manta yet) |
| `thrust_vert` (spine) | 0.30 | 0.70 | tunnel `Thruster` +z (`max_thrust` [50,500] 250, `noise` [0,10] 1) |
| `thrust_horiz` (spine) | 0.30 | 0.70 | tunnel `Thruster` +y (same options) |
| `agility_front` (spine) | 0.25 | 0.80 | 4 pods: 2 vertical (±y 0.15, counter-rotating) + 2 canted 25° INWARD (±y 0.17); shared thrust options |
| `agility_rear` (spine) | 0.25 | 0.80 | as front, canted 25° OUTWARD |
| `rear_thrust` (rear) | 0.40 | 0.70 | stern prop (x −0.20 from centre) + `SurfaceGPS` (thrust + gps options) |
| `fin_control` (rear) | 0.40 | 0.65 | stern prop + `SurfaceGPS` + 4 identical `ControlSurface` fins in a **+ tail** (top/bottom normal y = yaw, left/right normal z = pitch; at x −0.12, arms 0.16; `fin_area` [0.005,0.03] 0.011, `max_deflection` [10,40] 25°; parasitic fin drag recorded for damping/feedforward) |

Module centres: accumulate lengths front→back, shift so the spine is centred
on x = 0. GPS antenna geometry is shared (mast top at z = +0.16).

## Transform + mixer baked per design

* **Sim** (the only wasm bundle): `TargetWasm(Sim(world))` →
  `sim.js/.mjs/.wasm`. The JS `Sim` view is a noiseless oracle; the client
  injects **pre-scaled** σ·N(0,1) draws into the `step` entry's `noise`
  slot via `Runtime.call` (kernels add the noise verbatim — verified
  against `NoiseDriver` in the smoke test; each descriptor noise field
  carries its σ).
* **Mixer** (`meta.mixer`, plain JSON — no second bundle): the baked
  body-frame wrench→actuator map every client control law runs on. Demand
  `[F; τ]`, get thruster newtons with one n×6 matrix-vector product
  (`alloc` — Tikhonov-damped pseudo-inverse of the wrench matrix, λ=0.01
  so token counter-torque columns are ignored). `fins[]` carries each
  control surface's speed-scheduled torque coefficient `k` (τ ≈
  k·u²·deflection about its `axis`, zero at rest — the client blends fins
  in as u² grows) plus a SIGNED `roll_k` (lift at the fin's radial arm:
  opposing signs across a pair, so collective deflection steers pitch/yaw
  and differential deflection rolls — sign convention verified open-loop
  against the compiled sim). `drag` polynomials feed the client
  feedforward; `drag.cross_lin`/`cross_quad` (3×3) are the
  translation→torque weathervane coupling about the CoM (τ to command =
  cross_lin·v + cross_quad·v∘|v|) — only the yaw←surge/sway and
  pitch←heave entries are emitted (the roll row and pitch←surge ride on
  small vertical drag offsets the same order as the unmodeled metacentric
  righting, and measured worse than feedback alone); the client fades the
  feedforward out by fin share, where the same moment is the tail's
  stabilizing stiffness. `axes` says which of
  surge/sway/heave/roll/pitch/yaw the design can drive (thrusters at
  rest; roll/pitch/yaw credit fins-at-speed) — inert keys warn.
  `trim` (top-level) is the least-squares static wrench balance.

## API

Dev server: `python -m server.mako` (repo root) → serves `web/` statically on
`:8077` **and**:

`POST /api/mako/spawn` with design.json body →

* `200 {"ok": true, "base": "/mako/builds/<hash12>/", "meta": {...}}`
* `200 {"ok": false, "errors": ["human-readable design errors"]}` (validation)
* `500 {"ok": false, "errors": ["compile stage failed: ..."]}`

`<hash12>` = first 12 hex of sha256 of the canonicalised design (sorted keys,
clamped options — so equivalent designs share a cache entry). Artifacts land
in `web/mako/builds/<hash12>/` (gitignored) and are served statically; if the
dir already has `meta.json`, the server returns it without recompiling.
`emcc` comes from
`~/emsdk` (`source ~/emsdk/emsdk_env.sh`); if emcc is missing the endpoint
returns `ok:false` with a message saying so.

## meta.json (in the bundle dir, also inlined in the response)

```json
{
  "version": 2,
  "name": "my mako",
  "craft": "mako",
  "files": {"sim": "sim.js"},
  "inputs": [{"name": "mako.fin_control6_prop.throttle", "min": -250, "max": 250}],
  "scene": {"omega": 7.2921159e-5, "axis": [0,0,1], "center": [0,0,0],
            "anchor_planet": [0,0,6378137.0],
            "R_planet_from_scene": [[...3x3...]], "g0": 9.798285},
  "waves": {"amplitude": 0.25, "wavelength": 18.0, "k": 0.349,
            "omega": 1.849, "dir_planet": [1,0,0]},
  "seabed_z": -30.0, "hull_r": 0.1,
  "spawn": {"scene_position": [0, 0, -2.0], "heading_deg": 0},
  "mass": 62.0, "buoyancy_n": 6.7,
  "trim": {"agility_front1_vl.throttle": 0.02},
  "mixer": {"inputs": ["mako.…throttle"], "alloc": [[...6...]],
            "wrench_cap": [694.1, 128.9, 205.6, 55.2, 121.9, 88.0],
            "fins": [{"input": "mako.fin_control6_fint.deflection_cmd",
                      "axis": "yaw", "k": 2.84, "roll_k": -0.833,
                      "max_defl": 0.4363}],
            "axes": {"surge": true, "sway": true, "heave": true,
                     "roll": true, "pitch": true, "yaw": true},
            "drag": {"force_lin": [3], "force_quad": [3],
                     "torque_lin": [3], "torque_quad": [3],
                     "cross_lin": [[3, 3]], "cross_quad": [[3, 3]]}},
  "rate_scale": {"surge": 6.9, "sway": 0.81, "heave": 1.34, "yaw": 1.93},
  "warnings": []
}
```

`inputs` order matches the descriptor's input order (names carry the craft
prefix). `warnings` is free-form (trim clipping, heavy/negative buoyancy,
no-lateral-thrust hints, etc.). Two more fields beyond the example: `design`
(the canonicalised design, so a cached bundle is self-describing — the
client's `/mako/?load=<hash>` boots the drive view from it) and `timings`
(per-stage emit/emcc/wasm-opt seconds).

Implementation notes (measured): kernels over 300 KB compile at `-O0` plus
a Binaryen `wasm-opt -O2` post-pass — the Earth-world sim kernels run
0.5–1.5 MB (ISA atmosphere + waving ocean evaluated at every drag
surface/buoy/fin) and LLVM is superlinear on one enormous straight-line
CasADi function: 0.87 MB measured 184 s at `-O3` vs ~24 s down this path,
still 9.8× real time in the browser. All bundles build with
`-s STACK_SIZE=4194304` — the generated ABI shims stack-allocate CasADi
work arrays past Emscripten's 64 KB default. Starter-design spawn ≈ 27 s
cold, instant on cache hit.

## Client control (viewer)

All control runs client-side at FRAME = 0.02 s on `meta.mixer`; actuator
commands pass through a first-order low-pass (τ ≈ 60 ms — electric
thrusters are quick but not instantaneous) before entering the sim. Three
modes, toggled by a button:

* **manual** — keys map straight to wrench demand → mixer → actuators. No
  feedback anywhere.
* **ori hold** — attitude keys SLEW an orientation setpoint (shown on the
  navball); a quaternion-error PID (+ rate feedforward from the slew) turns
  the error into a body torque demand for the mixer. Translation keys stay
  manual (body-frame force demand). Roll auto-levels when its keys are
  idle.
* **auto** — keeps the ori-hold attitude loop; translation keys now move a
  POSITION setpoint (rendered as a translucent axis triplet in the world);
  a position PID (scene frame, rotated to body) produces the force demand.

Keys: W/S pitch down/up · A/D yaw left/right · Q/E roll ·
Up/Down arrows surge · Right/Left arrows sway · space/shift heave.
A key on an axis with `meta.mixer.axes[axis] == false` does nothing but
flash a small fading warning. Fins are blended into the torque mix as
dynamic pressure grows (`k·u²`), so fins-only boats steer at speed while
pod boats work from rest. Pitch/yaw deflect a pair collectively; roll
deflects the set differentially on the signed `roll_k` (least-squares),
and the two superpose under a ±0.9·max_defl clamp. The fin share of the
PID torque is boosted ~8× (blended by share): fin-steered axes are
dominated by the hull's own weathervane damping (measured c/I ≈ 22 s⁻¹ on
the torpedo vs the controller's K_D = 3.6), so a τ = I·ω̇-sized deflection
would sustain a rate ~8× below intent. Manual torque keys and the
integral clamp scale with the speed-dependent fin authority
(`caps + Σk·u²·0.9·max_defl`).

Sim loop: FRAME=0.02 s, DT=0.004 s, 5 substeps, pose interpolation — exactly
`web/quad/demo.js`'s accumulator pattern, with all rendering and control in
scene ENU coordinates via the `meta.scene` transform.

## Drive-view environment (frontend-only, 2026-07-17)

The world render (`web/mako/environment.js`) is an ENDLESS chunked
procedural environment — no square borders:

* **Chunks**: 64 m tiles keyed by integer chunk coords; content is seeded
  per chunk (`hash(i, j)` with the indices wrapped at the planet's
  circumference, so circumnavigating is theoretically consistent — the flat
  scene frame is the approximation, the physics runs on the round planet).
  Chunks stream in within 230 m and unload past 268 m, always outside fog
  sight range (the clear end of the vis slider is capped so ~9 % contrast
  is the worst-case pop). First frame builds the near field synchronously;
  after that the manager re-scans every 0.5 s and builds on a 6 ms/frame
  budget. Chunk geometry is CHUNK-LOCAL (float32-safe far from origin);
  the group carries the world offset.
* **Per chunk**: bed tile (16×16 grid of the global relief noise, vertex
  colours), one merged rock mesh, one merged coral mesh, an optional merged
  kelp mesh, an optional instanced sea-grass mesh — ≤5 draw calls/chunk,
  shared materials.
* **Rocks**: 42–68 per chunk, size `0.14 + 2.6·r⁴` (many pebbles, few
  boulders). Convex irregular polyhedra: icosa/dodecahedra with a
  consistent radial jitter per shared corner (soup duplicates get the same
  factor — no tears), flat-shaded. No boxes.
* **Coral**: rocks with s > 1.15 anchor mini reefs (≤3/chunk): 5–9 corals
  ringing the rock, four deliberate types (fan, brain dome, columnar,
  staghorn), per-coral scale 60–150 %, colours from a fixed 6-colour coral
  palette (2 picked per reef), per-face tint jitter.
* **Kelp**: 30 % of chunks get one cluster (24–36 plants within ~16 m).
  Heights 3 m … (surface − 1 m), biased short (`3 + r^1.3·(hMax−3)`). Each
  plant: tapered 5-sided stem tube on a mostly-upward random walk (bounded
  lean ⇒ gentle spiral), leaves usually in opposite pairs + octahedron
  nodules. Sway is GPU-side (`aBend = (height fraction, plant phase)`
  attribute + `onBeforeCompile`): ambient current + the wave's own orbital
  displacement `A·e^{k·z}` along the wave direction — physics-consistent
  depth decay for free.
* **Sea grass**: ~75 % of chunks get a ~18 m blob patch (+30 % chance of a second) (380–600 blades,
  instanced): shared patch direction with small per-blade jitter, shared
  lean, heights N(0.5, 0.2²) clamped [0.15, 1] m, per-instance colour from
  4 greens. Gentle GPU sway bends by (height fraction)² so roots stay put.
* **Marine life** (`web/mako/fauna.js`): ~30 species — 26 fish/sharks/
  rays/eels + spiny lobster + rock crab + harbor seal, sea lion, common
  dolphin — in a data-driven table (length, colours/pattern painter, depth
  band, abundance weight, cruise speed, group size). ~14 groups live at
  once: spawned by abundance weight in the 55–110 m fog ring, placed
  vertically by each species' depth band (surface fish up top, bed fish
  down deep), recycled past 160 m. Models are merged vertex-coloured
  primitives (body of revolution + fin quads); swimming bends each model
  along ONE axis in the vertex shader — laterally for fish/sharks/eels,
  vertically for mammals, wing-flap for rays — with per-group accumulated
  phase so tail-beat rate tracks swim speed. Fish school, mammals pod
  (≥3) and periodically run for the surface to breathe; lobsters and
  crabs walk ON the bed (crabs sideways); morays/pricklebacks/treefish
  hold station and sway. Sessile life (purple/red urchins with spikes,
  abalone shells) bakes into the chunk coral mesh around big rocks.
* **Marine snow**: 800 pixel-scale motes (no size attenuation; size 4
  DEVICE pixels ≈ a 2-px CSS speck, on a mostly-solid dot sprite — a soft
  gradient sampled that small disappears) in a 42 m sphere around the
  camera, slow drift, fade-in on respawn — parallax "starfield" that
  sells motion. Underwater only.
* **Bed colour regions**: two extra very-low-frequency noise lerps
  (~200 m scale) toward olive-sand and deep-teal tones, kept dark near
  the fog colour.
* **Surface**: exact physics η + small visual-only ripples and a
  brightness shimmer (vertex colours), so a flat sea doesn't read as
  glass. From above: opacity 0.92, additive sun-glint sparkle points, and
  the fog switches to a "deep water" mode (density 0.0105, sea-blue
  colour) so submerged terrain beyond the wave plane fades out instead of
  showing crisply; the sky dome/sun/clouds are fog-exempt. Dome sea
  gradient is tuned to that fogged far edge.
* **Thrust gating** (`viewer.js`): thrusters only produce force in the
  water — each prop's commanded throttle fades to zero over its last
  ~10 cm of submergence against the live wave η (smoothstep + a τ = 0.25 s
  low-pass on the gate, so a hard edge toggling in waves doesn't pump
  surface oscillations) before the command reaches the sim (client-side
  gate; the compiled physics is unchanged).
  Bubbles, prop-spin visuals and dust follow automatically since they
  read the gated command.
* **Dust** (`viewer.js`): pooled brown translucent points. Spawned when a
  thruster is within 1.6 m of the visual bed (rate ∝ throttle × proximity,
  initial shove away from the wash) or when the hull scrapes the physics
  floor at speed. They don't rise: heavy drag, slow settle, rest on the
  visual bed.
* **Camera** (`viewer.js`): body-frame orbit offset applied to a
  LOW-PASSED craft pose (world frame; τ = 0.1 s position, 0.3 s
  orientation) — wave rocking and control jitter don't shake the view,
  drags stay instant.
* **HUD**: `pos` line is true planet lat/long from the exact scene→planet
  rigid transform (anchor is the north pole, so lat starts at 90°).
