"""Design → World → WASM bundles, per `web/mako/SPEC.md`.

The pipeline, in the order `build_bundle` runs it:

  1. `validate_design`  — structural checks + option clamping. Returns the
     CANONICAL design (sorted keys, clamped options, defaults filled, nulls
     dropped) plus a list of human-readable errors; the canonical form is
     what `design_hash` runs on, so equivalent designs share a cache entry.
  2. `build_world`      — a real spinning `Earth` (sidereal rate, ISA air
     over a `SeaWaves` ocean, hydrostatic pressure) with the dive site
     anchored at the north pole via a `Scene`, per-module point `Collider`s
     against a seabed half-space at scene z = −30, and the `mako` craft
     assembled from the catalog, spawned at the design pose (co-rotating).
  3. analytics          — total mass, displaced volume, net buoyancy, a
     least-squares trim over the thruster wrench matrix, and the MIXER:
     the baked wrench→actuator allocation (thrusters + speed-scheduled
     fins) the client's manual/PID control laws run on. No runtime solves —
     everything is matrix-vector.
  4. transforms         — `TargetWasm(Sim)`; the browser drives it raw (the
     old LQR bundle is gone — control is client-side mixing + PIDs).
  5. `meta.json`        — the SPEC contract blob, written LAST (its presence
     is the server's cache marker, so it must only exist for a complete
     bundle).

Everything here is synchronous and stateless; the HTTP layer (`__main__`)
owns caching and locking.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from manta import Sim, TargetWasm, World
from manta.craft import Craft
from manta.fields.collision import CollisionField
from manta.parts import Collider
from manta.planets import Earth, SeaWaves

from .catalog import (
    BUOY_RESERVE, CATALOG, CRAFT, Contrib, GRAVITY, HULL_R, RHO_WATER,
    V_LIN,
    clamp_options, spine_length,
)

# --- world constants (SPEC "Coordinate/world conventions") -------------------
# A real spinning Earth with the dive site anchored at the NORTH POLE: local
# up there IS the spin axis, so the scene's ENU frame only yaws with the
# planet, gravity stays along scene −z, the co-rotating ocean is locally
# still, and a world-frame half-space seabed (normal = spin axis) stays
# exact at every t. The client re-expresses world state in scene ENU with
# the `meta.scene` transform (~15 lines of quaternion math).
SEABED_DEPTH = 30.0               # seabed at scene z = −SEABED_DEPTH
SURFACE_BLEND = 0.15              # air/water blend half-band (m)
# Waves are a per-design spawn setting (they bake into the compiled
# world): amplitude/wavelength clamps + defaults. amplitude 0 → flat sea
# (the SeaWaves field is skipped entirely).
WAVE_AMP = (0.0, 1.5, 0.25)       # (min, max, default) m
WAVE_LEN = (6.0, 80.0, 18.0)      # (min, max, default) m
ANCHOR = (0.0, 0.0, Earth.R_EQ)   # north-pole surface point (planet frame)
G0 = Earth.MU / Earth.R_EQ ** 2   # surface gravity implied by μ

# Contact: one point Collider per spine module, on the hull axis. Against a
# seabed half-space RAISED by HULL_R a point on the axis is exactly a
# HULL_R-sphere touching the true seabed — the "spherical collider per
# module" model without needing sphere-shape support. Constants sized like
# the rocket example for a ~60–100 kg craft: ~1 cm static penetration
# spread over the spine, well-damped, with tangential grip so a resting
# hull doesn't skate.
COL_STIFFNESS = 2.0e4             # N/m per module
COL_DAMPING = 800.0               # N·s/m along the contact normal
COL_FRICTION = 250.0              # N·s/m tangential (isotropic)

# Salted into design_hash: bump on any builder change that alters emitted
# bundles for the SAME design (world constants, meta layout…), so stale
# cache entries miss instead of being served.
BUILDER_REV = 20

# spawn clamps
SPAWN_XY = 200.0
SPAWN_DEPTH = (0.5, 25.0)


# ---------------------------------------------------------------------------
# 1. Validation → canonical design
# ---------------------------------------------------------------------------

def validate_design(design) -> tuple[dict | None, list[str]]:
    """Validate + canonicalise a design dict.

    Returns `(canonical, errors)`; `canonical` is None whenever `errors` is
    non-empty. The canonical form has: clamped/rounded options with defaults
    filled and unknown keys dropped, mounts reduced to occupied points only,
    a clamped spawn, and NO cosmetic `name` (so renaming a design doesn't
    bust the compile cache).
    """
    errors: list[str] = []
    if not isinstance(design, dict):
        return None, ["design must be a JSON object"]

    spine_in = design.get("spine")
    if not isinstance(spine_in, list) or len(spine_in) < 2:
        return None, ["spine must be a list of at least 2 modules "
                      "(a nose, then spine modules, then a stern)"]

    spine_out = []
    for i, el in enumerate(spine_in):
        if not isinstance(el, dict):
            errors.append(f"spine[{i}] must be an object")
            continue
        t = el.get("type")
        entry = CATALOG.get(t)
        if entry is None:
            errors.append(f"spine[{i}]: unknown module type {t!r}")
            continue
        expected = ("nose" if i == 0
                    else "rear" if i == len(spine_in) - 1 else "spine")
        if entry.kind != expected:
            errors.append(
                f"spine[{i}] must be a {expected} module "
                f"({t!r} is a {entry.kind} module)")
            continue
        opts, oerr = clamp_options(t, el.get("options") or {})
        errors.extend(oerr)
        spine_out.append({"options": opts, "type": t})

    spawn_in = design.get("spawn") or {}
    if not isinstance(spawn_in, dict):
        errors.append("spawn must be an object")
        spawn_in = {}

    def _num(key, default):
        v = spawn_in.get(key, default)
        try:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"spawn.{key}: expected a number, got {v!r}")
            return default
        return v

    spawn = {
        "depth": round(min(max(_num("depth", 2.0), SPAWN_DEPTH[0]),
                           SPAWN_DEPTH[1]), 9),
        "heading_deg": round(_num("heading_deg", 0.0) % 360.0, 9),
        # sea state is part of the compiled world → part of the canonical
        # design (and the cache hash). Cosmetic settings (visibility)
        # deliberately are NOT: the client applies them, same bundle.
        "wave_amp": round(min(max(_num("wave_amp", WAVE_AMP[2]),
                                  WAVE_AMP[0]), WAVE_AMP[1]), 9),
        "wave_len": round(min(max(_num("wave_len", WAVE_LEN[2]),
                                  WAVE_LEN[0]), WAVE_LEN[1]), 9),
        "x": round(min(max(_num("x", 0.0), -SPAWN_XY), SPAWN_XY), 9),
        "y": round(min(max(_num("y", 0.0), -SPAWN_XY), SPAWN_XY), 9),
    }

    if errors:
        return None, errors
    return {"spawn": spawn, "spine": spine_out, "version": 1}, []


def design_hash(canonical: dict) -> str:
    """First 12 hex chars of the sha256 of the canonical design (salted with
    BUILDER_REV so builder changes invalidate the bundle cache)."""
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"r{BUILDER_REV}:{blob}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 2. World construction
# ---------------------------------------------------------------------------

def build_world(canonical: dict):
    """The real-Earth polar dive site + the assembled `mako` craft.

    Returns `(world, craft, contrib, scene)`: `contrib` aggregates every
    module's analytic bookkeeping (masses, buoys, thrusters, inputs, fins)
    in spine order; `scene` is the local ENU frame at the anchor — the
    numeric glue between the huge world-frame state and the small scene
    coordinates the client renders and reasons in.
    """
    craft = Craft(CRAFT)
    contrib = Contrib()

    # Spine placement: accumulate module lengths front (+x) to back, then
    # shift so the whole spine is centred on x = 0 (SPEC "Module catalog").
    spine = canonical["spine"]
    lengths = [spine_length(el["type"], el["options"]) for el in spine]
    cursor = sum(lengths) / 2.0
    for i, (el, length) in enumerate(zip(spine, lengths)):
        xc = cursor - length / 2.0
        cursor -= length
        t = el["type"]
        contrib.merge(CATALOG[t].build(craft, f"{t}{i}", xc, el["options"]))
        # One point contact per module on the hull axis (see COL_* above:
        # against the raised seabed plane this is a HULL_R sphere).
        craft.add(Collider(f"{t}{i}_col", stiffness=COL_STIFFNESS,
                           damping=COL_DAMPING, friction=COL_FRICTION,
                           transform=(xc, 0.0, 0.0)))

    # A real Earth (SPEC "Coordinate/world conventions"): sidereal spin,
    # point-mass gravity, ISA atmosphere over a SeaWaves ocean (hydrostatic
    # pressure for the Barometer, wave orbital velocities for the drag
    # surfaces and fins). The sea-level-sphere obstacle is OFF — it would
    # wall off the dive; the solid ground is our seabed half-space instead,
    # raised HULL_R so the axis-mounted point colliders behave as
    # hull-radius spheres against the true seabed.
    sp = canonical["spawn"]
    waves = (SeaWaves(amplitude=sp["wave_amp"], wavelength=sp["wave_len"])
             if sp["wave_amp"] > 0.0 else None)
    earth = Earth(waves=waves, surface_smoothing=SURFACE_BLEND,
                  surface_collision=False)
    scene = earth.scene_at(ANCHOR)
    world = (World()
             .add_planet(earth)
             .add_field(CollisionField().add_half_space(
                 origin=(0.0, 0.0, Earth.R_EQ - SEABED_DEPTH + HULL_R))))

    # Spawn co-rotating with the planet at the scene-local design pose —
    # `at_rest` fills in the orbital velocity and body spin rate.
    spawn = canonical["spawn"]
    world.add_craft(craft, **scene.at_rest(
        (spawn["x"], spawn["y"], -spawn["depth"]),
        heading=math.radians(spawn["heading_deg"])))
    return world, craft, contrib, scene


# ---------------------------------------------------------------------------
# 3. Analytics: mass / buoyancy / trim
# ---------------------------------------------------------------------------

def analyze(contrib: Contrib) -> dict:
    """Analytic mass, net buoyancy and least-squares thruster trim.

    All in the craft (body) frame — the spawn attitude is yaw-only and
    gravity/buoyancy are along z, so the body-frame residual is
    yaw-invariant. Returns::

        {mass, buoyancy_n, trim: {<part>.throttle: N}, warnings: [...]}
    """
    warnings: list[str] = []
    mass = sum(m for m, _ in contrib.masses)
    volume = sum(v for v, _ in contrib.buoys)
    buoyancy_n = (RHO_WATER * volume - mass) * GRAVITY

    # Residual wrench at trim: gravity on every Mass + buoyancy on every
    # PointBuoy (fully-submerged assumption — the spawn depth minimum keeps
    # the hull under the blend band).
    force = np.zeros(3)
    torque = np.zeros(3)
    for m, at in contrib.masses:
        f = np.array([0.0, 0.0, -m * GRAVITY])
        force += f
        torque += np.cross(np.asarray(at, dtype=float), f)
    for v, at in contrib.buoys:
        f = np.array([0.0, 0.0, RHO_WATER * v * GRAVITY])
        force += f
        torque += np.cross(np.asarray(at, dtype=float), f)
    residual = np.concatenate([force, torque])

    # Thruster wrench matrix: column i = [axis; r × axis + ct·axis] — the
    # ct term is the prop counter-torque about its own axis. Solve
    # W·u = −residual by least squares, clip to ±max_thrust.
    trim: dict[str, float] = {}
    if contrib.thrusters:
        cols = []
        for th in contrib.thrusters:
            a = np.asarray(th["axis"], dtype=float)
            r = np.asarray(th["at"], dtype=float)
            cols.append(np.concatenate([a, np.cross(r, a) + th["ct"] * a]))
        W = np.stack(cols, axis=1)
        u, *_ = np.linalg.lstsq(W, -residual, rcond=None)
        for th, ui in zip(contrib.thrusters, u):
            lim = th["max_thrust"]
            clipped = float(np.clip(ui, -lim, lim))
            if abs(ui) > lim:
                warnings.append(
                    f"trim for {th['part']} clipped from {ui:+.1f} N to "
                    f"±{lim:.0f} N — the design cannot hold this pose")
            trim[f"{th['part']}.throttle"] = round(clipped, 6)
        left = residual + W @ np.array(
            [trim[f"{t['part']}.throttle"] for t in contrib.thrusters])
        # a leftover that is purely the designed float-up (up to the
        # reserve, +z, no torque) is intended — fins-only boats creep to
        # the surface by design. Anything else is a real imbalance.
        reserve_ok = (abs(left[0]) < 1.0 and abs(left[1]) < 1.0
                      and -1.0 < left[2] < mass * GRAVITY
                      * (BUOY_RESERVE - 1.0) + 1.0)
        if not reserve_ok or np.linalg.norm(left[3:]) > 1.0:
            warnings.append(
                f"trim leaves a residual wrench (|F| = "
                f"{np.linalg.norm(left[:3]):.1f} N, |τ| = "
                f"{np.linalg.norm(left[3:]):.1f} N·m) — the spawn pose is "
                f"not a true equilibrium")
    elif abs(buoyancy_n) > 0.5:
        warnings.append("no thrusters — the craft will drift with its "
                        "net buoyancy")

    # Every module carries the same designed reserve, so warn only when
    # the craft deviates from that float-up baseline (a future non-neutral
    # module, not the reserve itself).
    reserve_n = mass * GRAVITY * (BUOY_RESERVE - 1.0)
    if buoyancy_n > reserve_n + 5.0:
        warnings.append(f"net buoyancy +{buoyancy_n:.1f} N — vertical "
                        f"thrusters must push down to hold depth")
    elif buoyancy_n < reserve_n - 5.0:
        warnings.append(f"net buoyancy {buoyancy_n:.1f} N — the craft is "
                        f"heavy and will sink without thrust")

    # Authority hints. Count thrusters with meaningful projection on each
    # lateral axis (canted agility props count toward y): with none, that
    # translation axis has no direct actuator — surface a design hint so
    # the pilot knows why those keys are inert.
    n_side = sum(1 for t in contrib.thrusters if abs(t["axis"][1]) > 0.2)
    n_vert = sum(1 for t in contrib.thrusters if abs(t["axis"][2]) > 0.2)
    if n_side == 0:
        warnings.append("no lateral thrust (horizontal tunnel or agility "
                        "module) — no closed-loop sway")
    if n_vert == 0:
        warnings.append("no vertical thrust (vertical tunnel or agility "
                        "module) — no closed-loop depth")

    # --- the mixer (meta.mixer) ---------------------------------------------
    # The baked wrench→actuator map every client control law runs on
    # (manual keys, orientation PIDs, position PIDs): demand a body-frame
    # wrench [F; τ], get actuator commands with ONE matrix-vector product —
    # no runtime solves. Thrusters come from the Tikhonov-damped allocation
    # below; fins are speed-scheduled on the client (torque per deflection
    # ∝ q = ½ρu², zero at rest — `fins[].k` carries the coefficient). The
    # `drag` polynomials feed the client's feedforward (cancel drag at the
    # commanded rates so PIDs only carry the error). Body frame throughout —
    # actuators are body-fixed, so no attitude enters.
    mixer = None
    if contrib.thrusters and contrib.drags:
        com = (np.zeros(3) if mass <= 0.0 else
               sum(m * np.asarray(at, dtype=float)
                   for m, at in contrib.masses) / mass)
        f_lin = np.zeros(3)
        f_quad = np.zeros(3)
        t_lin = np.zeros(3)
        t_quad = np.zeros(3)
        # Translation→torque coupling about the CoM: drag surfaces sit off
        # the mass centre, so translating at v produces a weathervane
        # moment τ_d = −(cross_lin·v + cross_quad·v∘|v|). The matrices are
        # emitted in "torque you must COMMAND to hold attitude" sign (same
        # convention as force_lin/force_quad: positive = what the
        # actuators supply) — without this feedforward, sway/heave in ori
        # hold parks a 10–15° attitude error on the slow integrator.
        # Only the WEATHERVANE entries survive (yaw←surge/sway, pitch←
        # heave): those are set by the strong bow/stern drag distribution
        # and verified against the compiled sim (sway yaw error 15°→3°).
        # The rest (roll row, pitch←surge) ride on small vertical drag
        # offsets that are the same order as the unmodeled metacentric
        # righting — commanding them MEASURED WORSE than feedback alone
        # (sway rolled −21°, surge pitched ±8°), so they stay zero.
        c_lin = np.zeros((3, 3))
        c_quad = np.zeros((3, 3))
        cross_keep = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [1, 1, 0]], dtype=float)
        eye = np.eye(3)
        for d in contrib.drags:
            k = np.asarray(d["k"], dtype=float)      # per-axis ½·area·cd
            r = np.asarray(d["at"], dtype=float)
            f_lin += RHO_WATER * V_LIN * k
            f_quad += RHO_WATER * k
            r_com = r - com
            for a in range(3):
                rxe = np.cross(r_com, eye[a])
                c_lin[:, a] += RHO_WATER * V_LIN * k[a] * rxe
                c_quad[:, a] += RHO_WATER * k[a] * rxe
            for j in range(3):
                # rotation about axis j moves the surface along the OTHER
                # two axes (a, b) at ω·r_b / ω·r_a — each resisted by that
                # axis's own drag coefficient
                a, b = [(1, 2), (2, 0), (0, 1)][j]
                lever = k[a] * r[b] ** 2 + k[b] * r[a] ** 2
                t_lin[j] += RHO_WATER * V_LIN * lever
                t_quad[j] += RHO_WATER * (k[a] * abs(r[b]) ** 3
                                          + k[b] * abs(r[a]) ** 3)
        # Tikhonov-damped allocation (n_thrusters × 6): a plain pinv routes
        # demand through ANY nonzero column — e.g. a lone prop's 0.015
        # counter-torque would draw ~67 N of thrust per N·m of yaw demand.
        # The damping ignores such token authority; strong axes are
        # unaffected (their singular values ≫ λ).
        lam = 0.01
        alloc = W.T @ np.linalg.inv(W @ W.T + lam * np.eye(6))
        # Usable wrench authority per axis: summed |W| columns × max
        # thrust, derated for feedback headroom; heave also carries the
        # static buoyancy residual.
        frac = 0.6
        cap = np.abs(W) @ np.array([t["max_thrust"]
                                    for t in contrib.thrusters])
        cap[2] = max(0.0, cap[2] - abs(buoyancy_n))
        avail = frac * cap
        # Fins: torque per (m/s)²·rad about their steer axis, τ ≈
        # ½ρ·A·CLα·|x_arm|·FIN_EFF. Ideal thin-airfoil CLα = 2π overstates
        # the manta ControlSurface by ~7× (finite aspect ratio + flap
        # effectiveness); FIN_EFF is MEASURED against the compiled sim
        # (defl 0.1–0.3 rad at 4.4 m/s: k_true 2.80–2.93 vs analytic 19.3,
        # linear through 0.3 rad). normal "y" fins (vertical pair) steer
        # yaw; normal "z" fins (horizontal pair) steer pitch.
        FIN_EFF = 0.147
        # Per-fin SIGNED roll coefficient: lift at radial arm r_off makes
        # roll torque roll_k·u²·defl (N·m per rad per (m/s)²). Lift per
        # +defl points along +normal (sign fixed by the same open-loop
        # probes that measured FIN_EFF and the yaw/pitch fin signs), so a
        # lift-y fin at z = r_off rolls at −r_off·L and a lift-z fin at
        # y = r_off rolls at +r_off·L. Opposing signs across a pair =
        # differential deflection → pure roll, collective → pure yaw/pitch.
        fins_meta = [{
            "input": f"{CRAFT}.{f['input']}",
            "axis": "yaw" if f["normal"] == "y" else "pitch",
            "k": round(0.5 * RHO_WATER * f["area"] * 2.0 * math.pi
                       * abs(f["x"]) * FIN_EFF, 4),
            "roll_k": round(0.5 * RHO_WATER * f["area"] * 2.0 * math.pi
                            * FIN_EFF
                            * (-f["r_off"] if f["normal"] == "y"
                               else f["r_off"]), 4),
            "max_defl": round(f["max_defl"], 6),
        } for f in contrib.fins]
        # Which axes the design can actually drive (thrusters at rest;
        # rotation axes also credit fins-at-speed). The client uses this
        # for the "no actuator on that axis" key warnings.
        has_pitch_fin = any(f["axis"] == "pitch" for f in fins_meta)
        has_yaw_fin = any(f["axis"] == "yaw" for f in fins_meta)
        # roll needs a differential pair: roll_k of BOTH signs
        has_roll_fin = (any(f["roll_k"] > 0 for f in fins_meta)
                        and any(f["roll_k"] < 0 for f in fins_meta))
        axes = {
            "surge": bool(avail[0] > 1.0),
            "sway":  bool(avail[1] > 1.0),
            "heave": bool(avail[2] > 1.0),
            # roll needs > a lone prop's token counter-torque (~2 N·m)
            "roll":  bool(avail[3] > 3.0 or has_roll_fin),
            "pitch": bool(avail[4] > 0.5 or has_pitch_fin),
            "yaw":   bool(avail[5] > 0.5 or has_yaw_fin),
        }
        mixer = {
            "inputs": [f"{CRAFT}.{t['part']}.throttle"
                       for t in contrib.thrusters],
            "alloc": [[round(float(v), 6) for v in row] for row in alloc],
            "wrench_cap": [round(float(v), 3) for v in avail],
            "fins": fins_meta,
            "axes": axes,
            "drag": {
                "force_lin": [round(float(v), 4) for v in f_lin],
                "force_quad": [round(float(v), 4) for v in f_quad],
                "torque_lin": [round(float(v), 4) for v in t_lin],
                "torque_quad": [round(float(v), 4) for v in t_quad],
                "cross_lin": [[round(float(v), 4) for v in row]
                              for row in c_lin * cross_keep],
                "cross_quad": [[round(float(v), 4) for v in row]
                               for row in c_quad * cross_keep],
            },
        }

    # --- achievable-rate scales (meta.rate_scale) ---------------------------
    # The client maps its translation keys onto THIS design's steady-rate
    # capability — nothing is baked for one configuration. Per axis: usable
    # authority `avail` (from the mixer block), then solve
    # k_quad·r² + k_lin·r = authority for the steady rate r.
    rate_scale = None
    if mixer is not None:

        def steady(k_lin: float, k_quad: float, f: float) -> float:
            if f <= 0.0 or (k_lin <= 0.0 and k_quad <= 0.0):
                return 0.0
            if k_quad <= 0.0:
                return f / k_lin
            return (-k_lin + math.sqrt(k_lin ** 2 + 4.0 * k_quad * f)) \
                / (2.0 * k_quad)

        d = mixer["drag"]
        surge = steady(d["force_lin"][0], d["force_quad"][0], avail[0])

        # Fins have ZERO authority at rest, but at speed they are a
        # fins-only design's entire steering. Credit them here at a
        # reference speed so the client's key scales — and its fin
        # steering laws — have range:
        # vertical fins add yaw torque; horizontal fins let heave keys
        # command a pitch-to-climb (scale ≈ climb rate at ~12° at speed).
        u_ref = 0.6 * surge
        q_ref = 0.5 * RHO_WATER * u_ref ** 2
        yaw_fin = sum(
            q_ref * f["area"] * min(2.0 * math.pi * f["max_defl"], 1.1)
            * abs(f["x"]) for f in contrib.fins if f["normal"] == "y") * 0.6
        heave = steady(d["force_lin"][2], d["force_quad"][2], avail[2])
        if heave < 0.15 and any(f["normal"] == "z" for f in contrib.fins):
            heave = round(0.2 * u_ref, 3)

        rate_scale = {
            "surge": round(surge, 3),
            "sway":  round(steady(d["force_lin"][1], d["force_quad"][1],
                                  avail[1]), 3),
            "heave": round(heave, 3),
            "yaw":   round(steady(d["torque_lin"][2], d["torque_quad"][2],
                                  avail[5] + yaw_fin), 3),
        }

    return {"mass": round(mass, 3), "buoyancy_n": round(buoyancy_n, 3),
            "trim": trim, "mixer": mixer, "rate_scale": rate_scale,
            "warnings": warnings}


# ---------------------------------------------------------------------------
# 4 + 5. Transforms → bundle dir → meta.json
# ---------------------------------------------------------------------------

def _tangent_dim(slot: dict) -> int:
    """A state slot's tangent dimension (quaternion 4 → 3)."""
    return 3 if slot["dim"] == 4 else slot["dim"]


def _emit(transform, out_dir: Path, *, class_name: str, basename: str):
    """TargetWasm emission + rename build.sh → build_<basename>.sh (three
    bundles share one directory; everything else is basename-prefixed)."""
    res = TargetWasm(transform, out_dir, class_name=class_name,
                     basename=basename)
    script = out_dir / f"build_{basename}.sh"
    (out_dir / "build.sh").replace(script)
    return res, script


def build_bundle(canonical: dict, out_dir: str | Path, *,
                 name: str = "mako", compile: bool = True) -> dict:
    """Build the bundle for a canonical design into `out_dir`.

    Emits the Sim sources, optionally compiles with emcc, and writes
    `meta.json` last. Returns the meta dict (with an extra non-SPEC
    `"timings"` entry when compiled). Raises on hard failures (emission,
    emcc errors).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world, craft, contrib, scene = build_world(canonical)
    analysis = analyze(contrib)
    warnings = list(analysis["warnings"])
    scripts: list[Path] = []
    timings: dict[str, float] = {}

    # --- Sim (the one mandatory transform) --------------------------------
    t0 = time.monotonic()
    sim_res, script = _emit(Sim(world), out_dir,
                            class_name="MakoSim", basename="sim")
    scripts.append(script)
    timings["sim_emit_s"] = round(time.monotonic() - t0, 2)
    desc = sim_res.descriptor_obj
    files = {"sim": "sim.js"}

    # meta.inputs in the descriptor's input order, limits from the catalog.
    limits = {i["name"]: (i["min"], i["max"]) for i in contrib.inputs}
    inputs_meta = []
    for f in desc["inputs"]:
        # descriptor names are craft-qualified ("mako.<part>.<input>");
        # the catalog records them part-local.
        local = f["name"].removeprefix(f"{CRAFT}.")
        lo, hi = limits.get(local, (None, None))
        if lo is None:
            raise RuntimeError(f"input {f['name']} missing from the catalog "
                               f"bookkeeping — builder bug")
        inputs_meta.append({"name": f["name"], "min": lo, "max": hi})

    # --- inertia (meta.inertia) ----------------------------------------------
    # Diagonal craft-frame inertia about the origin, module cylinders via
    # parallel axis — the same per-module model catalog._body registers
    # (axial 0.8·m·r², transverse m(3r² + L²)/12), so the client's attitude
    # PIDs can size their gains as τ = I·ω̇ instead of guessing.
    inertia = np.zeros(3)
    lengths = [spine_length(el["type"], el["options"])
               for el in canonical["spine"]]
    for (m, at), length in zip(contrib.masses, lengths):
        i_ax = 0.8 * m * HULL_R ** 2
        i_tr = m * (3.0 * HULL_R ** 2 + length ** 2) / 12.0
        x, y, z = at
        inertia += [i_ax + m * (y * y + z * z),
                    i_tr + m * (x * x + z * z),
                    i_tr + m * (x * x + y * y)]

    # --- scene / waves (the client's world→ENU + surface math) --------------
    # Ship the constants for `Scene.relative` in JS: p_scene =
    # R_ws(t)ᵀ·(p_world − o(t)) with R_ws(t) = R_axis(ωt)·R_ps and o(t) =
    # R_axis(ωt)·anchor + center. At the pole o(t) is invariant, but the
    # client implements the general form.
    planet = scene.planet
    scene_meta = {
        "omega": planet.omega,
        "axis": [float(v) for v in planet.axis],
        "center": [float(v) for v in planet.center],
        "anchor_planet": [float(v) for v in scene.anchor_planet],
        "R_planet_from_scene": [[round(float(v), 12) for v in row]
                                for row in scene.R_planet_from_scene],
        "g0": round(G0, 6),
    }
    # The exact surface the physics uses: η = A·cos(k·ξ − ω_w·t), ξ =
    # p_planet·dir, deep-water dispersion ω_w = k·√(g0·λ/2π) (mirrors
    # Earth.register_disturbances).
    wave_amp = canonical["spawn"]["wave_amp"]
    wave_len = canonical["spawn"]["wave_len"]
    k_wave = 2.0 * math.pi / wave_len
    waves_meta = {
        "amplitude": wave_amp,
        "wavelength": wave_len,
        "k": round(k_wave, 9),
        "omega": round(k_wave * math.sqrt(G0 * wave_len
                                          / (2.0 * math.pi)), 9),
        "dir_planet": [1.0, 0.0, 0.0],
    }

    # --- compile ---------------------------------------------------------------
    if compile:
        compile_scripts(scripts, out_dir, timings=timings)

    spawn = canonical["spawn"]
    meta = {
        "version": 2,
        "name": name,
        "craft": CRAFT,
        "files": files,
        "inputs": inputs_meta,
        "scene": scene_meta,
        "waves": waves_meta,
        "seabed_z": -SEABED_DEPTH,
        "hull_r": HULL_R,
        "spawn": {"scene_position": [spawn["x"], spawn["y"],
                                     -spawn["depth"]],
                  "heading_deg": spawn["heading_deg"]},
        "mass": analysis["mass"],
        "inertia": [round(float(v), 4) for v in inertia],
        "buoyancy_n": analysis["buoyancy_n"],
        "trim": analysis["trim"],
        "mixer": analysis["mixer"],
        "rate_scale": analysis["rate_scale"],
        "warnings": warnings,
        # The canonicalised design, so a bundle is self-describing: the
        # client rebuilds the 3D meshes from it when booting straight from
        # a cached bundle (`/mako/?load=<hash>`).
        "design": canonical,
    }
    if timings:
        meta["timings"] = timings
    # meta.json is the cache marker — written LAST, only for a bundle that
    # emitted (and, when requested, compiled) end to end.
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# Kernels above this size compile with -O0 instead of the build script's
# default -O3, then get a Binaryen `wasm-opt -O2` post-pass. CasADi emits
# each kernel as one enormous straight-line function; clang's optimizers are
# superlinear in that, so multi-MB kernels (a long, sensor-rich spine) take
# minutes at -O1+ versus ~13 s at -O0. But raw -O0 code runs ~50× slower —
# past the browser's real-time budget. wasm-opt recovers it: it is
# near-linear on huge functions (~25 s for a 5 MB module) and its -O2
# output benchmarks within ~1.4× of LLVM's -O1. Extra flags after the
# script's baked -O3 win — the last -O on the emcc command line applies.
# 300 KB: the Earth-world sim kernels run 0.5–1.5 MB (ISA atmosphere +
# waving ocean evaluated at every drag surface/buoy/fin) and measure 184 s
# at -O3 for 0.87 MB vs ~20 s down this path; only trivial kernels stay on
# -O3.
_O0_KERNEL_BYTES = 300_000


def _wasm_opt() -> str | None:
    """Binaryen's wasm-opt, expected beside emcc in the emsdk tree."""
    exe = shutil.which("wasm-opt")
    if exe:
        return exe
    emcc = shutil.which("emcc")
    if emcc:
        cand = Path(emcc).resolve().parent.parent / "bin" / "wasm-opt"
        if cand.exists():
            return str(cand)
    return None


def compile_scripts(scripts: list[Path], out_dir: Path, *,
                    timings: dict | None = None) -> None:
    """Run each bundle's emcc build script, in parallel threads."""
    if not shutil.which("emcc"):
        raise RuntimeError(
            "emcc is not on PATH — `source ~/emsdk/emsdk_env.sh` and "
            "restart the server")

    def _run(script: Path) -> None:
        base = script.stem.removeprefix("build_")
        # 4 MB wasm stack: the generated ABI shims stack-allocate the CasADi
        # work arrays (`double w[SZ_W]`), which for a sensor-rich design can
        # exceed Emscripten's 64 KB default — that crashes at runtime with
        # "memory access out of bounds".
        cmd = ["sh", script.name, "-s", "STACK_SIZE=4194304"]
        kernels = out_dir / f"{base}_kernels.c"
        post_opt = (kernels.exists()
                    and kernels.stat().st_size > _O0_KERNEL_BYTES)
        if post_opt:
            cmd.append("-O0")
        t0 = time.monotonic()
        r = subprocess.run(cmd, cwd=str(out_dir),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"{script.name} failed:\n{r.stderr[-2000:]}")
        if timings is not None:
            timings[f"{base}_emcc_s"] = round(time.monotonic() - t0, 2)
        if post_opt and (opt := _wasm_opt()):
            t0 = time.monotonic()
            wasm = out_dir / f"{base}.wasm"
            r = subprocess.run(
                [opt, "-O2", "--enable-mutable-globals",
                 str(wasm), "-o", str(wasm)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"wasm-opt on {wasm.name} failed:\n{r.stderr[-2000:]}")
            if timings is not None:
                timings[f"{base}_wasm_opt_s"] = round(time.monotonic() - t0, 2)

    with ThreadPoolExecutor(max_workers=max(1, len(scripts))) as pool:
        for fut in [pool.submit(_run, s) for s in scripts]:
            fut.result()
