// The mako's client-side control core — pure math, no DOM/THREE, so the
// whole thing runs headless in node for verification.
//
// Everything sits on the SAME baked mixing the pilot touches directly in
// manual mode (`meta.mixer`): a wrench demand [F; τ] (body frame, N / N·m)
// becomes actuator commands with one n×6 matrix-vector product for the
// thrusters plus a speed-scheduled share for the fins (their torque per
// deflection is k·u² — nothing at rest, dominant at speed). The three
// modes only differ in WHO produces the wrench:
//
//   manual   — keys ARE the wrench (fractions of the design's authority).
//   ori hold — attitude keys slew an orientation setpoint; a quaternion
//              log-error PD (+ slew-rate feedforward, gains sized from
//              meta.inertia as τ = I·ω̇) produces the torque demand.
//              Translation keys stay manual force.
//   auto     — keeps the ori-hold attitude loop; translation keys move a
//              position setpoint and a cascade (position → velocity →
//              force, clamped by meta.rate_scale, drag feedforward from
//              meta.mixer.drag) produces the force demand.
//
// Thruster commands pass through a first-order low-pass (τ = 60 ms —
// electric pods respond fast, but not in one control tick); fins skip it
// (the ControlSurface state already models the servo lag).

// ---------------------------------------------------------------------------
// Quaternion / vector helpers ([w, x, y, z], plain arrays)
// ---------------------------------------------------------------------------

export const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];

export function qmul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export function qrot(q, v) {          // rotate v by q
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + y * tz - z * ty,
          v[1] + w * ty + z * tx - x * tz,
          v[2] + w * tz + x * ty - y * tx];
}

export function qaxis(axis, angle) {  // unit axis, radians
  const h = angle / 2, s = Math.sin(h);
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

export function qfromRotmat(R) {      // 3×3 row-major nested → quat
  const tr = R[0][0] + R[1][1] + R[2][2];
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = s / 4; x = (R[2][1] - R[1][2]) / s;
    y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    w = (R[2][1] - R[1][2]) / s; x = s / 4;
    y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s;
    y = s / 4; z = (R[1][2] + R[2][1]) / s;
  } else {
    const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s;
    y = (R[1][2] + R[2][1]) / s; z = s / 4;
  }
  return [w, x, y, z];
}

/** Euler ZYX (yaw about z, pitch about y, roll about x) → quat.
 *  Axes: x forward, y LEFT, z up — positive pitch is nose DOWN. */
export function qeuler(roll, pitch, yaw) {
  return qmul(qaxis([0, 0, 1], yaw),
              qmul(qaxis([0, 1, 0], pitch), qaxis([1, 0, 0], roll)));
}

/** quat → {roll, pitch, yaw} (same ZYX convention). */
export function rpyOf(q) {
  const [w, x, y, z] = q;
  return {
    roll: Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    pitch: Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))),
    yaw: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
  };
}

/** Body-frame rotation vector taking `q` to `qTarget` (quaternion log of
 *  the error, exact — no small-angle assumption). */
export function attitudeError(q, qTarget) {
  let e = qmul(qconj(q), qTarget);
  if (e[0] < 0) e = e.map((v) => -v);          // shortest arc
  const n = Math.hypot(e[1], e[2], e[3]);
  if (n < 1e-9) return [0, 0, 0];
  const angle = 2 * Math.atan2(n, e[0]);
  return [e[1] / n * angle, e[2] / n * angle, e[3] / n * angle];
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---------------------------------------------------------------------------
// Scene frame — the JS port of manta's Scene.relative (meta.scene)
// ---------------------------------------------------------------------------

/** World-frame sim state → small scene-ENU state. The planet spins about
 *  `axis` at `omega`; the scene is rigid in its body frame. */
export function makeSceneFrame(sc) {
  const qPS = qfromRotmat(sc.R_planet_from_scene);
  const { omega, axis, center, anchor_planet: anchor } = sc;
  const omegaW = [axis[0] * omega, axis[1] * omega, axis[2] * omega];
  return {
    omega,
    /** {qWS, origin} of the scene at time t. */
    at(t) {
      const qA = qaxis(axis, omega * t);
      const o = qrot(qA, anchor);
      return { qWS: qmul(qA, qPS),
               origin: [o[0] + center[0], o[1] + center[1], o[2] + center[2]] };
    },
    /** Re-express {p, q, v, w} (world pos, world-from-craft quat, world
     *  velocity, BODY rates) in scene coordinates; velocity/rates come out
     *  relative to the co-rotating ground. */
    toScene(p, q, v, w, t) {
      const { qWS, origin } = this.at(t);
      const qSW = qconj(qWS);
      const rel = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
      const rC = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
      const vRel = [v[0] - (omegaW[1] * rC[2] - omegaW[2] * rC[1]),
                    v[1] - (omegaW[2] * rC[0] - omegaW[0] * rC[2]),
                    v[2] - (omegaW[0] * rC[1] - omegaW[1] * rC[0])];
      const wEarthBody = qrot(qconj(q), omegaW);
      return {
        p: qrot(qSW, rel),
        q: qmul(qSW, q),                       // scene-from-craft
        v: qrot(qSW, vRel),
        w: [w[0] - wEarthBody[0], w[1] - wEarthBody[1],
            w[2] - wEarthBody[2]],             // body rates vs the ground
      };
    },
  };
}

// ---------------------------------------------------------------------------
// The controller
// ---------------------------------------------------------------------------

export const MODES = ["manual", "ori", "auto"];

// Attitude loop: critically damped PID, sized from meta.inertia (τ = I·ω̇).
// The integrator is what holds pitch at speed — the fins weathervane and
// the hull carries a Munk moment, a steady load a PD alone leaves as a
// ~10°+ droop (measured in the headless harness).
const ATT_WN = 1.8;                   // rad/s closed-loop bandwidth
const ATT_KP = ATT_WN * ATT_WN;       // 1/s²
const ATT_KD = 2.0 * ATT_WN;          // 1/s
const ATT_KI = 0.9;                   // 1/s³ (× inertia)
const ATT_INT_FRAC = 0.5;             // integral torque ≤ this × wrench_cap
const ATT_ERR_MAX = 0.7;              // rad — bound the P kick after a slam
const SLEW = { roll: 1.2, pitch: 1.0, yaw: 1.2 };   // setpoint rad/s
const ROLL_RELEVEL = 0.8;             // rad/s the roll setpoint decays at
const PITCH_SP_MAX = 1.35;            // rad — keep clear of gimbal lock

// Position cascade (auto): error → velocity (clamped by rate_scale) →
// force, with drag feedforward at the desired velocity.
const POS_KP = 0.7;                   // 1/s
const VEL_KP = 1.6;                   // 1/s (force = m·VEL_KP·Δv)

const THR_LPF_TAU = 0.06;             // s — electric thruster response
const MANUAL_FRAC = 1.0;              // manual wrench = frac · wrench_cap

export function createController({ meta, onWarn = () => {} }) {
  const mx = meta.mixer;
  const caps = mx ? mx.wrench_cap : [0, 0, 0, 0, 0, 0];
  const axes = mx ? mx.axes : {};
  const inertia = meta.inertia ?? [1, 10, 10];
  const mass = meta.mass ?? 60;
  const rs = meta.rate_scale ?? { surge: 1, sway: 1, heave: 1, yaw: 1 };

  // Fin signs, MEASURED against the compiled sim (open-loop deflection
  // probes at cruise): vertical pair +defl → NEGATIVE yaw torque;
  // horizontal pair +defl → POSITIVE pitch (nose down, +0.2 rad ≈ +14°).
  const FIN_SIGN = { yaw: -1, pitch: +1 };
  const finsByAxis = { pitch: [], yaw: [] };
  for (const f of mx?.fins ?? []) finsByAxis[f.axis].push(f);
  // roll comes from DIFFERENTIAL deflection across the set — each fin
  // carries a signed roll_k (meta), opposing signs across a pair
  const finsRoll = (mx?.fins ?? []).filter((f) => f.roll_k);

  // torque capacity of a fin group at dynamic pressure ∝ u² (linear
  // through ~0.3 rad → credit 90 % of the hardware deflection)
  const finCap = (fins, u2, key) => {
    let c = 0;
    for (const f of fins) c += Math.abs(f[key]) * u2 * f.max_defl * 0.9;
    return c;
  };
  // Fin-steered axes are dominated by the hull's own weathervane/parasitic
  // damping — a deflection sustains a RATE, not a torque (torpedo yaw,
  // measured: c/I ≈ 22 s⁻¹ vs the controller's KD = 3.6, so a τ = I·ω̇
  // sized deflection turns ~8× slower than modeled). Deflections get
  // boosted by (KD + c/I)/KP ≈ 8, blended by the fin share so thruster-
  // dominated axes keep the plain torque law. The ±0.9·max_defl clamp
  // bounds the aggression.
  const FIN_AGGR = 8;
  const finBoost = (share) => 1 + (FIN_AGGR - 1) * share;
  // fins-at-speed torque authority per rotation axis [roll, pitch, yaw]
  const finTorqueCaps = (u2) => [
    finCap(finsRoll, u2, "roll_k"),
    finCap(finsByAxis.pitch, u2, "k"),
    finCap(finsByAxis.yaw, u2, "k"),
  ];

  // setpoints
  const spawnYaw = (meta.spawn.heading_deg ?? 0) * Math.PI / 180;
  const sp = {
    roll: 0, pitch: 0, yaw: spawnYaw,
    pos: [...meta.spawn.scene_position],
  };

  let mode = "manual";
  const uLP = {};                     // low-passed thruster commands
  const warned = {};                  // per-axis warn edge detect
  const intE = [0, 0, 0];             // attitude integral state (body)

  // Axis demand from the key map, each in [-1, 1]:
  //   {surge, sway, heave, roll, pitch, yaw}
  function checkWarn(a) {
    for (const [axis, v] of Object.entries(a)) {
      if (!v) { warned[axis] = false; continue; }
      if (!axes[axis] && !warned[axis]) {
        warned[axis] = true;
        onWarn(axis);
      }
    }
  }

  /** Wrench [F;τ] (body) → actuator dict. Fins take a share of the torque
   *  that grows with dynamic pressure; thrusters get the rest. Pitch/yaw
   *  deflect a pair collectively; roll deflects the set differentially
   *  (least-squares on the signed roll_k), and the two superpose. */
  function mix(wrench, vbx) {
    const u = {};
    if (!mx) return u;
    const u2 = vbx * vbx;
    const w = wrench.slice();
    const defl = {};                  // accumulated per-fin deflection
    for (const axis of ["pitch", "yaw"]) {
      const fins = finsByAxis[axis];
      if (!fins.length) continue;
      const wi = axis === "pitch" ? 4 : 5;
      let kSum = 0;
      for (const f of fins) kSum += f.k * u2;
      const capF = finCap(fins, u2, "k");
      const share = capF / (capF + caps[wi] + 1e-9);
      const tauFin = w[wi] * share;
      w[wi] -= tauFin;
      if (kSum > 1e-6) {
        const d = FIN_SIGN[axis] * tauFin * finBoost(share) / kSum;
        for (const f of fins) defl[f.input] = (defl[f.input] ?? 0) + d;
      }
    }
    if (finsRoll.length >= 2) {
      let k2Sum = 0;
      for (const f of finsRoll) k2Sum += f.roll_k * f.roll_k;
      const capF = finCap(finsRoll, u2, "roll_k");
      const share = capF / (capF + caps[3] + 1e-9);
      const tauFin = w[3] * share;
      w[3] -= tauFin;
      if (u2 * k2Sum > 1e-9)
        for (const f of finsRoll)
          defl[f.input] = (defl[f.input] ?? 0)
            + f.roll_k * tauFin * finBoost(share) / (u2 * k2Sum);
    }
    for (const f of mx.fins ?? [])
      if (f.input in defl)
        u[f.input] = clamp(defl[f.input],
                           -f.max_defl * 0.9, f.max_defl * 0.9);
    mx.inputs.forEach((name, i) => {
      let s = 0;
      for (let j = 0; j < 6; j++) s += mx.alloc[i][j] * w[j];
      u[name] = s;
    });
    return u;
  }

  function dragFF(vDes) {             // body-frame drag at the desired vel
    const d = mx?.drag;
    if (!d) return [0, 0, 0];
    return [0, 1, 2].map((j) =>
      d.force_lin[j] * vDes[j]
      + d.force_quad[j] * vDes[j] * Math.abs(vDes[j]));
  }

  /** One control tick. `rel` = scene-frame {p, q, v, w} (from
   *  makeSceneFrame.toScene), `a` = key axis demands, `dt` seconds.
   *  Returns the actuator command dict (thrusters low-passed). */
  function update(rel, a, dt) {
    checkWarn(a);
    const qb = qconj(rel.q);
    const vb = qrot(qb, rel.v);                 // body-frame ground-rel vel
    // torque authority at the CURRENT speed (fins are dead at rest, the
    // whole steering at cruise) — manual keys and the integral clamp
    // scale against this, so fins-only designs aren't capped to the
    // thrusters' token torques
    const fCaps = finTorqueCaps(vb[0] * vb[0]);
    const tCaps = fCaps.map((c, j) => caps[3 + j] + c);
    const wrench = [0, 0, 0, 0, 0, 0];
    let omegaFF = [0, 0, 0];

    if (mode === "manual") {
      wrench[0] = a.surge * caps[0] * MANUAL_FRAC;
      wrench[1] = a.sway * caps[1] * MANUAL_FRAC;
      wrench[2] = a.heave * caps[2] * MANUAL_FRAC;
      wrench[3] = a.roll * tCaps[0] * MANUAL_FRAC;
      wrench[4] = a.pitch * tCaps[1] * MANUAL_FRAC;
      wrench[5] = a.yaw * tCaps[2] * MANUAL_FRAC;
    } else {
      // --- attitude setpoint slew (shared by ori + auto) ---------------
      if (axes.roll) sp.roll += a.roll * SLEW.roll * dt;
      if (!a.roll) {                            // auto-level when idle
        const d = ROLL_RELEVEL * dt;
        sp.roll = Math.abs(sp.roll) < d ? 0 : sp.roll - Math.sign(sp.roll) * d;
      }
      if (axes.pitch)
        sp.pitch = clamp(sp.pitch + a.pitch * SLEW.pitch * dt,
                         -PITCH_SP_MAX, PITCH_SP_MAX);
      if (axes.yaw) sp.yaw += a.yaw * SLEW.yaw * dt;
      const qSp = qeuler(sp.roll, sp.pitch, sp.yaw);

      // --- attitude PD + slew feedforward, τ = I·ω̇ ----------------------
      const e = attitudeError(rel.q, qSp)
        .map((v) => clamp(v, -ATT_ERR_MAX, ATT_ERR_MAX));
      omegaFF = [a.roll * SLEW.roll, a.pitch * SLEW.pitch, a.yaw * SLEW.yaw];
      for (let j = 0; j < 3; j++) {
        // conditional integration: freeze while the error is still slew-
        // limited large (anti-windup), clamp the integral torque share
        if (Math.abs(e[j]) < 0.35) {
          // mix() boosts the fin share of this torque by finBoost — the
          // wind-up limit shrinks by the same factor
          const lim = ATT_INT_FRAC * (tCaps[j] + 1e-9)
            / (inertia[j] * ATT_KI
               * finBoost(fCaps[j] / (tCaps[j] + 1e-9)));
          intE[j] = clamp(intE[j] + e[j] * dt, -lim, lim);
        }
        wrench[3 + j] = inertia[j] *
          (ATT_KP * e[j] + ATT_KI * intE[j]
           + ATT_KD * (omegaFF[j] - rel.w[j]));
      }
      // Weathervane feedforward: drag surfaces sit off the CoM, so
      // translating produces a steady moment (meta cross matrices, τ =
      // cross_lin·v + cross_quad·v∘|v| in "torque to command" sign).
      // Cancel it at source — without this, sway/heave parks a 10–15°
      // error on the integrator (measured on the starter). Faded out by
      // the fin share: on a fin-steered axis this same moment is the
      // tail's stabilizing stiffness, already priced into FIN_AGGR —
      // cancelling it there makes slews overshoot (measured).
      const cl = mx?.drag?.cross_lin, cq = mx?.drag?.cross_quad;
      if (cl) {
        for (let j = 0; j < 3; j++) {
          let tau = 0;
          for (let a = 0; a < 3; a++)
            tau += cl[j][a] * vb[a]
              + cq[j][a] * vb[a] * Math.abs(vb[a]);
          wrench[3 + j] += tau * (1 - fCaps[j] / (tCaps[j] + 1e-9));
        }
      }

      if (mode === "ori") {
        wrench[0] = a.surge * caps[0] * MANUAL_FRAC;
        wrench[1] = a.sway * caps[1] * MANUAL_FRAC;
        wrench[2] = a.heave * caps[2] * MANUAL_FRAC;
      } else {
        // --- auto: translation keys move the position setpoint ---------
        const fwd = qrot(qSp, [1, 0, 0]);
        const left = qrot(qSp, [0, 1, 0]);
        sp.pos[0] += (a.surge * fwd[0] * rs.surge + a.sway * left[0] * rs.sway) * dt;
        sp.pos[1] += (a.surge * fwd[1] * rs.surge + a.sway * left[1] * rs.sway) * dt;
        sp.pos[2] += (a.surge * fwd[2] * rs.surge + a.sway * left[2] * rs.sway
                      + a.heave * rs.heave) * dt;
        sp.pos[2] = Math.min(sp.pos[2], 1.0);   // no flying
        // position → velocity (clamped per body axis) → force + drag FF
        const eP = qrot(qb, [sp.pos[0] - rel.p[0],
                             sp.pos[1] - rel.p[1],
                             sp.pos[2] - rel.p[2]]);
        const vLim = [rs.surge || 0.3, Math.max(rs.sway, 0.2 * rs.surge) || 0.3,
                      Math.max(rs.heave, 0.2 * rs.surge) || 0.3];
        const vDes = eP.map((v, j) =>
          clamp(POS_KP * v, -vLim[j], vLim[j]));
        const ff = dragFF(vDes);
        for (let j = 0; j < 3; j++)
          wrench[j] = mass * VEL_KP * (vDes[j] - vb[j]) + ff[j];
      }
    }

    const u = mix(wrench, vb[0]);
    // Low-pass the thruster newtons (fins keep their own servo model).
    const alpha = 1 - Math.exp(-dt / THR_LPF_TAU);
    for (const name of mx?.inputs ?? []) {
      uLP[name] = (uLP[name] ?? 0) + ((u[name] ?? 0) - (uLP[name] ?? 0)) * alpha;
      u[name] = uLP[name];
    }
    return u;
  }

  return {
    update,
    get mode() { return mode; },
    setMode(m) {
      if (!MODES.includes(m)) throw new Error(`bad mode ${m}`);
      mode = m;
      return mode;
    },
    /** Re-seat the setpoints on the craft's CURRENT pose (called on mode
     *  entry so the controller doesn't lunge at a stale target). */
    syncSetpoints(rel) {
      const r = rpyOf(rel.q);
      sp.roll = r.roll; sp.pitch = r.pitch; sp.yaw = r.yaw;
      sp.pos = [...rel.p];
      intE[0] = intE[1] = intE[2] = 0;
    },
    get setpoints() {
      return { roll: sp.roll, pitch: sp.pitch, yaw: sp.yaw,
               q: qeuler(sp.roll, sp.pitch, sp.yaw), pos: [...sp.pos] };
    },
    get axes() { return axes; },
  };
}
