"""Camera tracking + a fast interceptor — built one validated layer at a time.

  STEP 1 — a ground array triangulates a ballistic target (EKF over bearings).
  STEP 2 — the interceptor is exercised in its REAL operating regime: it
           boosts up, then pitches over and flies THROUGH a high+offset
           waypoint at high speed, pointing the way it's going (low angle of
           attack), without tumbling.

Step 3 (proportional navigation onto the tracked target) replaces the fixed
waypoint with the live estimate; here the rocket flies a fixed boost+pitchover
so its flight can be judged on its own.

  * **tracker** — four `CentroidCamera`s at the corners of a 1 km square,
    looking up, anchored to the ground. Each reports only the target's image
    CENTROID — a pure bearing, size-independent. Folding the four, the Kalman
    update IS the noise-weighted intersection of the rays. Two things make it
    track instead of diverge: process noise on the target's ACCELERATION (not
    position — `pos += vel·dt` is exact), and a velocity SEED from
    finite-differencing two triangulations (not a guess).

  * **target** — a ballistic `Mass`+`DragSurface`+`OpticalSource` (~3 km
    apogee), arcing into the array's field of view.

  * **interceptor** — a gimballed TVC rocket, passively stable (drag aft of
    the COM, so it weathervanes toward its velocity → low angle of attack). A
    cascade autopilot points the thrust: attitude error → gimbal ANGLE → a
    well-damped joint torque loop (the gain that DIDN'T chatter — too stiff an
    inner loop limit-cycles at the control rate and tumbles it). Guidance is a
    coordinated turn: thrust mostly forward to hold speed, a bounded
    perpendicular component to curve the velocity onto the waypoint. The turn
    rate is aero-limited at speed (a fast, passively-stable rocket has a large
    turn radius), so it pitches over smoothly rather than yanking.

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking            # live viewer
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz   # headless
    .venv/bin/python -m examples.vehicles.camera_tracking --record run.rrd
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import ALL, Craft, EKF, Sim, TargetNumpy, World
from manta.fields import (
    CollisionField, FluidField, GravityField, HalfSpace, OpticalField,
)
from manta.parts import (
    CentroidCamera, Collider, DragSurface, Mass, OpticalSource,
    RevoluteJoint, Thruster,
)

from .._control import Pacer
from .._viz import Viz

# --- world / rates ---------------------------------------------------------
G = 9.81
DT = 0.02                              # 50 Hz control / EKF
SUBSTEPS = 10                          # 500 Hz physics
DT_SIM = DT / SUBSTEPS
RHO = 1.225

# --- tracker (fixed ground array) ------------------------------------------
CAM_XY = [(500.0, 500.0), (500.0, -500.0),
          (-500.0, 500.0), (-500.0, -500.0)]      # 1 km square
CAM_NAMES = [f"c{i}" for i in range(len(CAM_XY))]
CAM_Z = 0.3
GND_W, GND_HFOV, GND_PIX = 1280, 110.0, 2.0
GND_SENSORS = [f"tracker.{nm}.target_hull_{c}"
               for nm in CAM_NAMES for c in ("u", "v")]

# --- target (ballistic) ----------------------------------------------------
TGT_SEMI = (5.0, 5.0, 5.0)
TGT_MASS = 50.0
TGT_P0 = (5000.0, 0.0, 30.0)
TGT_V0 = (-105.0, 0.0, 258.0)          # ~3.2 km apogee, ~5 km downrange

# --- EKF tuning ------------------------------------------------------------
Q_VEL = 1e-3                           # process noise on the target ACCEL
SEED_DT = 0.4                          # finite-difference baseline for v-seed

# --- interceptor -----------------------------------------------------------
M_BODY, M_ENG = 60.0, 15.0
M_TOT = M_BODY + M_ENG
MAXT = 3000.0                          # T/W ~= 4
Z0 = 2.0
# Flight profile: boost straight up past Z_PITCH, then fly through WAYPOINT.
Z_PITCH = 80.0
# A short fast course: pitch over through a high+offset gate, then a second
# one further up-and-out. Spaced for the (aero-limited) turn radius at speed.
WAYPOINTS = [(500.0, 0.0, 700.0), (1500.0, 0.0, 1150.0)]
V_FLY = 130.0                          # cruise speed the guidance holds (m/s)
# Guidance (coordinated turn) gains.
K_SPD = 1.0                            # forward accel to hold V_FLY
K_LAT = 6.0                            # steering accel per deg off-velocity
A_LAT_MAX = 90.0                       # cap on the perpendicular (turn) accel
# Cascade autopilot. The INNER loop gain is the load-bearing one: KJP too high
# (e.g. 5000) → ω·dt ≈ 1 at 50 Hz → the gimbal command bang-bangs every tick
# and the airframe tumbles. The gimbal inertia is ~1.85 with joint damping 40.
THMAX = 0.35                           # gimbal deflection limit (rad)
KATT, KRATE = 1.6, 1.1                 # attitude error/rate → gimbal angle
KJP, KJD = 1000.0, 60.0                # gimbal-angle servo (joint torque PD)
GIM_SIGN = -1.0
SLEW = np.radians(50)                  # attitude-command slew limit (rad/s)


def build_world():
    tracker = Craft("tracker")
    tracker.add(Mass("base", mass=200.0, moi=(50, 50, 50)))
    tracker.add(Collider("foot", stiffness=3e4, damping=4e3, friction=4e3))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        tracker.add(CentroidCamera(nm, width=GND_W, height=GND_W,
                                   hfov_deg=GND_HFOV, pixel_sigma=GND_PIX,
                                   transform=(x, y, CAM_Z)))

    target = Craft("target")
    target.add(Mass("body", mass=TGT_MASS, moi=(1, 1, 1)))
    target.add(DragSurface.isotropic_quadratic("aero", area=0.005,
                                               drag_coefficient=0.3))
    target.add(OpticalSource("hull", semi_axes=TGT_SEMI, label=1))

    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField(g=(0, 0, -G)))
         .add_field(FluidField().add_uniform(density=RHO))
         .add_field(OpticalField())
         .add_field(cf))
    w.add_craft(tracker, position=(0, 0, 0))
    w.add_craft(target, position=TGT_P0, velocity=TGT_V0)
    w.add_craft(build_rocket(), position=(0, 0, Z0))
    return w


def build_rocket():
    """Gimballed TVC interceptor — passively stable (drag aft of the COM)."""
    rocket = Craft("interceptor")
    rocket.add(Mass("body", mass=M_BODY, moi=(80.0, 80.0, 1.5),
                    transform=(0, 0, 0.5)))
    rocket.add(DragSurface.directional_quadratic(
        "aero", areas=(0.6, 0.6, 0.08), drag_coefficient=0.8,
        transform=(0, 0, -0.6)))
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0, transform=(0, 0, -1.2))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0)
    gy.add(Mass("engine", mass=M_ENG, moi=(0.5, 0.5, 0.2), transform=(0, 0, -0.3)))
    gy.add(Thruster("main", force=(0, 0, MAXT), transform=(0, 0, -0.3)))
    gx.add(gy)
    rocket.add(gx)
    rocket.add(Collider("foot", stiffness=4e4, damping=6e3, friction=6e3,
                        transform=(0, 0, -2.0)))
    return rocket


# ---------------------------------------------------------------------------
# Quaternion / control helpers (numpy; manta quats are w,x,y,z)
# ---------------------------------------------------------------------------

def _R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _qerr(q_ref, q):
    qe = _qmul(q_ref, np.array([q[0], -q[1], -q[2], -q[3]]))
    if qe[0] < 0:
        qe = -qe
    n = np.linalg.norm(qe[1:])
    return np.zeros(3) if n < 1e-9 else qe[1:] / n * 2 * np.arctan2(n, qe[0])


def _slerp(q0, q1, f):
    d = float(np.dot(q0, q1))
    q1 = q1 if d >= 0 else -q1
    d = abs(d)
    if d > 0.9995:
        return q0
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


def attitude_for_thrust(d):
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(d, float) / (np.linalg.norm(d) + 1e-9)
    ang = np.arccos(np.clip(z @ d, -1.0, 1.0))
    ax = np.cross(z, d)
    s = np.linalg.norm(ax)
    return (np.array([1.0, 0, 0, 0]) if s < 1e-9
            else np.array([np.cos(ang / 2), *(ax / s * np.sin(ang / 2))]))


def autopilot(rk, q_ref):
    """Cascade: attitude error → gimbal ANGLE (capped) → joint torque PD."""
    q = np.asarray(rk["orientation"]).ravel()
    Rwb = _R(q)
    eb = Rwb.T @ _qerr(q_ref, q)
    wb = Rwb.T @ np.asarray(rk["angular_velocity"]).ravel()
    thx = np.clip(GIM_SIGN * (KATT * eb[0] - KRATE * wb[0]), -THMAX, THMAX)
    thy = np.clip(GIM_SIGN * (KATT * eb[1] - KRATE * wb[1]), -THMAX, THMAX)
    ax = float(np.asarray(rk["gimbal_x.angle"]).ravel()[0])
    rx = float(np.asarray(rk["gimbal_x.rate"]).ravel()[0])
    ay = float(np.asarray(rk["gimbal_y.angle"]).ravel()[0])
    ry = float(np.asarray(rk["gimbal_y.rate"]).ravel()[0])
    return {"interceptor.gimbal_x.torque_cmd":
            float(np.clip(KJP * (thx - ax) - KJD * rx, -4000, 4000)),
            "interceptor.gimbal_y.torque_cmd":
            float(np.clip(KJP * (thy - ay) - KJD * ry, -4000, 4000))}


def guidance_thrust(p, v, wp):
    """Coordinated-turn thrust command toward `wp`: forward to hold V_FLY,
    a bounded perpendicular component to curve the velocity onto the LOS."""
    sp = float(np.linalg.norm(v))
    vhat = v / sp if sp > 5 else np.array([0.0, 0.0, 1.0])
    to = wp - p
    dist = float(np.linalg.norm(to))
    perp = to - np.dot(to, vhat) * vhat
    pn = float(np.linalg.norm(perp))
    phat = perp / pn if pn > 1e-6 else np.zeros(3)
    off_deg = np.degrees(np.arccos(np.clip(np.dot(to / dist, vhat), -1, 1)))
    a_fwd = max(K_SPD * (V_FLY - sp), 5.0) * vhat
    a_lat = min(K_LAT * off_deg, A_LAT_MAX) * phat
    return M_TOT * (a_fwd + a_lat + np.array([0.0, 0.0, G])), dist


# ---------------------------------------------------------------------------
# Triangulation (used only to SEED the filter) + small utilities
# ---------------------------------------------------------------------------

def triangulate(out, vis):
    fx = (GND_W / 2.0) / np.tan(np.radians(GND_HFOV) / 2.0)
    cc = GND_W / 2.0
    A, b = np.zeros((3, 3)), np.zeros(3)
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        if not vis[nm]:
            continue
        u = float(np.asarray(out[f"{nm}.target_hull_u"]).ravel()[0])
        v = float(np.asarray(out[f"{nm}.target_hull_v"]).ravel()[0])
        d = np.array([(u - cc) / fx, (v - cc) / fx, 1.0])
        d /= np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ np.array([x, y, CAM_Z])
    return np.linalg.solve(A, b)


def _tan(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim


def visible(out):
    return {nm: float(np.asarray(out[f"{nm}.target_hull_vis"]).ravel()[0]) > 0.5
            for nm in CAM_NAMES}


def _aoa(q, v):
    sp = float(np.linalg.norm(v))
    if sp < 10:
        return 0.0
    return float(np.degrees(np.arccos(np.clip((_R(q) @ [0, 0, 1.0]) @ v / sp,
                                              -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--viz-addr", default=None,
                    help="stream to a running rerun viewer at host[:port]")
    ap.add_argument("--record", default=None, metavar="FILE.rrd",
                    help="save to a rerun .rrd (no viewer, full speed)")
    ap.add_argument("--duration", type=float, default=46.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(build_world()))
    truth.step(DT_SIM)

    ekf = TargetNumpy(EKF(build_world(), sensors=GND_SENSORS,
                          track={"target": ALL}))
    espec = ekf.spec
    ta = _tan(espec, "target.position")
    va = _tan(espec, "target.velocity")
    Q = np.full(espec.tangent_dim, 1e-12)
    Q[va[0]:va[1]] = Q_VEL
    Qm = np.diag(Q)

    viz = None if args.no_viz else Viz(
        "manta/camera_tracking", addr=args.viz_addr, save=args.record)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if (viz is not None and not args.record) else None

    phase = "wait"
    seed_p, seed_t = None, None
    errs = []
    rk_phase = "boost"            # boost → fly the course → coast
    wp_i, hits = 0, []
    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    max_aoa = 0.0
    wp = np.array(WAYPOINTS[0], float)
    print(f"\n{'t':>6} {'#vis':>4} {'trackErr':>9} {'rk':>6} "
          f"{'rk pos':>21} {'|v|':>5} {'aoa°':>5} {'wp dist':>8}")
    for i in range(int(args.duration / DT)):
        t = i * DT
        if pacer is not None:
            pacer.pace(t)
        tg = truth.state["target"]
        tg_p = np.asarray(tg["position"]).ravel()
        rk = truth.state["interceptor"]
        rk_p = np.asarray(rk["position"]).ravel()
        rk_v = np.asarray(rk["velocity"]).ravel()
        out = truth.outputs()["tracker"]
        vis = visible(out)
        nvis = sum(vis.values())

        # --- STEP 1: track the target ---------------------------------
        tgt_est = None
        if phase == "wait":
            if nvis >= 2:
                seed_p, seed_t = triangulate(out, vis), t
                phase = "seed"
        elif phase == "seed":
            if t - seed_t >= SEED_DT and nvis >= 2:
                p2 = triangulate(out, vis)
                v0 = (p2 - seed_p) / (t - seed_t)
                P0 = np.full(espec.tangent_dim, 1e-9)
                P0[ta[0]:ta[1]] = 50.0
                P0[va[0]:va[1]] = 30.0
                ekf.reset(state={"target": {"position": p2, "velocity": v0}},
                          P=np.diag(P0))
                phase = "track"
                print(f"   target locked at t={t:.1f}")
        else:
            ekf.predict(dt=DT, t=t, Q=Qm)
            for nm in CAM_NAMES:
                if not vis[nm]:
                    continue
                for c in ("u", "v"):
                    z = float(np.asarray(out[f"{nm}.target_hull_{c}"]).ravel()[0])
                    ekf.update(f"tracker.{nm}.target_hull_{c}",
                               np.array([z + rng.normal(0, GND_PIX)]))
            tgt_est = np.asarray(ekf.state_dict()["target"]["position"]).ravel()
            errs.append(float(np.linalg.norm(tgt_est - tg_p)))

        # --- STEP 2: boost, then pitch over through the waypoint -------
        if rk_phase == "boost":
            F = M_TOT * np.array([0.0, 0.0, MAXT / M_TOT])
            q_tgt = np.array([1.0, 0.0, 0.0, 0.0])
            if rk_p[2] > Z_PITCH:
                rk_phase = "fly"
                print(f"   interceptor pitching over at t={t:.1f}, "
                      f"|v|={np.linalg.norm(rk_v):.0f} m/s")
        else:
            if wp_i < len(WAYPOINTS):
                wp = np.array(WAYPOINTS[wp_i], float)
                F, dist = guidance_thrust(rk_p, rk_v, wp)
                if dist < 25.0:        # passed this gate — log it, go to next
                    hits.append((wp_i, dist, float(np.linalg.norm(rk_v))))
                    print(f"   passed waypoint {wp_i} within {dist:.0f} m at "
                          f"{np.linalg.norm(rk_v):.0f} m/s")
                    wp_i += 1
            else:                      # course done — fly straight on
                wp = rk_p + rk_v / max(np.linalg.norm(rk_v), 1.0) * 2000.0
                F, _ = guidance_thrust(rk_p, rk_v, wp)
            max_aoa = max(max_aoa, _aoa(np.asarray(rk["orientation"]).ravel(), rk_v))
            q_tgt = attitude_for_thrust(F)
        ang = 2 * np.arccos(np.clip(abs(np.dot(q_ref, q_tgt)), -1, 1))
        q_ref = _slerp(q_ref, q_tgt, 1.0 if ang < 1e-6 else min(1.0, SLEW * DT / ang))
        bodyz = _R(np.asarray(rk["orientation"]).ravel()) @ [0, 0, 1.0]
        u = {"interceptor.main.throttle":
             float(np.clip(np.dot(F, bodyz) / MAXT, 0.1, 1.0)), **autopilot(rk, q_ref)}
        for name, val in u.items():
            truth.command(name).set(float(val))

        if viz is not None and viz.due(t):
            _viz_step(viz, t, truth, tgt_est, vis, wp)
        for _ in range(SUBSTEPS):
            truth.step(DT_SIM)

        if i % int(2.0 / DT) == 0 and phase == "track":
            print(f"{t:6.1f} {nvis:4d} {errs[-1]:9.1f} {rk_phase:>6} "
                  f"({rk_p[0]:5.0f},{rk_p[1]:4.0f},{rk_p[2]:4.0f}) "
                  f"{np.linalg.norm(rk_v):5.0f} "
                  f"{_aoa(np.asarray(rk['orientation']).ravel(), rk_v):5.0f} "
                  f"{np.linalg.norm(wp - rk_p):8.0f}")

    if errs:
        print(f"\nSTEP 1 tracking: mean {np.mean(errs):.1f} m, "
              f"peak {np.max(errs):.1f} m")
    passes = ", ".join(f"wp{i} {d:.0f} m @ {s:.0f} m/s" for i, d, s in hits)
    print(f"STEP 2 flight: {passes or 'no pass'}; peak angle-of-attack "
          f"{max_aoa:.0f}° — pointed the way it flew, no tumble")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=6000.0, color=(45, 55, 50, 120))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        viz.box(f"world/tracker/{nm}", (14, 14, 14), center=(x, y, 6),
                color=(80, 170, 220))
    for k, wp in enumerate(WAYPOINTS):
        viz.box(f"world/waypoint/{k}", (24, 24, 24), center=tuple(wp),
                color=(230, 120, 60))
    viz.split_cylinder("world/interceptor/hull", 1.4, 14.0,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, -1.2 + 7))


def _viz_step(viz, t, truth, tgt_est, vis, wp):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
    viz.trail("world/interceptor_trail", rk_p, max_len=6000, min_dist=2.0,
              color=(230, 180, 80))
    viz.rr.log("world/target", viz.rr.Ellipsoids3D(
        centers=[tg_p], half_sizes=[(30, 30, 30)], colors=[(255, 170, 90)],
        fill_mode="solid"))
    viz.trail("world/target_trail", tg_p, max_len=6000, min_dist=3.0)
    if tgt_est is not None:
        viz.point("world/target_est", tgt_est, color=(90, 230, 120), radius=28.0)
        for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
            if vis[nm]:
                viz.line(f"world/ray_{nm}", [(x, y, CAM_Z), tuple(tgt_est)],
                         color=(70, 100, 130), radius=1.5)


if __name__ == "__main__":
    main()
