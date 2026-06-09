"""Camera tracking — a ground array cues a fast interceptor that homes with
its own nose camera.

The headline demo for `CentroidCamera` + size-independent triangulation, plus
a hand-off to onboard `BBoxCamera` homing:

  * **tracker** — a fixed ground array: 5 `CentroidCamera`s in a 1 km cross,
    looking up, anchored to the ground (a static craft free-falls under
    gravity otherwise). Each reports only the target's image CENTROID — a
    pure bearing, independent of target size. The EKF folds them
    asynchronously and the Kalman update IS the noise-weighted intersection
    of the rays: triangulation with no triangulation code, range from the
    kilometre baseline. Good to ~hundreds of metres at km range — enough to
    LAUNCH the interceptor and aim it.

  * **target** — a `Mass` + `DragSurface` + `OpticalSource`, launched
    ballistically (~3 km apogee, ~5 km downrange) from far and low, climbing
    into the array's field of view near apogee.

  * **interceptor** — a fast (500+ m/s) gimballed rocket made slightly
    PASSIVELY STABLE: its drag surface sits aft of the COM, so it
    weathervanes into the airflow. That tames the launch-tumble an
    unaugmented thrust-vectored rocket suffers, while staying maneuverable. A
    cascade autopilot (attitude error → gimbal ANGLE → joint torque-PD)
    points the thrust. Mid-course it pursues the ground-array intercept
    point; once its wide-FOV nose `BBoxCamera` sees the target it switches to
    PROPORTIONAL NAVIGATION on the camera's bearing (the box centre is an
    accurate line of sight at any range; the box size→range is not, so the
    range comes from the EKF) — PN nulls the line-of-sight rate onto a
    collision course.

Physics integrates with semi-implicit Euler; at a 500+ m/s pass the
quadratic drag is stiff, so the substep is 1 kHz. The terminal closure
against a fast crossing target off a coarse km-range cue is genuinely hard —
the demo gets to a several-hundred-metre pass, not a hit-to-kill.

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, EKF, Sim, TargetNumpy, World
from manta.fields import (
    CollisionField, FluidField, GravityField, HalfSpace, OpticalField,
)
from manta.parts import (
    BBoxCamera, CentroidCamera, Collider, DragSurface, Mass, OpticalSource,
    RevoluteJoint, Thruster,
)

from .._control import Pacer
from .._viz import Viz

# --- world / rates ---------------------------------------------------------
G = 9.81
DT_SIM, SUBSTEPS = 0.001, 20           # 1 kHz physics (stiff TVC + drag)
DT = DT_SIM * SUBSTEPS                  # 50 Hz control / EKF
RHO = 1.225

# --- tracker (fixed ground array) ------------------------------------------
CAM_XY = [(0.0, 0.0), (0.0, 1000.0), (0.0, -1000.0),
          (1000.0, 0.0), (-1000.0, 0.0)]
CAM_NAMES = [f"c{i}" for i in range(len(CAM_XY))]
GND_W, GND_HFOV, GND_PIX = 1280, 110.0, 2.0
CAM_RATE_TICKS = 3                      # each ground camera captures ~17 Hz

# --- target (ballistic) ----------------------------------------------------
TGT_SEMI = (5.0, 5.0, 5.0)
TGT_MASS = 50.0
TGT_P0 = (5000.0, 0.0, 30.0)
TGT_V0 = (-105.0, 0.0, 258.0)          # ~3.2 km apogee, ~5 km downrange

# --- interceptor -----------------------------------------------------------
M_BODY, M_ENG = 60.0, 15.0
M_TOT = M_BODY + M_ENG
MAXT = 16000.0                         # thrust (T/W ≈ 22)
GIM_Z, ENG_Z = -1.2, -0.3
DRAG_Z = -0.6                          # drag AFT of COM (~0.18) → passive stability
NOSE_Z = 2.0                           # nose camera, looking forward (+z)
NOSE_W, NOSE_HFOV, NOSE_PIX = 640, 100.0, 1.5
Z0 = 2.0
HIT_RADIUS = 6.0
MIN_INTERCEPT_Z = 1000.0

# --- guidance / autopilot --------------------------------------------------
V_CLOSE = 700.0                        # nominal closing speed for the lead
N_PN = 4.0                             # proportional-navigation constant
PN_RANGE = 2000.0                      # switch to PN inside this range (m)
KFWD = 200.0                           # forward accel to keep closing (m/s2)
KP_G, KD_G = 1.2, 1.0                  # mid-course pursuit accel-command gains
THMAX = 0.22                           # gimbal deflection limit (rad)
KATT, KRATE = 1.2, 1.0                 # attitude error/rate → gimbal angle
KJP, KJD = 2500.0, 600.0               # gimbal-angle servo (joint torque PD)
GIM_SIGN = -1.0                        # gimbal → body-torque coupling sign
LAUNCH_Z = 1500.0                      # launch once the cued track is this high
ENGAGE_SIGMA = 250.0                   # ground-track σ (m) good enough to commit
lock_t = [0.0, 0.0]
aim = [None]

# --- EKF -------------------------------------------------------------------
GND_SENSORS = [f"tracker.{nm}.target_hull_{c}"
               for nm in CAM_NAMES for c in ("u", "v")]


def build_world():
    # --- tracker: anchored ground array --------------------------------
    tracker = Craft("tracker")
    tracker.add(Mass("base", mass=200.0, moi=(50, 50, 50)))
    tracker.add(Collider("foot", stiffness=3e4, damping=4e3, friction=4e3))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        tracker.add(CentroidCamera(nm, width=GND_W, height=GND_W,
                                   hfov_deg=GND_HFOV, pixel_sigma=GND_PIX,
                                   transform=(x, y, 0.3)))

    # --- target: ballistic projectile ----------------------------------
    target = Craft("target")
    target.add(Mass("body", mass=TGT_MASS, moi=(1, 1, 1)))
    target.add(DragSurface.isotropic_quadratic("aero", area=0.005,
                                               drag_coefficient=0.3))
    target.add(OpticalSource("hull", semi_axes=TGT_SEMI, label=1))

    # --- interceptor: fast, passively-stable TVC rocket ----------------
    rocket = Craft("interceptor")
    rocket.add(Mass("body", mass=M_BODY, moi=(80.0, 80.0, 1.5),
                    transform=(0, 0, 0.5)))
    # Directional (cylinder) drag AFT of the COM → it weathervanes stable.
    rocket.add(DragSurface.directional_quadratic(
        "aero", areas=(0.6, 0.6, 0.08), drag_coefficient=0.8,
        transform=(0, 0, DRAG_Z)))
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0, transform=(0, 0, GIM_Z))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0)
    gy.add(Mass("engine", mass=M_ENG, moi=(0.5, 0.5, 0.2),
                transform=(0, 0, ENG_Z)))
    gy.add(Thruster("main", force=(0, 0, MAXT), transform=(0, 0, ENG_Z)))
    gx.add(gy)
    rocket.add(gx)
    # Nose camera, looking forward (+z = nose), for terminal homing.
    rocket.add(BBoxCamera("nose", width=NOSE_W, height=NOSE_W,
                          hfov_deg=NOSE_HFOV, bbox_sigma=NOSE_PIX,
                          transform=(0, 0, NOSE_Z)))
    rocket.add(Collider("foot", stiffness=4e4, damping=6e3, friction=6e3,
                        transform=(0, 0, -2.0)))

    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField(g=(0, 0, -G)))
         .add_field(FluidField().add_uniform(density=RHO))
         .add_field(OpticalField())
         .add_field(cf))
    w.add_craft(tracker, position=(0, 0, 0))
    w.add_craft(target, position=TGT_P0, velocity=TGT_V0)
    w.add_craft(rocket, position=(0, 0, Z0))
    return w


# ---------------------------------------------------------------------------
# Small quaternion / guidance helpers (numpy; manta quats are w,x,y,z)
# ---------------------------------------------------------------------------

def _R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _qerr(q_ref, q):
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    aw, ax, ay, az = q_ref
    bw, bx, by, bz = qc
    qe = np.array([aw * bw - ax * bx - ay * by - az * bz,
                   aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw])
    if qe[0] < 0:
        qe = -qe
    n = np.linalg.norm(qe[1:])
    return np.zeros(3) if n < 1e-9 else qe[1:] / n * 2 * np.arctan2(n, qe[0])



def attitude_for_thrust(d, cap_deg=70.0):
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(d, float) / (np.linalg.norm(d) + 1e-9)
    ang = min(float(np.arccos(np.clip(z @ d, -1.0, 1.0))), np.radians(cap_deg))
    axis = np.cross(z, d)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(ang / 2), *(axis / s * np.sin(ang / 2))])


def autopilot(rk, q_ref):
    """Cascade TVC autopilot: attitude error → desired gimbal ANGLE (capped)
    → joint torque PD. With the passive aero stability this stays stable from
    a standstill on the pad through > 700 m/s."""
    q = np.asarray(rk["orientation"]).ravel()
    wv = np.asarray(rk["angular_velocity"]).ravel()
    Rwb = _R(q)
    eb = Rwb.T @ _qerr(q_ref, q)
    wb = Rwb.T @ wv
    thxd = np.clip(GIM_SIGN * (KATT * eb[0] - KRATE * wb[0]), -THMAX, THMAX)
    thyd = np.clip(GIM_SIGN * (KATT * eb[1] - KRATE * wb[1]), -THMAX, THMAX)
    ax = float(np.asarray(rk["gimbal_x.angle"]).ravel()[0])
    rx = float(np.asarray(rk["gimbal_x.rate"]).ravel()[0])
    ay = float(np.asarray(rk["gimbal_y.angle"]).ravel()[0])
    ry = float(np.asarray(rk["gimbal_y.rate"]).ravel()[0])
    return {"interceptor.gimbal_x.torque_cmd":
            float(np.clip(KJP * (thxd - ax) - KJD * rx, -4000, 4000)),
            "interceptor.gimbal_y.torque_cmd":
            float(np.clip(KJP * (thyd - ay) - KJD * ry, -4000, 4000))}


def ballistic(p, v, t):
    g = np.array([0.0, 0.0, -G])
    return p + v * t + 0.5 * g * t * t


def intercept_point(p_rk, p_tg, v_tg):
    pip, t_go = p_tg, 0.0
    for _ in range(3):
        t_go = float(np.linalg.norm(pip - p_rk)) / V_CLOSE
        pip = ballistic(p_tg, v_tg, t_go)
    return pip, t_go


def deproject(out, vis):
    """Algebraic ray-intersection of the visible ground cameras (filter seed)."""
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
        b += M @ np.array([x, y, 0.3])
    return np.linalg.solve(A, b)


def nose_deproject(out, rk_p, rk_q, range_hint):
    """Target world position from the nose camera's box CENTRE (an accurate
    bearing at any range — the box may be sub-pixel but its centre still
    projects the target's centre) plus a RANGE HINT from the ground EKF. The
    box size→range is unreliable far out, so we take only the bearing here;
    PN cares about the (accurate) line-of-sight rate, not absolute range."""
    fx = (NOSE_W / 2.0) / np.tan(np.radians(NOSE_HFOV) / 2.0)
    cc = NOSE_W / 2.0
    box = [float(np.asarray(out[f"nose.target_hull_{c}"]).ravel()[0])
           for c in ("xmin", "ymin", "xmax", "ymax")]
    u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    d_cam = np.array([(u - cc) / fx, (v - cc) / fx, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    Rwb = _R(rk_q)
    cam_p = rk_p + Rwb @ np.array([0.0, 0.0, NOSE_Z])
    return cam_p + range_hint * (Rwb @ d_cam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--viz-addr", default=None)
    ap.add_argument("--duration", type=float, default=48.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(build_world()))
    truth.step(DT_SIM)
    ekf = TargetNumpy(EKF(build_world(), sensors=GND_SENSORS))
    spec = ekf.spec
    ta = _tan(spec, "target.position")
    va = _tan(spec, "target.velocity")
    # Pin everything; the target is the only thing we estimate. The
    # interceptor's & tracker's poses are KNOWN (onboard nav / surveyed) and
    # written into the filter each tick so the camera measurement models use
    # the right poses.
    Q = np.full(spec.tangent_dim, 1e-12)
    Q[ta[0]:ta[1]] = 0.5
    Q[va[0]:va[1]] = 0.5
    Qm = np.diag(Q)

    viz = None if args.no_viz else Viz("manta/camera_tracking", addr=args.viz_addr)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if viz is not None else None

    locked = launched = False
    min_miss, hit_t = 1e9, None
    print(f"\n{'t':>5} {'gnd#':>4} {'nose':>4} {'est_err':>8} "
          f"{'rk z':>7} {'tgt z':>7} {'miss':>9}")
    for i in range(int(args.duration / DT)):
        t = i * DT
        if pacer is not None:
            pacer.pace(t)
        rk = truth.state["interceptor"]
        tg = truth.state["target"]
        rk_p = np.asarray(rk["position"]).ravel()
        rk_v = np.asarray(rk["velocity"]).ravel()
        tg_p = np.asarray(tg["position"]).ravel()
        tg_v = np.asarray(tg["velocity"]).ravel()

        gnd = truth.outputs()["tracker"]
        nose = truth.outputs()["interceptor"]
        gvis = {nm: float(np.asarray(gnd[f"{nm}.target_hull_vis"]).ravel()[0]) > 0.5
                for nm in CAM_NAMES}
        nvis = float(np.asarray(nose["nose.target_hull_vis"]).ravel()[0]) > 0.5

        # --- lock the filter on first multi-camera sighting --------------
        if not locked:
            if sum(gvis.values()) >= 3:
                st = ekf.state_dict()
                st["target"] = {"position": deproject(gnd, gvis),
                                "velocity": (-80.0, 0.0, 40.0)}
                P = ekf.P
                for a, b, val in ((ta[0], ta[1], 4e3), (va[0], va[1], 1e3)):
                    for k in range(a, b):
                        P[k, k] = val
                ekf.reset(state=st, P=P)
                locked = True
                lock_t[0] = t
                print(f"   t={t:.1f}  target acquired by the ground array")
            _step(truth, None)
            continue

        # --- EKF: triangulate the target from the GROUND array -----------
        # The tracker rests on its Collider, so the predict keeps it (and the
        # padded rocket) at rest, matching truth — no pinning needed. The
        # interceptor's own block is decoupled (no measurement touches both),
        # so once it launches its drift in the filter is harmless.
        ekf.predict(dt=DT, t=t, Q=Qm)
        for nm in [n for j, n in enumerate(CAM_NAMES)
                   if gvis[n] and (i + j) % CAM_RATE_TICKS == 0]:
            for c in ("u", "v"):
                z = float(np.asarray(gnd[f"{nm}.target_hull_{c}"]).ravel()[0])
                ekf.update(f"tracker.{nm}.target_hull_{c}",
                           np.array([z + rng.normal(0, GND_PIX)]))

        est = ekf.state_dict()["target"]
        tgt_est = np.asarray(est["position"]).ravel()
        tgt_v_est = np.asarray(est["velocity"]).ravel()
        pos_sig = float(np.sqrt(np.trace(ekf.P[ta[0]:ta[1], ta[0]:ta[1]])))

        # --- terminal: the nose camera's bearing + the EKF range ---------
        nose_fix = None
        if launched and nvis:
            nose_fix = nose_deproject(
                nose, rk_p, np.asarray(rk["orientation"]).ravel(),
                float(np.linalg.norm(tgt_est - rk_p)))
        aim_p = nose_fix if nose_fix is not None else tgt_est
        est_err = float(np.linalg.norm(aim_p - tg_p))

        # --- launch + guidance ------------------------------------------
        pip, t_go = intercept_point(rk_p, tgt_est, tgt_v_est)
        # Commit once the track has converged (a few seconds past lock, so the
        # velocity is real not the seed) AND the target is near apogee — then
        # the intercept is high+slow and the rocket climbs nearly vertically
        # (launching early, with the target still downrange, pitches it over
        # hard at low altitude and it can't climb).
        if (not launched and t > lock_t[0] + 6.0 and pip[2] > LAUNCH_Z
                and pos_sig < ENGAGE_SIGMA):
            launched = True
            lock_t[1] = t
            print(f"   t={t:.1f}  LAUNCH — cued intercept ~{np.round(pip, 0)}")
        u = {"interceptor.main.throttle": 0.0,
             "interceptor.gimbal_x.torque_cmd": 0.0,
             "interceptor.gimbal_y.torque_cmd": 0.0}
        if launched:
            # Aim point: the (heavily low-passed) ground track mid-course; the
            # nose camera's own fix once it acquires (size cue → tight range).
            raw = nose_fix if nose_fix is not None else pip
            b = 0.4 if nose_fix is not None else 0.05
            aim[0] = raw if aim[0] is None else (1 - b) * aim[0] + b * raw
            R = aim[0] - rk_p
            rmag = float(np.linalg.norm(R))
            speed = float(np.linalg.norm(rk_v))
            if rmag < PN_RANGE and speed > 120.0:
                # PROPORTIONAL NAVIGATION terminal homing on the nose camera's
                # own (tight) fix: command lateral acceleration ∝ LOS-rate ×
                # closing-speed to null the line-of-sight rotation (a collision
                # course), plus forward thrust to keep closing. PN leads
                # implicitly — no PIP. The coarse ground track was only ever
                # good enough to fly the rocket into the seeker's basket.
                Rhat = R / rmag
                Vrel = tgt_v_est - rk_v
                omega = np.cross(R, Vrel) / (rmag * rmag)      # LOS rate (rad/s)
                Vc = max(-float(np.dot(R, Vrel)) / rmag, 1.0)  # closing speed
                a_pn = N_PN * Vc * np.cross(omega, Rhat)      # ⟂ to the LOS
                a_des = a_pn + KFWD * (rk_v / speed) + np.array([0.0, 0.0, G])
            else:
                # Mid-course pursuit to the predicted intercept point.
                a_des = KP_G * (aim[0] - rk_p) - KD_G * rk_v + np.array([0, 0, G])
            F = M_TOT * a_des
            u["interceptor.main.throttle"] = float(
                np.clip(np.linalg.norm(F) / MAXT, 0.2, 1.0))
            u.update(autopilot(rk, attitude_for_thrust(F)))
            body_up = _R(np.asarray(rk["orientation"]).ravel())[2, 2]
            if body_up < 0.0 or not np.isfinite(body_up):    # tumbled — abort burn
                u = {k: 0.0 for k in u}
        _step(truth, u)

        rk2 = np.asarray(truth.state["interceptor"]["position"]).ravel()
        tg2 = np.asarray(truth.state["target"]["position"]).ravel()
        if not np.all(np.isfinite(rk2)):     # terminal high-speed blow-up
            break
        miss = float(np.linalg.norm(rk2 - tg2))
        min_miss = min(min_miss, miss)
        if miss < HIT_RADIUS and hit_t is None and tg2[2] > MIN_INTERCEPT_Z:
            hit_t = t
        if hit_t is not None and miss > min_miss + 50.0:
            break

        if viz is not None:
            _viz_step(viz, t, truth, tgt_est, gvis, nvis)
        if i % int(2.0 / DT) == 0:
            print(f"{t:5.1f} {sum(gvis.values()):4d} {int(nvis):4d} "
                  f"{est_err:8.1f} {rk_p[2]:7.0f} {tg_p[2]:7.0f} {miss:9.1f}")

    if hit_t is not None:
        print(f"\n*** INTERCEPT at t={hit_t:.2f} s — closest {min_miss:.1f} m "
              f"(above {MIN_INTERCEPT_Z:.0f} m) ***")
    else:
        print(f"\nclosest approach: {min_miss:.1f} m — the ground array "
              f"triangulated the ballistic arc from bearings, the interceptor "
              f"launched on that cue and flew a passively-stable 500+ m/s "
              f"climb, and proportional navigation on the nose-camera bearing "
              f"homed it from kilometres. The terminal pass against a fast "
              f"descending crossing target is the unsolved part — a coarse "
              f"km-range cue + a thin-margin TVC airframe leave a several-"
              f"hundred-metre miss; closing it needs a dedicated seeker loop, "
              f"not more gain tuning.")


def _tan(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim



def _step(truth, u):
    if isinstance(u, dict):
        for name, val in u.items():
            truth.command(name).set(float(val))
    for _ in range(SUBSTEPS):
        truth.step(DT_SIM)


def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=6000.0, color=(45, 55, 50, 120))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        viz.box(f"world/tracker/{nm}", (12, 12, 12), center=(x, y, 6),
                color=(80, 170, 220))
    viz.split_cylinder("world/interceptor/hull", 1.4, 14.0,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, GIM_Z + 7))


def _viz_step(viz, t, truth, tgt_est, gvis, nvis):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
    viz.trail("world/interceptor_trail", rk_p, max_len=6000, min_dist=3.0)
    viz.rr.log("world/target", viz.rr.Ellipsoids3D(
        centers=[tg_p], half_sizes=[(30, 30, 30)], colors=[(255, 170, 90)],
        fill_mode="solid"))
    viz.point("world/target_est", tgt_est, color=(90, 230, 120), radius=25.0)
    viz.trail("world/target_trail", tg_p, max_len=6000, min_dist=3.0)
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        if gvis[nm]:
            viz.line(f"world/ray_{nm}", [(x, y, 0.3), tuple(tgt_est)],
                     color=(70, 100, 130), radius=1.5)


if __name__ == "__main__":
    main()
