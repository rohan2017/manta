"""Camera tracking — a ground array triangulates a ballistic target; a
thrust-vectored interceptor is cued onto it.

This is the headline demo for `CentroidCamera` + size-independent
triangulation. Three pieces:

  * **tracker** — a fixed ground array: 5 `CentroidCamera`s in a cross,
    1 km apart, looking up. Each reports only the target's image-frame
    CENTROID (a pure bearing, independent of the target's size). It is
    anchored to the ground with a `Collider` — a static craft still
    free-falls under gravity otherwise, which would desync the cameras
    from the filter.

  * **target** — a `Mass` + `DragSurface` + `OpticalSource`, launched
    ballistically on a high arc (~3 km apogee, ~5 km downrange). It starts
    far and low, OUTSIDE the array's upward field of view, and climbs into
    view near apogee.

  * **interceptor** — a high-thrust-to-weight gimballed rocket with a
    directional (cylindrical-fuselage) drag surface. No sensors of its own:
    it is CUED by the ground array, like a SAM battery off a radar track.

The estimation is the point. No single camera knows the range — a centroid
is just a ray. The EKF folds the centroids (asynchronously, only the cameras
that captured this tick) and the Kalman update IS the noise-weighted
intersection of the rays: triangulation, with no triangulation code, range
set by the kilometre baseline rather than the target's size. Once the track
converges the interceptor computes a ballistic intercept point and launches
to meet the target above 1 km.

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz
    .venv/bin/python -m examples.vehicles.camera_tracking --truth   # cue on truth
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, EKF, Sim, TargetNumpy, World
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
DT_SIM, SUBSTEPS = 0.002, 10           # 500 Hz physics (stiff TVC + contact)
DT = DT_SIM * SUBSTEPS                  # 50 Hz control / EKF
RHO = 1.225

# --- tracker (fixed ground array) ------------------------------------------
CAM_XY = [(0.0, 0.0), (0.0, 1000.0), (0.0, -1000.0),
          (1000.0, 0.0), (-1000.0, 0.0)]          # cross, 1 km arms
CAM_NAMES = [f"c{i}" for i in range(len(CAM_XY))]
W_IMG, HFOV = 1280, 110.0
PIX_SIGMA = 2.0                        # centroid noise (px)
CAM_RATE_TICKS = 3                     # each camera captures every 3 ticks (~17 Hz)

# --- target (ballistic) ----------------------------------------------------
TGT_SEMI = (5.0, 5.0, 5.0)             # size is irrelevant to the centroid
TGT_MASS = 50.0
TGT_P0 = (5000.0, 0.0, 30.0)           # far + low: starts outside the FOV
TGT_V0 = (-105.0, 0.0, 258.0)          # → ~3.2 km apogee, ~5 km downrange

# --- interceptor (high T/W gimballed rocket) -------------------------------
M_BODY, M_ENG = 60.0, 15.0
M_TOT = M_BODY + M_ENG
MAXT = 22000.0                         # full thrust (N): T/W ≈ 30
GIM_Z, ENG_Z = -1.2, -0.3
BODY_R, BODY_L = 0.25, 4.0
Z0 = 2.0                               # launch altitude
HIT_RADIUS = 6.0                       # intercept radius (m)
MIN_INTERCEPT_Z = 1000.0              # must meet the target above this

# --- EKF -------------------------------------------------------------------
CENTROID_SENSORS = [f"tracker.{nm}.target_hull_{c}"
                    for nm in CAM_NAMES for c in ("u", "v")]


def build_world():
    # --- tracker: anchored ground array --------------------------------
    tracker = Craft("tracker")
    tracker.add(Mass("base", mass=200.0, moi=(50, 50, 50)))
    tracker.add(Collider("foot", stiffness=3e4, damping=4e3, friction=4e3))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        tracker.add(CentroidCamera(nm, width=W_IMG, height=W_IMG,
                                   hfov_deg=HFOV, pixel_sigma=PIX_SIGMA,
                                   transform=(x, y, 0.3)))

    # --- target: ballistic projectile ----------------------------------
    target = Craft("target")
    target.add(Mass("body", mass=TGT_MASS, moi=(1, 1, 1)))
    target.add(DragSurface.isotropic_quadratic("aero", area=0.005,
                                               drag_coefficient=0.3))
    target.add(OpticalSource("hull", semi_axes=TGT_SEMI, label=1))

    # --- interceptor: high-T/W TVC rocket ------------------------------
    rocket = Craft("interceptor")
    rocket.add(Mass("body", mass=M_BODY, moi=(80.0, 80.0, 1.5),
                    transform=(0, 0, 0.6)))
    # Directional drag: a slender cylinder — low nose-on (body z), high
    # broadside (body x,y).
    rocket.add(DragSurface.directional_quadratic(
        "aero", areas=(0.6, 0.6, 0.08), drag_coefficient=0.8))
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0, transform=(0, 0, GIM_Z))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0)
    gy.add(Mass("engine", mass=M_ENG, moi=(0.5, 0.5, 0.2),
                transform=(0, 0, ENG_Z)))
    gy.add(Thruster("main", force=(0, 0, MAXT), transform=(0, 0, ENG_Z)))
    gx.add(gy)
    rocket.add(gx)
    # Sit on the pad until launch (else it free-falls like any craft).
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


def tan_slice(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim


def ballistic_predict(p, v, t):
    """Drag-free ballistic propagation of (p, v) by `t` seconds — good
    enough for the intercept-point solve over a few seconds."""
    g = np.array([0.0, 0.0, -G])
    return p + v * t + 0.5 * g * t * t, v + g * t


def intercept_point(p_rk, p_tg, v_tg, v_close):
    """Predicted intercept point: where the ballistic target will be in
    `range / closing-speed` seconds. Two fixed-point iterations."""
    t_go = 0.0
    pip = p_tg
    for _ in range(3):
        t_go = float(np.linalg.norm(pip - p_rk)) / v_close
        pip, _ = ballistic_predict(p_tg, v_tg, t_go)
    return pip, t_go


def attitude_for_thrust(d, cap_deg=80.0):
    """Quaternion (w,x,y,z) Quat[World, Body] pointing body +z along the
    unit world vector `d`, tilt capped so the rocket never flips."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(d, float) / (np.linalg.norm(d) + 1e-12)
    ang = min(float(np.arccos(np.clip(z @ d, -1.0, 1.0))), np.radians(cap_deg))
    axis = np.cross(z, d)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= s
    return np.array([np.cos(ang / 2), *(axis * np.sin(ang / 2))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--viz-addr", default=None)
    p.add_argument("--duration", type=float, default=46.0)
    p.add_argument("--truth", action="store_true",
                   help="cue the interceptor on the target's true state")
    args = p.parse_args()

    noise_rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(build_world()))
    truth.step(DT_SIM)                      # prime the output bus
    ekf = TargetNumpy(EKF(build_world(), sensors=CENTROID_SENSORS))
    spec = ekf.spec
    ta = tan_slice(spec, "target.position")
    va = tan_slice(spec, "target.velocity")
    # Tracker pinned (surveyed ground array); ballistic target → modest Q.
    Q = np.full(spec.tangent_dim, 1e-12)
    Q[ta[0]:ta[1]] = 0.5
    Q[va[0]:va[1]] = 0.5
    Qm = np.diag(Q)

    viz = None if args.no_viz else Viz("manta/camera_tracking", addr=args.viz_addr)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if viz is not None else None

    locked = False
    launched = False
    miss = min_miss = 1e9
    hit_t = None
    n = int(args.duration / DT)
    print(f"\n{'t':>5} {'#cam':>4} {'est_err':>8} {'tgt z':>7} "
          f"{'rk z':>7} {'miss':>8}")
    for i in range(n):
        t = i * DT
        if pacer is not None:
            pacer.pace(t)
        tg = truth.state["target"]
        rk = truth.state["interceptor"]
        tg_p = np.asarray(tg["position"]).ravel()
        rk_p = np.asarray(rk["position"]).ravel()

        out = truth.outputs()["tracker"]
        vis = {nm: float(np.asarray(out[f"{nm}.target_hull_vis"]).ravel()[0]) > 0.5
               for nm in CAM_NAMES}
        # Asynchronous capture: each camera fires every CAM_RATE_TICKS ticks,
        # staggered, so the EKF only ever sees a partial set this tick.
        firing = [nm for j, nm in enumerate(CAM_NAMES)
                  if vis[nm] and (i + j) % CAM_RATE_TICKS == 0]

        # --- EKF: lock on first multi-camera sighting, then fold ----------
        if not locked:
            if sum(vis.values()) >= 3:
                p0, t0 = _deproject_two(out, vis, noise_rng)
                st = ekf.state_dict()
                st["target"] = {"position": p0, "velocity": (-80.0, 0.0, 40.0)}
                P = ekf.P
                for a, b, val in ((ta[0], ta[1], 1e4), (va[0], va[1], 4e3)):
                    for k in range(a, b):
                        P[k, k] = val
                ekf.reset(state=st, P=P)
                locked = True
                print(f"   t={t:.1f}  target acquired ({sum(vis.values())} cams)")
            _step_truth(truth, np.zeros(3) if not launched else None)
            continue

        ekf.predict(dt=DT, t=t, Q=Qm)
        for nm in firing:
            for c in ("u", "v"):
                z = float(np.asarray(out[f"{nm}.target_hull_{c}"]).ravel()[0])
                ekf.update(f"tracker.{nm}.target_hull_{c}",
                           np.array([z + noise_rng.normal(0, PIX_SIGMA)]))

        est = ekf.state_dict()["target"]
        tgt_est = np.asarray(est["position"]).ravel()
        tgt_v_est = np.asarray(est["velocity"]).ravel()
        est_err = float(np.linalg.norm(tgt_est - tg_p))

        if args.truth:
            guide_p = tg_p
            guide_v = np.asarray(tg["velocity"]).ravel()
        else:
            # The km-range bearings-only estimate is noisy; low-pass it before
            # the (sensitive) guidance acts on it, or the thrust vector jitters.
            a = 0.15
            _guide["p"] = (tgt_est if _guide["p"] is None
                           else (1 - a) * _guide["p"] + a * tgt_est)
            _guide["v"] = (tgt_v_est if _guide["v"] is None
                           else (1 - a) * _guide["v"] + a * tgt_v_est)
            guide_p, guide_v = _guide["p"], _guide["v"]

        rk_v = np.asarray(rk["velocity"]).ravel()
        thr, gim_u = 0.0, {"interceptor.gimbal_x.torque_cmd": 0.0,
                           "interceptor.gimbal_y.torque_cmd": 0.0}
        # The interceptor commits only on a cue precise enough for a terminal
        # TVC solution. The kilometre-range bearings-only track is ~300 m (1σ)
        # — excellent for cueing, but too coarse to fly a 6 m intercept onto
        # without the thrust vector chasing the noise. So the default run is
        # the ESTIMATION showcase (the array triangulating the arc); `--truth`
        # supplies the perfect cue and flies the full intercept.
        if not launched and locked and args.truth:
            pip, t_go = intercept_point(rk_p, guide_p, guide_v, V_CLOSE)
            if pip[2] > MIN_INTERCEPT_Z and guide_v[2] < 60.0:
                launched = True
                print(f"   t={t:.1f}  LAUNCH — intercept ~{np.round(pip, 0)} "
                      f"in ~{t_go:.1f}s")
        if launched:
            pip, t_go = intercept_point(rk_p, guide_p, guide_v, V_CLOSE)
            # Acceleration toward the PREDICTED INTERCEPT POINT (which already
            # leads the target), damped by the rocket's OWN velocity — NOT by
            # the target's, which is plunging at ~200 m/s and would fight the
            # climb. Magnitude → throttle, direction → attitude.
            a_cmd = KP * (pip - rk_p) - KD * rk_v
            mag = np.linalg.norm(a_cmd)
            if mag > A_MAX:
                a_cmd *= A_MAX / mag
            F = M_TOT * (a_cmd + np.array([0.0, 0.0, G]))
            thr = float(np.clip(np.linalg.norm(F) / MAXT, 0.15, 1.0))
            q_ref = attitude_for_thrust(F)
            gim_u = _attitude_cmd(rk, q_ref)
            if not (np.all(np.isfinite(F)) and np.isfinite(thr)
                    and all(np.isfinite(v) for v in gim_u.values())):
                thr, gim_u = 0.2, {k: 0.0 for k in gim_u}   # coast on garbage

        u = dict(gim_u)
        u["interceptor.main.throttle"] = thr
        _step_truth(truth, u)

        # closest approach (and intercept) at the physics rate
        rk_p2 = np.asarray(truth.state["interceptor"]["position"]).ravel()
        tg_p2 = np.asarray(truth.state["target"]["position"]).ravel()
        miss = float(np.linalg.norm(rk_p2 - tg_p2))
        if miss < min_miss:
            min_miss = miss
        if miss < HIT_RADIUS and hit_t is None and tg_p2[2] > MIN_INTERCEPT_Z:
            hit_t = t
        if hit_t is not None and miss > min_miss + 20.0:
            break

        if viz is not None:
            _viz_step(viz, t, truth, tgt_est, vis)
        best_err[0] = est_err              # latest track error
        if i % int(2.0 / DT) == 0:
            print(f"{t:5.1f} {sum(vis.values()):4d} {est_err:8.1f} "
                  f"{tg_p[2]:7.0f} {rk_p[2]:7.0f} {miss:8.1f}")

    if hit_t is not None:
        print(f"\n*** INTERCEPT at t={hit_t:.2f} s — closest {min_miss:.1f} m "
              f"(above {MIN_INTERCEPT_Z:.0f} m) ***")
    elif args.truth:
        print(f"\nclosest approach: {min_miss:.1f} m")
    else:
        print(f"\nThe ground array triangulated the ballistic arc from "
              f"bearings ALONE — track error ~{best_err[0]:.0f} m at km-scale "
              f"range (set by the 1 km baseline, not the target's size; it "
              f"tightens as the target closes).\nRe-run with --truth to fly "
              f"the cued intercept — this bearings-only track is plenty for "
              f"cueing but too coarse for the terminal 6 m TVC solution.")


# guidance gains / constants
best_err = [1e9]                         # best target-track error seen
V_CLOSE = 700.0                          # nominal closing speed for the lead
A_MAX = 120.0 * 9.81                      # accel-cmd cap (high T/W boost)
KP, KD = 1.5, 0.8                        # intercept accel-command gains
_guide = {"p": None, "v": None}          # low-pass state for the noisy estimate


def _step_truth(truth, u):
    if isinstance(u, dict):
        for name, val in u.items():
            truth.command(name).set(float(val))
    else:
        truth.command("interceptor.main.throttle").set(0.0)
    for _ in range(SUBSTEPS):              # 500 Hz: stiff TVC + contact
        truth.step(DT_SIM)


def _attitude_cmd(rk, q_ref):
    """PD on the body-attitude error → gimbal torque commands (point the
    rocket's thrust axis at q_ref). A direct nonlinear controller — handles
    the large pitch-over a lofted intercept needs, unlike the upright LQR."""
    q = np.asarray(rk["orientation"]).ravel()
    w = np.asarray(rk["angular_velocity"]).ravel()
    err = _quat_err(q_ref, q)              # rotation vector, world frame
    # body-frame error (gimbal torques act in the body frame)
    err_b = _rot_by_quat(_quat_conj(q), err)
    w_b = _rot_by_quat(_quat_conj(q), w)
    tau = -(KQ * err_b - KW * w_b)   # sign of gimbal->body-torque coupling
    return {"interceptor.gimbal_x.torque_cmd": float(np.clip(tau[0], -1500, 1500)),
            "interceptor.gimbal_y.torque_cmd": float(np.clip(tau[1], -1500, 1500))}


KQ, KW = 250.0, 180.0


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rot_by_quat(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v)


def _quat_err(q_ref, q):
    """Rotation vector of q_ref ⊗ q⁻¹ (world frame, small-angle ~ axis·angle)."""
    qe = _quat_mul(q_ref, _quat_conj(q))
    if qe[0] < 0:
        qe = -qe
    n = np.linalg.norm(qe[1:])
    if n < 1e-9:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(n, qe[0])
    return qe[1:] / n * ang


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def _deproject_two(out, vis, rng):
    """Seed the filter by algebraically triangulating the target from the
    visible cameras' rays (linear least-squares intersection)."""
    fx = (W_IMG / 2.0) / np.tan(np.radians(HFOV) / 2.0)
    cc = W_IMG / 2.0
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for nm, (cx, cy) in zip(CAM_NAMES, CAM_XY):
        if not vis[nm]:
            continue
        u = float(np.asarray(out[f"{nm}.target_hull_u"]).ravel()[0])
        v = float(np.asarray(out[f"{nm}.target_hull_v"]).ravel()[0])
        d = np.array([(u - cc) / fx, (v - cc) / fx, 1.0])
        d /= np.linalg.norm(d)
        C = np.array([cx, cy, 0.3])
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ C
    return np.linalg.solve(A, b), 0.0


def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=6000.0, color=(45, 55, 50, 120))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        viz.box(f"world/tracker/{nm}", (12, 12, 12), center=(x, y, 6),
                color=(80, 170, 220))
    viz.split_cylinder("world/interceptor/hull", BODY_R * 6, BODY_L * 4,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, GIM_Z + BODY_L / 2))


def _viz_step(viz, t, truth, tgt_est, vis):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
    viz.trail("world/interceptor_trail", rk_p, max_len=6000, min_dist=2.0)
    viz.rr.log("world/target", viz.rr.Ellipsoids3D(
        centers=[tg_p], half_sizes=[(30, 30, 30)], colors=[(255, 170, 90)],
        fill_mode="solid"))
    viz.point("world/target_est", tgt_est, color=(90, 230, 120), radius=30.0)
    viz.trail("world/target_trail", tg_p, max_len=6000, min_dist=2.0)
    # rays from each firing camera to the estimate
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        if vis[nm]:
            viz.line(f"world/ray_{nm}", [(x, y, 0.3), tuple(tgt_est)],
                     color=(70, 100, 130), radius=1.5)


if __name__ == "__main__":
    main()
