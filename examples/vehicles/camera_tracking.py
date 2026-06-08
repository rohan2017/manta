"""Camera tracking — a TVC rocket intercepts a ballistic target it sees.

Two craft and a single joint EKF that estimates BOTH:

  * **target** — a small, low-mass projectile (a `Mass` + a `DragSurface`
    + an `OpticalSource` so it has a visual signature). It is *launched
    ballistically*: an initial velocity, then gravity + drag. No actuator,
    no guidance — it just flies its arc, up over the rocket and down.

  * **interceptor** — a thrust-vectored rocket (a gimballed engine flown by
    LQR, the same machine as `rocket.py`) with a wide-FOV `Camera` looking
    up its nose, an `IMU`, and a GPS (`PositionSensor`). It launches off the
    pad, finds the target in its camera, and pursues it.

The estimation is the point. The rocket never gets the target's true state.
It gets *bounding boxes*: the camera projects the target's known-size
ellipsoid to an image box, and the EKF folds those four edge pixels as a
measurement (bearing from the box center, range from its apparent size —
`Camera(bbox_sigma=...)`). Because a box depends on BOTH craft poses, the
filter is one joint block: the rocket's own IMU+GPS keep its block tight,
and the camera informs the target.

The rocket climbs to a holding altitude and keeps its nose camera up while
the filter locks (chasing a not-yet-converged fix would saturate its tilt
and dump it); once the target-position covariance collapses it engages. The
intercept itself is a stable altitude-HOLD plus lateral pursuit of the
EKF's estimate — the fast-descending target falls THROUGH the hold altitude
onto the laterally-aligned rocket (chasing the target's altitude would
overshoot, since a thrust-limited rocket can't arrest a climb in time).

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz
    .venv/bin/python -m examples.vehicles.camera_tracking --truth   # guide on truth
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import Craft, EKF, LQR, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField, OpticalField
from manta.parts import (
    Camera, DragSurface, IMU, Mass, OpticalSource, PositionSensor,
    RevoluteJoint, Thruster,
)

from .._control import Pacer
from .._viz import Viz

# --- world / rates ---------------------------------------------------------
G = 9.81
DT_SIM, SUBSTEPS = 0.004, 5          # 250 Hz physics under a 50 Hz loop
DT_CTRL = DT_SIM * SUBSTEPS          # 50 Hz — LQR + EKF rate
V_CLOSE = 12.0                       # nominal closing speed for the lead
KZ, KDZ = 3.0, 2.5                   # vertical-closure throttle gains (P, rate-damp)
# The gimbal→body-torque gain scales with thrust, but the LQR is solved at
# ONE trim throttle (see rocket.py). Keep the throttle in a band around that
# trim so the attitude loop stays near its design gain (wide swings make it
# tip). The band still spans descend-slowly … climb-hard for the intercept.
THR_TRIM_G = 1.55                    # trim climb accel (g) the LQR is solved at
THR_MIN, THR_MAX = 0.09, 0.42        # throttle band. MIN is below hover (≈0.18)
#   so the rocket can actually BRAKE a climb (else it coasts past the hold
#   altitude); MAX is capped so the high-thrust attitude gain stays in design.
ENGAGE_SIGMA = 2.0                   # target-pos σ (m) below which pursuit engages
ENGAGE_HOLD = 5                      # consecutive converged frames before engaging
HOLD_Z = 10.0                        # waiting altitude during acquisition
HIT_RADIUS = 1.5                     # intercept radius (target 0.6 + body + margin)

# --- rocket (a slimmed rocket.py: no legs, launches powered) ---------------
M_BODY, M_ENG = 8.0, 1.5
M_TOT = M_BODY + M_ENG
MAXT = 520.0                          # full thrust (N): T/W ≈ 5.6
GIM_Z, ENG_Z = -0.8, -0.18            # gimbal pivot / engine below it
BODY_R, BODY_L = 0.16, 2.0
NOSE_Z = 1.0                          # camera up on the nose
Z0 = 2.0                              # launch altitude (just off the pad)
W_IMG, H_IMG, HFOV = 640, 480, 115.0
BBOX_SIGMA = 2.0                     # camera box-edge noise (px)
GPS_SIGMA, GYRO_SIGMA = 0.05, 2e-3   # rocket GPS (m) / gyro (rad/s) noise

# --- target (ballistic) ----------------------------------------------------
TGT_SEMI = (0.6, 0.6, 0.6)            # known size — the range cue
TGT_MASS = 0.8
TGT_P0 = (8.0, 0.0, 16.0)             # launch point (high, nearly overhead)
TGT_V0 = (-3.0, 0.0, 4.0)             # launch → arcs over the rocket, descends


def build_world():
    # --- interceptor -------------------------------------------------------
    rocket = Craft("interceptor")
    rocket.add(Mass("body", mass=M_BODY, moi=(2.8, 2.8, 0.35),
                    transform=(0, 0, 0.3)))
    rocket.add(DragSurface.isotropic_quadratic("aero", area=0.07,
                                               drag_coefficient=0.8))
    # Two-axis TVC gimbal: engine Mass + Thruster ride the inner frame.
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=30.0, damping=1.0, transform=(0, 0, GIM_Z))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=30.0, damping=1.0)
    gy.add(Mass("engine", mass=M_ENG, moi=(0.02, 0.02, 0.01),
                transform=(0, 0, ENG_Z)))
    gy.add(Thruster("main", force=(0, 0, MAXT), transform=(0, 0, ENG_Z)))
    gx.add(gy)
    rocket.add(gx)
    # Nose camera looking up (+z); IMU + GPS for the rocket's own block.
    rocket.add(Camera("cam", width=W_IMG, height=H_IMG, hfov_deg=HFOV,
                      bbox_sigma=BBOX_SIGMA, transform=(0, 0, NOSE_Z)))
    rocket.add(IMU("imu", gyro_noise_sigma=GYRO_SIGMA, accel_noise_sigma=2e-2))
    rocket.add(PositionSensor("gps", position_noise_sigma=GPS_SIGMA))

    # --- target ------------------------------------------------------------
    target = Craft("target")
    target.add(Mass("body", mass=TGT_MASS, moi=(0.05, 0.05, 0.05)))
    target.add(DragSurface.isotropic_quadratic("aero", area=0.05,
                                               drag_coefficient=0.6))
    target.add(OpticalSource("hull", semi_axes=TGT_SEMI, label=1))

    w = (World()
         .add_field(GravityField(g=(0, 0, -G)))
         .add_field(FluidField().add_uniform(density=1.225))
         .add_field(OpticalField()))
    w.add_craft(rocket, position=(0, 0, Z0))
    w.add_craft(target, position=TGT_P0, velocity=TGT_V0)
    return w, rocket, target


# ---------------------------------------------------------------------------
# Rocket LQR (gimbal TVC) — the rocket.py recipe, ascent trim
# ---------------------------------------------------------------------------

def build_lqr(w):
    THR_TRIM = M_TOT * THR_TRIM_G * G / MAXT
    Q = np.diag([8.0, 8.0, 4.0,        # position (aggressive lateral chase)
                 6.0, 6.0, 0.0,        # orientation (roll unactuated)
                 5.0, 5.0, 4.0,        # velocity
                 1.0, 1.0, 0.0,        # angular rate (roll unactuated)
                 0.5, 0.05, 0.5, 0.05])  # gimbal x/y angle,rate
    R = np.diag([2.0, 2.0, 0.4])       # gimbal_x τ, gimbal_y τ, throttle
    lqr_t = LQR(
        w, x_ref={"interceptor": {"position": (0.0, 0.0, Z0)}},
        u_ref={"main.throttle": THR_TRIM},
        track=["interceptor.position", "interceptor.velocity",
               "interceptor.orientation", "interceptor.angular_velocity",
               "interceptor.gimbal_x.angle", "interceptor.gimbal_x.rate",
               "interceptor.gimbal_y.angle", "interceptor.gimbal_y.rate"],
        Q=Q, R=R, dt=DT_CTRL)
    return TargetNumpy(lqr_t)


# ---------------------------------------------------------------------------
# EKF — joint filter over both crafts. Camera edges + rocket IMU/GPS.
# ---------------------------------------------------------------------------

EDGES = ("xmin", "ymin", "xmax", "ymax")
CAM_EDGES = [f"interceptor.cam.target_hull_{s}" for s in EDGES]
ROCKET_SENSORS = ["interceptor.gps.position", "interceptor.imu.gyro"]
EKF_INPUTS = ["interceptor.main.throttle",
              "interceptor.gimbal_x.torque_cmd",
              "interceptor.gimbal_y.torque_cmd"]


def make_ekf(w):
    ekf_t = EKF(w, sensors=[*ROCKET_SENSORS, *CAM_EDGES], inputs=EKF_INPUTS)
    return TargetNumpy(ekf_t)


def tan_slice(spec, name):
    s = next(s for s in spec.slots if s.name == name)
    return s.tangent_offset, s.tangent_offset + s.tangent_dim


def deproject(box, cam_pos, cam_quat):
    """Rough world position of the target from one bounding box: bearing
    from the box center, range from its apparent width (known semi-axes).
    Used to seed the filter on first detection."""
    fx = (W_IMG / 2.0) / np.tan(np.radians(HFOV) / 2.0)
    cx, cy = W_IMG / 2.0, H_IMG / 2.0
    u = 0.5 * (box[0] + box[2])
    v = 0.5 * (box[1] + box[3])
    width = max(box[2] - box[0], 1.0)
    rng = 2.0 * float(np.mean(TGT_SEMI)) * fx / width      # diameter·f / px
    d_cam = np.array([(u - cx) / fx, (v - cy) / fx, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    # camera +z is the optical axis; rotate the ray into world via cam_quat
    # (Quat[World, Part], so apply maps Part→World).
    w, x, y, z = cam_quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return np.asarray(cam_pos) + rng * (R @ d_cam)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--viz-addr", default=None)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--truth", action="store_true",
                   help="guide on the target's true state (filter still runs)")
    args = p.parse_args()

    noise_rng = np.random.default_rng(0)
    w, _, _ = build_world()
    truth = TargetNumpy(Sim(w))
    truth.step(DT_SIM)                  # prime the output bus (sensor readings)
    lqr = build_lqr(build_world()[0])
    ekf = make_ekf(build_world()[0])
    spec = ekf.spec

    # Filter prior: rocket known (launches from a known pad), target unknown
    # until first detection.
    ekf.reset(state={"interceptor": {"position": (0.0, 0.0, Z0)}},
              P=np.eye(spec.tangent_dim) * 1e-4)
    Q = np.full(spec.tangent_dim, 1e-4)
    ta, tb = tan_slice(spec, "target.position")
    va, vb = tan_slice(spec, "target.velocity")
    # The target is purely ballistic — its dynamics (gravity + drag) are
    # KNOWN, so a TIGHT process model lets the filter ride the physics and
    # converge robustly; the boxes only correct it. (A loose Q here lets the
    # weakly-observed cross-range run away and the filter diverges.)
    Q[ta:tb] = 1e-3
    Q[va:vb] = 1e-2
    Qm = np.diag(Q)

    viz = None if args.no_viz else Viz("manta/camera_tracking", addr=args.viz_addr)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if viz is not None else None   # real-time playback for viz

    locked = False
    n = int(args.duration / DT_CTRL)
    print(f"\n{'t':>5} {'rocket pos':>20} {'target est':>20} "
          f"{'est err':>8} {'miss':>7} {'vis':>4} {'thr':>5}")
    miss = min_miss = 1e9
    hit_t = None
    for i in range(n):
        t = i * DT_CTRL
        if pacer is not None:
            pacer.pace(t)
        rk = truth.state["interceptor"]
        tg = truth.state["target"]
        rk_p = np.asarray(rk["position"]).ravel()
        tg_p = np.asarray(tg["position"]).ravel()
        miss = float(np.linalg.norm(rk_p - tg_p))
        if hit_t is not None and miss > min_miss + 2.0:
            break                            # past closest approach — stop

        out = truth.outputs()["interceptor"]
        # Real measurements are noisy — perturb each by its declared σ so the
        # EKF is doing genuine fusion, not reading an oracle.
        box = [float(np.asarray(out[f"cam.target_hull_{s}"]).ravel()[0])
               + noise_rng.normal(0.0, BBOX_SIGMA) for s in EDGES]
        vis = float(np.asarray(out["cam.target_hull_vis"]).ravel()[0]) > 0.5

        # --- EKF predict + measurement folds -------------------------------
        u_ekf = {"interceptor.main.throttle": _last_thr[0],
                 "interceptor.gimbal_x.torque_cmd": _last_u[0],
                 "interceptor.gimbal_y.torque_cmd": _last_u[1]}
        ekf.predict(dt=DT_CTRL, t=t, u=u_ekf, Q=Qm)
        ekf.update("interceptor.gps.position",
                   np.asarray(rk["position"]).ravel() + noise_rng.normal(0, GPS_SIGMA, 3))
        ekf.update("interceptor.imu.gyro",
                   np.asarray(out["imu.gyro"]).ravel() + noise_rng.normal(0, GYRO_SIGMA, 3))
        if vis:
            if not locked:                # seed the target from first sight
                cam_p = rk_p + np.asarray([0, 0, NOSE_Z])
                cam_q = np.asarray(rk["orientation"]).ravel()
                p0 = deproject(box, cam_p, cam_q)
                st = ekf.state_dict()
                st["target"] = {"position": p0, "velocity": (0.0, 0.0, 0.0)}
                P = ekf.P
                for a, b, val in ((ta, tb, 4.0), (va, vb, 25.0)):
                    for k in range(a, b):
                        P[k, k] = val
                ekf.reset(state=st, P=P)
                locked = True
                print(f"   t={t:.2f}  target acquired, deproject "
                      f"{np.round(p0, 1)}")
            for s, z in zip(EDGES, box):
                ekf.update(f"interceptor.cam.target_hull_{s}", np.array([z]))

        est = ekf.state_dict()
        tgt_est = np.asarray(est["target"]["position"]).ravel()
        tgt_v_est = np.asarray(est["target"]["velocity"]).ravel()
        est_err = float(np.linalg.norm(tgt_est - tg_p))
        rk_v = np.asarray(rk["velocity"]).ravel()
        # Target-position uncertainty (filter confidence). Don't chase a
        # not-yet-converged estimate — a bad early fix would saturate the
        # rocket's tilt and dump it. Climb and hold the camera up until the
        # boxes have pinned the target, THEN engage pursuit.
        pos_sig = float(np.sqrt(np.trace(ekf.P[ta:tb, ta:tb])))
        if locked and pos_sig < ENGAGE_SIGMA:
            _converged[0] += 1
        else:
            _converged[0] = 0
        # Same acquire→engage profile whether we guide on the estimate or
        # (with --truth) on the target's true state; --truth just isolates
        # the guidance/control from the estimation.
        engaged = _converged[0] >= ENGAGE_HOLD

        if args.truth:
            guide_p = tg_p
            guide_v = np.asarray(tg["velocity"]).ravel()
        else:
            guide_p, guide_v = tgt_est, tgt_v_est

        if engaged:
            # --- lateral pursuit, altitude HELD at the intercept band ------
            # Chasing the target's altitude overshoots: the rocket builds
            # climb velocity it can't arrest inside the stability throttle
            # band. Instead HOLD a fixed intercept altitude (a stable PD hover)
            # and only pursue laterally — the fast-descending target falls
            # THROUGH that altitude onto the laterally-aligned rocket. Lead
            # the lateral track by range/closing-speed (→ pure pursuit close).
            rng = float(np.linalg.norm(guide_p - rk_p))
            t_go = float(np.clip(rng / V_CLOSE, 0.0, 1.5))
            pip = guide_p + guide_v * t_go
            aim = np.array([pip[0], pip[1], rk_p[2]])   # z handled by throttle
            lqr.retarget({"interceptor": {"position": aim,
                                          "velocity": np.zeros(3)}})
            u = lqr.control({"interceptor": rk})
            az = float(np.clip(KZ * (HOLD_Z - rk_p[2]) - KDZ * rk_v[2],
                               -0.6 * G, 2.5 * G))
            thr = float(np.clip(M_TOT * (G + az) / MAXT, THR_MIN, THR_MAX))
        else:
            # Acquire/converge: hold a steady upright climb to a waiting
            # altitude. A stable, near-vertical camera gives clean boxes over
            # several frames so the filter locks before any aggressive
            # maneuver (which would swing the camera and break the lock).
            lqr.retarget({"interceptor": {"position": (0.0, 0.0, HOLD_Z),
                                          "velocity": np.zeros(3)}})
            u = lqr.control({"interceptor": rk})
            dz = HOLD_Z - rk_p[2]
            az = float(np.clip(1.5 * dz - 1.0 * rk_v[2], -0.5 * G, 1.5 * G))
            thr = float(np.clip(M_TOT * (G + az) / MAXT, THR_MIN, THR_MAX))
        u["interceptor.main.throttle"] = thr
        _last_thr[0] = thr
        _last_u[0] = u["interceptor.gimbal_x.torque_cmd"]
        _last_u[1] = u["interceptor.gimbal_y.torque_cmd"]

        for name_u, val in u.items():
            truth.command(name_u).set(float(val))
        for k in range(SUBSTEPS):            # 250 Hz: catch the fast crossing
            truth.step(DT_SIM)
            d = float(np.linalg.norm(
                np.asarray(truth.state["interceptor"]["position"]).ravel()
                - np.asarray(truth.state["target"]["position"]).ravel()))
            if d < min_miss:
                min_miss = d
            if d < HIT_RADIUS and hit_t is None:
                hit_t = t + (k + 1) * DT_SIM

        if viz is not None and viz.due(t):   # throttle to ~30 Hz
            _viz_step(viz, t, truth, tgt_est, box, vis)
        if i % int(0.5 / DT_CTRL) == 0:
            print(f"{t:>5.2f} {np.round(rk_p, 1)!s:>20} "
                  f"{np.round(tgt_est, 1)!s:>20} {est_err:>8.2f} "
                  f"{miss:>7.2f} {int(vis):>4} {_last_thr[0]:>5.2f}")

    if hit_t is not None:
        print(f"\n*** INTERCEPT at t={hit_t:.2f} s — closest "
              f"{min_miss:.2f} m ***")
    else:
        print(f"\nclosest approach: {min_miss:.2f} m (no intercept)")


_last_thr = [0.0]
_last_u = [0.0, 0.0]
_converged = [0]


def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=60.0, color=(55, 60, 70, 120))
    viz.split_cylinder("world/interceptor/hull", BODY_R, BODY_L,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, GIM_Z + BODY_L / 2))
    viz.rr.log("image/frame", viz.rr.Boxes2D(
        mins=[[0, 0]], sizes=[[W_IMG, H_IMG]], colors=[(70, 70, 80)]),
        static=True)


def _viz_step(viz, t, truth, tgt_est, box, vis):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
    viz.arrow("world/interceptor/los", (0, 0, NOSE_Z), (0, 0, 4.0),
              color=(120, 220, 140), radius=0.05)
    # Trails live at world level, NOT under the posed `world/interceptor`
    # entity — a child would inherit the rocket's transform and the path
    # would ride along with the rocket instead of staying world-fixed.
    viz.trail("world/interceptor_trail", rk_p, max_len=4000, min_dist=0.05)
    viz.rr.log("world/target", viz.rr.Ellipsoids3D(
        centers=[tg_p], half_sizes=[TGT_SEMI], colors=[(255, 170, 90)],
        fill_mode="solid"))
    viz.point("world/target_est", tgt_est, color=(90, 230, 120), radius=0.3)
    viz.trail("world/target_trail", tg_p, max_len=4000, min_dist=0.05)
    if vis:
        viz.rr.log("image/boxes", viz.rr.Boxes2D(
            mins=[[box[0], box[1]]], sizes=[[box[2] - box[0], box[3] - box[1]]],
            colors=[(255, 170, 90)]))


if __name__ == "__main__":
    main()
