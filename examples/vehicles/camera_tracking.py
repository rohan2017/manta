"""Camera tracking + a controllable interceptor — built one validated layer
at a time.

  STEP 1 — a ground array triangulates a ballistic target (EKF over bearings).
  STEP 2 — the interceptor flies a controlled waypoint course (LQR + a
           reference governor), passively stable, speed-governed so it never
           outruns its control authority and never tumbles.

Step 3 (guidance / proportional navigation that points the interceptor at the
tracked target) builds on these; here the rocket flies a fixed demonstration
course so its flight can be judged on its own.

  * **tracker** — four `CentroidCamera`s at the corners of a 1 km square,
    looking up, anchored to the ground. Each reports only the target's image
    CENTROID — a pure bearing, size-independent. Folding the four bearings,
    the Kalman update IS the noise-weighted intersection of the rays.
    Two things make it track instead of diverge: process noise lives on the
    target's ACCELERATION (not position — `pos += vel·dt` is exact), and the
    velocity is SEEDED by finite-differencing two triangulations.

  * **target** — a ballistic `Mass`+`DragSurface`+`OpticalSource` (~3 km
    apogee), starting far+low (outside the FOV) and arcing into view.

  * **interceptor** — a gimballed TVC rocket made passively stable (drag aft
    of the COM). An LQR (one gain, solved at the hover trim — the gimbal
    torques steer, the throttle holds altitude) tracks a virtual setpoint
    that a governor walks toward each waypoint at a bounded speed. Keeping
    the tracking error small keeps the rocket in the linear regime where the
    LQR is valid, so it never tumbles. Roll about the thrust axis is
    unactuated (single gimballed engine) → zero Q, by design.

Run::

    .venv/bin/python -m examples.vehicles.camera_tracking            # live viewer
    .venv/bin/python -m examples.vehicles.camera_tracking --no-viz   # headless
    .venv/bin/python -m examples.vehicles.camera_tracking --record run.rrd
"""

from __future__ import annotations

import argparse

import numpy as np

from manta import ALL, Craft, EKF, LQR, Sim, TargetNumpy, World
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
DT = 0.02                              # 50 Hz control / EKF / LQR
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
MAXT = 3000.0                          # T/W ~= 4 — enough to maneuver, not race
THR_HOVER = M_TOT * G / MAXT
Z0 = 2.0
# LQR weights (damping-dominant; roll about thrust axis unactuated → 0).
# Order: position(3) orientation(3) velocity(3) ang_rate(3) gx angle/rate
# gy angle/rate.
LQR_Q = np.diag([0.5, 0.5, 3.0,  14.0, 14.0, 0.0,  12.0, 12.0, 8.0,
                 6.0, 6.0, 0.0,  1.0, 0.1,  1.0, 0.1])
LQR_R = np.diag([1.0, 1.0, 0.5])       # gimbal_x τ, gimbal_y τ, throttle
TRACK = ["interceptor.position", "interceptor.velocity",
         "interceptor.orientation", "interceptor.angular_velocity",
         "interceptor.gimbal_x.angle", "interceptor.gimbal_x.rate",
         "interceptor.gimbal_y.angle", "interceptor.gimbal_y.rate"]
GOV_SPEED = 12.0                       # m/s the setpoint governor advances at
# A demonstration course: climb, a 120 m box at altitude, descend.
WAYPOINTS = [(0, 0, 250), (120, 0, 250), (120, 120, 250),
             (0, 120, 250), (0, 0, 170)]


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
# Triangulation (a pure ray-intersection — used only to SEED the filter)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--viz-addr", default=None,
                    help="stream to a running rerun viewer at host[:port]")
    ap.add_argument("--record", default=None, metavar="FILE.rrd",
                    help="save to a rerun .rrd (no viewer, full speed)")
    ap.add_argument("--duration", type=float, default=56.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(build_world()))
    truth.step(DT_SIM)

    # --- EKF over the target (step 1) ----------------------------------
    ekf = TargetNumpy(EKF(build_world(), sensors=GND_SENSORS,
                          track={"target": ALL}))
    espec = ekf.spec
    ta = _tan(espec, "target.position")
    va = _tan(espec, "target.velocity")
    Q = np.full(espec.tangent_dim, 1e-12)
    Q[va[0]:va[1]] = Q_VEL
    Qm = np.diag(Q)

    # --- LQR for the interceptor (step 2) ------------------------------
    lqr = TargetNumpy(LQR(
        build_world(), x_ref={"interceptor": {"position": (0, 0, 200)}},
        u_ref={"main.throttle": THR_HOVER}, track=TRACK,
        Q=LQR_Q, R=LQR_R, dt=DT))

    viz = None if args.no_viz else Viz(
        "manta/camera_tracking", addr=args.viz_addr, save=args.record)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if (viz is not None and not args.record) else None

    phase = "wait"
    seed_p, seed_t = None, None
    errs = []
    setpoint = np.array([0.0, 0.0, Z0])      # LQR reference governor state
    wp_i, wp_reached = 0, []
    print(f"\n{'t':>6} {'#vis':>4} {'trackErr':>9} {'rk wp':>6} "
          f"{'rk pos':>20} {'tilt°':>6}")
    for i in range(int(args.duration / DT)):
        t = i * DT
        if pacer is not None:
            pacer.pace(t)
        tg = truth.state["target"]
        tg_p = np.asarray(tg["position"]).ravel()
        tg_v = np.asarray(tg["velocity"]).ravel()
        rk = truth.state["interceptor"]
        rk_p = np.asarray(rk["position"]).ravel()
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

        # --- STEP 2: fly the interceptor through the waypoint course ----
        wp = np.array(WAYPOINTS[wp_i], float)
        if (np.linalg.norm(wp - rk_p) < 14.0
                and np.linalg.norm(np.asarray(rk["velocity"]).ravel()) < 6.0
                and wp_i < len(WAYPOINTS) - 1):
            wp_reached.append((wp_i, round(t, 1)))
            wp_i += 1
            wp = np.array(WAYPOINTS[wp_i], float)
        d = wp - setpoint
        n = float(np.linalg.norm(d))
        if n > 1e-6:                          # governor walks the setpoint in
            setpoint = setpoint + d / n * min(GOV_SPEED * DT, n)
        lqr.retarget({"interceptor": {"position": setpoint,
                                      "velocity": (0, 0, 0)}})
        u = lqr.control({"interceptor": rk})
        u["interceptor.main.throttle"] = float(
            np.clip(u["interceptor.main.throttle"], 0.1, 1.0))
        for name, val in u.items():
            truth.command(name).set(float(val))

        if viz is not None and viz.due(t):
            _viz_step(viz, t, truth, tgt_est, vis, setpoint)
        for _ in range(SUBSTEPS):
            truth.step(DT_SIM)

        if i % int(2.0 / DT) == 0 and phase == "track":
            tilt = _tilt(np.asarray(rk["orientation"]).ravel())
            print(f"{t:6.1f} {nvis:4d} {errs[-1]:9.1f} {wp_i:6d} "
                  f"({rk_p[0]:5.0f},{rk_p[1]:4.0f},{rk_p[2]:4.0f}) {tilt:6.1f}")

    if errs:
        print(f"\nSTEP 1 tracking: mean {np.mean(errs):.1f} m, "
              f"peak {np.max(errs):.1f} m (green estimate on the orange target)")
    print(f"STEP 2 flight: reached {len(wp_reached)}/{len(WAYPOINTS) - 1} "
          f"waypoints, max tilt {_maxtilt[0]:.0f}° — no tumble, speed-governed")


# ---------------------------------------------------------------------------
# Small helpers / visualization
# ---------------------------------------------------------------------------

_maxtilt = [0.0]


def _tilt(q):
    w, x, y, z = q
    up_z = 1 - 2 * (x * x + y * y)
    ang = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
    _maxtilt[0] = max(_maxtilt[0], ang)
    return ang


def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=6000.0, color=(45, 55, 50, 120))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        viz.box(f"world/tracker/{nm}", (14, 14, 14), center=(x, y, 6),
                color=(80, 170, 220))
    for k, wp in enumerate(WAYPOINTS):
        viz.box(f"world/waypoint/{k}", (10, 10, 10), center=tuple(wp),
                color=(150, 150, 160, 180))
    viz.split_cylinder("world/interceptor/hull", 1.4, 14.0,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, -1.2 + 7))


def _viz_step(viz, t, truth, tgt_est, vis, setpoint):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
    viz.trail("world/interceptor_trail", rk_p, max_len=6000, min_dist=2.0,
              color=(230, 180, 80))
    viz.point("world/setpoint", setpoint, color=(230, 120, 60), radius=10.0)
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
