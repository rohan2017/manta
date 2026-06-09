"""Camera tracking — a ground array triangulates a ballistic target.

Step 1 of the demo, built and validated on its own: a fixed ground array of
four `CentroidCamera`s in a square watches a ballistic target and an EKF
recovers its 3-D state from the bearings alone.

  * **tracker** — four cameras at the corners of a 1 km square, looking up,
    anchored to the ground (a static craft free-falls under gravity
    otherwise). Each reports only the target's image CENTROID — a pure
    bearing, independent of the target's size. Folding the four bearings, the
    Kalman update IS the noise-weighted intersection of the rays:
    triangulation with no triangulation code.

  * **target** — a `Mass` + `DragSurface` + `OpticalSource` on a ballistic
    arc (~3 km apogee, ~5 km downrange), starting far + low (outside the
    upward field of view) and climbing into view.

Two things make the filter actually track instead of diverging:
  * process noise lives on the target's ACCELERATION (velocity state), not
    its position — `pos += vel·dt` is exact, so independent position noise
    only severs the position↔velocity correlation the filter needs to read
    velocity off the bearings;
  * the velocity is SEEDED by finite-differencing two triangulations, not
    guessed — a 100 m/s seed error takes tens of seconds of noisy bearings to
    burn off, by which point the track has already wandered.

The interceptor rocket is parked on its pad here; flying it is step 2.

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
SUBSTEPS = 10                          # 500 Hz physics (plenty for this scene)
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

    rocket = build_rocket()

    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField(g=(0, 0, -G)))
         .add_field(FluidField().add_uniform(density=RHO))
         .add_field(OpticalField())
         .add_field(cf))
    w.add_craft(tracker, position=(0, 0, 0))
    w.add_craft(target, position=TGT_P0, velocity=TGT_V0)
    w.add_craft(rocket, position=(0, 0, 2.0))
    return w


def build_rocket():
    """Interceptor — parked on its pad in step 1; flown in step 2."""
    rocket = Craft("interceptor")
    rocket.add(Mass("body", mass=60.0, moi=(80.0, 80.0, 1.5),
                    transform=(0, 0, 0.5)))
    rocket.add(DragSurface.directional_quadratic(
        "aero", areas=(0.6, 0.6, 0.08), drag_coefficient=0.8,
        transform=(0, 0, -0.6)))
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0, transform=(0, 0, -1.2))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=4000.0, damping=40.0)
    gy.add(Mass("engine", mass=15.0, moi=(0.5, 0.5, 0.2), transform=(0, 0, -0.3)))
    gy.add(Thruster("main", force=(0, 0, 16000.0), transform=(0, 0, -0.3)))
    gx.add(gy)
    rocket.add(gx)
    rocket.add(Collider("foot", stiffness=4e4, damping=6e3, friction=6e3,
                        transform=(0, 0, -2.0)))
    return rocket


# ---------------------------------------------------------------------------
# Triangulation (a pure ray-intersection — used only to SEED the filter)
# ---------------------------------------------------------------------------

def triangulate(out, vis):
    """Least-squares intersection of the visible cameras' bearing rays."""
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
        M = np.eye(3) - np.outer(d, d)     # projector onto the ray's complement
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
    ap.add_argument("--duration", type=float, default=46.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    truth = TargetNumpy(Sim(build_world()))
    truth.step(DT_SIM)
    ekf = TargetNumpy(EKF(build_world(), sensors=GND_SENSORS,
                          track={"target": ALL}))
    spec = ekf.spec
    ta = _tan(spec, "target.position")
    va = _tan(spec, "target.velocity")
    # Process noise on the ACCELERATION only (velocity state); position is
    # exact under pos += vel·dt.
    Q = np.full(spec.tangent_dim, 1e-12)
    Q[va[0]:va[1]] = Q_VEL
    Qm = np.diag(Q)

    viz = None if args.no_viz else Viz(
        "manta/camera_tracking", addr=args.viz_addr, save=args.record)
    if viz is not None:
        _viz_setup(viz)
    pacer = Pacer() if (viz is not None and not args.record) else None

    phase = "wait"            # wait → seed → track
    seed_p, seed_t = None, None
    errs = []
    print(f"\n{'t':>6} {'#vis':>4} {'posErr':>8} {'velErr':>8}")
    for i in range(int(args.duration / DT)):
        t = i * DT
        if pacer is not None:
            pacer.pace(t)
        tg = truth.state["target"]
        tg_p = np.asarray(tg["position"]).ravel()
        tg_v = np.asarray(tg["velocity"]).ravel()
        out = truth.outputs()["tracker"]
        vis = visible(out)
        nvis = sum(vis.values())

        tgt_est = None
        if phase == "wait":
            if nvis >= 2:                       # first multi-camera sighting
                seed_p, seed_t = triangulate(out, vis), t
                phase = "seed"
        elif phase == "seed":
            if t - seed_t >= SEED_DT and nvis >= 2:   # finite-difference v-seed
                p2 = triangulate(out, vis)
                v0 = (p2 - seed_p) / (t - seed_t)
                P0 = np.full(spec.tangent_dim, 1e-9)
                P0[ta[0]:ta[1]] = 50.0
                P0[va[0]:va[1]] = 30.0
                ekf.reset(state={"target": {"position": p2, "velocity": v0}},
                          P=np.diag(P0))
                phase = "track"
                print(f"   locked at t={t:.1f} "
                      f"(velocity seed error {np.linalg.norm(v0 - tg_v):.1f} m/s)")
        else:
            ekf.predict(dt=DT, t=t, Q=Qm)
            for nm in CAM_NAMES:
                if not vis[nm]:
                    continue
                for c in ("u", "v"):
                    z = float(np.asarray(out[f"{nm}.target_hull_{c}"]).ravel()[0])
                    ekf.update(f"tracker.{nm}.target_hull_{c}",
                               np.array([z + rng.normal(0, GND_PIX)]))
            est = ekf.state_dict()["target"]
            tgt_est = np.asarray(est["position"]).ravel()
            errs.append(float(np.linalg.norm(tgt_est - tg_p)))
            if i % int(2.0 / DT) == 0:
                ve = np.asarray(est["velocity"]).ravel()
                print(f"{t:6.1f} {nvis:4d} {errs[-1]:8.1f} "
                      f"{np.linalg.norm(ve - tg_v):8.1f}")

        if viz is not None and viz.due(t):
            _viz_step(viz, t, truth, tgt_est, vis)
        for _ in range(SUBSTEPS):
            truth.step(DT_SIM)

    if errs:
        print(f"\ntracking: mean {np.mean(errs):.1f} m, peak {np.max(errs):.1f} m "
              f"over a {np.linalg.norm(TGT_P0):.0f}->{0:.0f} m ballistic arc — "
              f"the green estimate tracks the orange target throughout.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _viz_setup(viz):
    viz.plane("world/ground", z=0.0, size=6000.0, color=(45, 55, 50, 120))
    for nm, (x, y) in zip(CAM_NAMES, CAM_XY):
        viz.box(f"world/tracker/{nm}", (14, 14, 14), center=(x, y, 6),
                color=(80, 170, 220))
    viz.split_cylinder("world/interceptor/hull", 1.4, 14.0,
                       colors=((220, 222, 228), (180, 60, 50)))
    viz.pose("world/interceptor/hull", (0, 0, -1.2 + 7))


def _viz_step(viz, t, truth, tgt_est, vis):
    rk_p = np.asarray(truth.state["interceptor"]["position"]).ravel()
    rk_q = np.asarray(truth.state["interceptor"]["orientation"]).ravel()
    tg_p = np.asarray(truth.state["target"]["position"]).ravel()
    viz.t(t)
    viz.pose("world/interceptor", rk_p, rk_q)
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
