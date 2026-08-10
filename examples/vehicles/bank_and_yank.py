"""Bank-and-yank — a commanded yaw deviation on a submarine that CANNOT yaw.

The hull: one forward-only prop and two independent elevators on the
tail, mounted left and right. Collective deflection pitches; a
differential split rolls. There is no rudder and no thruster off the
surge axis — the actuator set spans **pitch and roll only**, and fin
authority is ½ρV²·A·CL: zero at rest, bought with speed.

The command is pure guidance-language, through the runtime objective:
hold this depth, make 1.2 m/s, and put the nose on a course 45° off
the bow — the heading term (`set_objective(heading=d)`, Euler-free:
nose·d = qᵀM(d)q). No yaw actuator exists, so the only mechanism
physics offers is the aircraft turn, and `MPC(world)` discovers it
from the model alone — no maneuver script:

    roll over   — split the elevators, heel the hull toward the turn
    yank        — pull collective; pitch, banked, IS world-frame yaw
    roll back   — level out and run on course; the centre of buoyancy
                  sits above the mass centre (`PointBuoy` at +z), so
                  the righting moment makes wings-level the cheap
                  place to be once the turn is done

Two solver layers share the work, and the demo shows the seam
honestly. The fixed-work tick flies the plan — but initiating a bank
from level flight is a roll×pitch SADDLE (zero first-order yaw
payoff), invisible to warm Gauss-Newton sweeps, so the closed loop
alone stalls a few degrees short of the commanded course. The mission
layer closes it: when the heading residual persists, re-birth the
plan (`plan(multi_start=True)` from the current state — the same
replan-on-residual doctrine controld will run). Two replans take the
nose to 45.0° ± half a degree, wings level.

Run::

    .venv/bin/python -m examples.vehicles.bank_and_yank            # rerun viz
    .venv/bin/python -m examples.vehicles.bank_and_yank --no-viz   # headless
    .venv/bin/python -m examples.vehicles.bank_and_yank --save out.rrd
    .venv/bin/python -m examples.vehicles.bank_and_yank --viz-addr host:9876

On WSL, ``--save`` (scrub the .rrd in a viewer afterwards) or
``--viz-addr`` (stream to a Windows-native viewer) beat the WSLg
window. Live runs are paced to real time; the elevons animate their
actual servo deflections — watch the split (differential = roll)
become collective (pitch) as the bank turns into the yank.
"""

from __future__ import annotations

import numpy as np

import math

from manta import MPC, Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    Aerofoil, AddedMass, ControlSurface, DragSurface, Mass, PointBuoy,
    RotationalDrag, Thruster,
)

from .._control import Pacer, common_args
from .._geom import euler_zyx, wrap_pi
from .._viz import Viz

RHO = 1025.0
MASS = 30.0
ZB = 0.02            # CB height above CG (m) — the righting arm
COURSE_DEG = 45.0                     # the commanded yaw deviation
SPEED = 1.2                           # m/s along the new course
DT, HORIZON = 0.25, 60
PLANT_DT = 0.05


def build_world() -> tuple[World, dict[str, tuple[float, float]]]:
    c = Craft("sub")
    c.add(Mass("hull", mass=MASS, moi=(0.4, 4.0, 4.0)))
    # slender-body added mass: transverse ~100 % of displaced mass —
    # what makes velocity-aligned flight (and therefore a real turn,
    # not a skid) strongly preferred
    c.add(AddedMass("am", translational=(3.0, 30.0, 30.0),
                    rotational=(0.05, 3.0, 3.0)))
    c.add(DragSurface.directional_quadratic(
        "drag", areas=(0.03, 0.35, 0.35), drag_coefficient=1.0))
    c.add(DragSurface("skin", force=(-4.0 / RHO, -10.0 / RHO,
                                     -10.0 / RHO)))
    c.add(RotationalDrag("spin", torque=(-1.0 / RHO, -8.0 / RHO,
                                         -8.0 / RHO)))
    # Neutrally buoyant, centre of buoyancy ZB above the mass centre:
    # zero net force at every attitude, pure righting torque when
    # heeled — the physics that closes the maneuver (roll BACK).
    c.add(PointBuoy("cb", volume=MASS / RHO,
                    mount_offset=(0.0, 0.0, ZB)))
    c.add(Thruster("prop", force=(50.0, 0.0, 0.0)))
    # STATIC STABILITY. The transverse added mass above buys the
    # nose-first preference but also the Munk moment — the classic
    # destabilizing broach torque on any slender hull (~61 N·m/rad
    # here at cruise). In the PITCH plane the elevators' fixed area
    # weathercocks it away (~139 N·m/rad restoring). The YAW plane
    # has no control surface at all — that is the premise — so it
    # gets a fixed symmetric tail fin (~116 N·m/rad): the hull is
    # statically stable in both planes, and the controller shapes the
    # trajectory instead of fencing with a broach.
    half = math.pi / 4.0
    c.add(Aerofoil("vfin", area=0.02, chord=0.1,
                   mount_offset=(-0.85, 0.0, 0.0),
                   mount_orientation=(math.cos(half), math.sin(half),
                                      0.0, 0.0)))
    tail = dict(area=0.016, chord=0.1, servo_gain=4.0,
                hinge_damping=0.4)
    c.add(ControlSurface("elev_l", mount_offset=(-0.8, 0.35, 0.0), **tail))
    c.add(ControlSurface("elev_r", mount_offset=(-0.8, -0.35, 0.0), **tail))
    w = (World(name="banksub")
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=RHO)))
    w.add_craft(c)
    bounds = {"prop.throttle": (0.0, 1.0),
              "elev_l.deflection_cmd": (-0.4, 0.4),
              "elev_r.deflection_cmd": (-0.4, 0.4)}
    return w, bounds


def main() -> None:
    p = common_args(__doc__)
    p.add_argument("--save", metavar="FILE.rrd", default=None,
                   help="record to a .rrd instead of a live viewer "
                        "(full speed, scrub afterwards — best on WSL)")
    args = p.parse_args()

    w, bounds = build_world()
    mpc = MPC(w, u_bounds=bounds, horizon=HORIZON, dt=DT)
    spec = mpc.module().port("x").manifold
    d_cmd = np.array([np.cos(np.radians(COURSE_DEG)),
                      np.sin(np.radians(COURSE_DEG)), 0.0])
    objective = dict(w_position=(0.0, 0.0, 40.0),      # depth-hold only
                     velocity=SPEED * d_cmd, w_velocity=(15.0,) * 3,
                     heading=d_cmd, w_heading=30.0,
                     P=np.zeros((mpc.nx, mpc.nx)))
    print(f"command: nose to {COURSE_DEG:.0f}°, {SPEED} m/s, hold "
          f"depth — pitch+roll authority only\n")

    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    print("plan birth (multi-start — initiating a bank is a roll×pitch "
          "saddle single seeds cannot see)...")
    plan = mpc.plan(x0, goal=(0.0, 0.0, 0.0), multi_start=True,
                    **objective)
    print(f"  cost {plan.cost:.1f}, converged={plan.converged}\n")

    rt = TargetNumpy(mpc)
    rt.set_objective(position=(0.0, 0.0, 0.0),
                     w_position=(0.0, 0.0, 40.0),
                     velocity=SPEED * d_cmd, w_velocity=(15.0,) * 3,
                     heading=d_cmd, w_heading=30.0,
                     P=np.zeros((mpc.nx, mpc.nx)))
    rt.reset_plan(plan.U)
    plant = TargetNumpy(Sim(build_world()[0]))

    viz = (None if args.no_viz
           else Viz("manta/bank_and_yank", addr=args.viz_addr,
                    save=args.save))
    if viz is not None:
        viz.box("world/sub/hull", (1.8, 0.24, 0.24),
                color=(150, 150, 158))
        viz.box("world/sub/vfin", (0.10, 0.02, 0.28),
                center=(-0.85, 0.0, 0.0), color=(120, 120, 130))
        # elevons are POSED children: geometry logged once at each
        # hinge frame's origin, the frame itself re-posed every frame
        # with the servo's actual deflection (rotation about body y)
        viz.box("world/sub/elev_l/geom", (0.10, 0.28, 0.02),
                color=(200, 120, 80))
        viz.box("world/sub/elev_r/geom", (0.10, 0.28, 0.02),
                color=(80, 140, 220))
        # the commanded course, drawn as a long thin rail
        half_c = np.radians(COURSE_DEG) / 2.0
        viz.box("world/course/geom", (40.0, 0.03, 0.03),
                center=(20.0, 0.0, 0.0), color=(90, 200, 90))
        viz.pose("world/course", (0.0, 0.0, 0.0),
                 (np.cos(half_c), 0.0, 0.0, np.sin(half_c)))
    # pace live viewing to real time; .rrd recording runs full speed
    pacer = Pacer() if viz is not None and not args.save else None

    def hinge_quat(deflection) -> tuple:
        h = 0.5 * float(np.asarray(deflection).ravel()[0])
        return (np.cos(h), 0.0, np.sin(h), 0.0)

    def nose_err_deg(st) -> float:
        q = np.asarray(st["orientation"], dtype=float).ravel()
        n = np.array([1 - 2 * (q[2] ** 2 + q[3] ** 2),
                      2 * (q[1] * q[2] + q[0] * q[3]),
                      2 * (q[1] * q[3] - q[0] * q[2])])
        return float(np.degrees(np.arccos(np.clip(n @ d_cmd, -1, 1))))

    ticks = int((args.duration or 30.0) / DT)
    print(f"{'t':>5} {'roll':>6} {'pitch':>6} {'yaw':>6} "
          f"{'speed':>6} {'nose err':>9}")
    max_bank, replans = 0.0, 0
    for k in range(ticks):
        st = plant.state["sub"]
        flat = {f"sub.{n}": np.asarray(v, dtype=float).ravel()
                for n, v in st.items()}
        x = spec.pack_any(flat)
        # the mission layer: a persistent heading residual means the
        # tick is stuck at the bank-initiation saddle — re-birth the
        # plan from the current state (replan-on-residual doctrine)
        if k > 0 and k % 20 == 0 and nose_err_deg(st) > 3.0:
            rt.reset_plan(mpc.plan(x, goal=(0.0, 0.0, 0.0),
                                   multi_start=True, **objective).U)
            replans += 1
            print(f"      *** heading residual persists "
                  f"({nose_err_deg(st):.1f}°) -> replan #{replans} ***")
        u = rt.tick(x)
        # the controller holds u for one period; the PLANT (and the
        # viewer) run at the fine step — poses log every 50 ms, so the
        # replay is smooth even though the tick is 4 Hz
        for j in range(round(DT / PLANT_DT)):
            plant.step(PLANT_DT, u=u)
            t = k * DT + (j + 1) * PLANT_DT
            if pacer is not None:
                pacer.pace(t)
            if viz is not None and viz.due(t):
                st = plant.state["sub"]
                viz.t(t)
                viz.pose("world/sub",
                         np.asarray(st["position"], dtype=float).ravel(),
                         np.asarray(st["orientation"],
                                    dtype=float).ravel())
                viz.pose("world/sub/elev_l", (-0.8, 0.35, 0.0),
                         hinge_quat(st["elev_l.deflection"]))
                viz.pose("world/sub/elev_r", (-0.8, -0.35, 0.0),
                         hinge_quat(st["elev_r.deflection"]))
                viz.trail("world/trail",
                          np.asarray(st["position"],
                                     dtype=float).ravel())

        st = plant.state["sub"]
        q = np.asarray(st["orientation"], dtype=float).ravel()
        yaw_, pitch, roll = euler_zyx(q)
        speed = float(np.linalg.norm(
            np.asarray(st["velocity"], dtype=float)))
        max_bank = max(max_bank, abs(roll))
        t = (k + 1) * DT
        if k % 4 == 3:
            dg = np.rad2deg
            print(f"{t:5.1f} {dg(roll):+6.0f} {dg(pitch):+6.0f} "
                  f"{dg(wrap_pi(yaw_)):+6.0f} {speed:6.2f} "
                  f"{nose_err_deg(st):8.1f}°")

    err = nose_err_deg(plant.state["sub"])
    print(f"\nbank-and-yank: peak bank {np.rad2deg(max_bank):.0f}°, "
          f"nose {err:.1f}° off the commanded course, final roll "
          f"{np.rad2deg(roll):+.0f}° (righted), {replans} replan(s)")
    print("no yaw actuator ever existed — the turn is roll × pitch, "
          "discovered from the model; the last degrees are the "
          "mission layer's replan closing a saddle the tick cannot.")


if __name__ == "__main__":
    main()
