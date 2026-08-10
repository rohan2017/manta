"""Bank-and-yank — a turn performed by a submarine that CANNOT yaw.

The hull: one forward-only prop and two independent elevators on the
tail, mounted left and right. Collective deflection pitches; a
differential split rolls. There is no rudder and no thruster off the
surge axis — the actuator set spans **pitch and roll only**, and fin
authority is ½ρV²·A·CL: zero at rest, bought with speed.

Command a waypoint 45° off the bow and the controller must change
heading anyway. The only mechanism physics offers is the aircraft
turn, and `MPC(world)` discovers it from the model alone — no yaw
term, no attitude cost, no maneuver script:

    roll over   — split the elevators, heel the hull toward the turn
    yank        — pull collective; pitch, banked, IS world-frame yaw
    roll back   — level out and run in; the centre of buoyancy sits
                  above the mass centre (`PointBuoy` at +z), so the
                  righting moment makes wings-level the cheap place
                  to be once the turn is done

Plan birth uses `plan(multi_start=True)`: the payoff of rolling is
bilinear (roll × pitch — nothing improves until BOTH move), the exact
saddle family Gauss-Newton cannot see from a single seed. The flight
loop is the fixed-work tick from the compiled Module, closed against
the world's own Sim at a finer step.

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

from manta import MPC, Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    AddedMass, ControlSurface, DragSurface, Mass, PointBuoy, Thruster,
)

from .._control import Pacer, common_args
from .._geom import euler_zyx, wrap_pi
from .._viz import Viz

RHO = 1025.0
MASS = 30.0
ZB = 0.02            # CB height above CG (m) — the righting arm
GOAL = np.array([8.0, 8.0, 0.0])      # 45° off the bow, same depth
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
    c.add(DragSurface("spin", torque=(-0.8 / RHO, -8.0 / RHO,
                                      -8.0 / RHO)))
    # Neutrally buoyant, centre of buoyancy ZB above the mass centre:
    # zero net force at every attitude, pure righting torque when
    # heeled — the physics that closes the maneuver (roll BACK).
    c.add(PointBuoy("cb", volume=MASS / RHO,
                    mount_offset=(0.0, 0.0, ZB)))
    c.add(Thruster("prop", force=(50.0, 0.0, 0.0)))
    tail = dict(area=0.012, chord=0.1, servo_gain=4.0,
                hinge_damping=0.4)
    c.add(ControlSurface("elev_l", mount_offset=(-0.8, 0.3, 0.0), **tail))
    c.add(ControlSurface("elev_r", mount_offset=(-0.8, -0.3, 0.0), **tail))
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
    mpc = MPC(w, u_bounds=bounds, horizon=HORIZON, dt=DT,
              w_pos_terminal=120.0, w_vel_terminal=0.0)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    bearing = np.rad2deg(np.arctan2(GOAL[1], GOAL[0]))
    print(f"goal {GOAL} — a {bearing:.0f}° heading change, "
          f"with pitch+roll authority only\n")

    print("plan birth (multi-start — the roll×pitch payoff is a "
          "saddle single seeds cannot see)...")
    plan = mpc.plan(x0, GOAL, multi_start=True)
    X = mpc._rollout(x0, plan.U.T)
    qs = mpc.module().port("x").manifold.slot("sub.orientation")
    rolls = np.array([euler_zyx(q)[2] for q in
                      X[:, qs.ambient_offset:qs.ambient_offset + 4]])
    d = np.rad2deg
    print(f"  cost {plan.cost:.1f}, converged={plan.converged}")
    print(f"  the plan's own arc: roll in to {d(rolls.min()):+.0f}°, "
          f"yank, roll back out to {d(rolls[-1]):+.0f}° at arrival\n")

    rt = TargetNumpy(mpc)
    rt.reset_plan(plan.U)
    plant = TargetNumpy(Sim(build_world()[0]))
    spec = mpc.module().port("x").manifold

    viz = (None if args.no_viz
           else Viz("manta/bank_and_yank", addr=args.viz_addr,
                    save=args.save))
    if viz is not None:
        viz.box("world/sub/hull", (1.8, 0.24, 0.24),
                color=(150, 150, 158))
        # elevons are POSED children: geometry logged once at each
        # hinge frame's origin, the frame itself re-posed every frame
        # with the servo's actual deflection (rotation about body y)
        viz.box("world/sub/elev_l/geom", (0.10, 0.28, 0.02),
                color=(200, 120, 80))
        viz.box("world/sub/elev_r/geom", (0.10, 0.28, 0.02),
                color=(80, 140, 220))
        viz.box("world/goal", (0.2, 0.2, 0.2), center=tuple(GOAL),
                color=(90, 200, 90))
    # pace live viewing to real time; .rrd recording runs full speed
    pacer = Pacer() if viz is not None and not args.save else None

    def hinge_quat(deflection) -> tuple:
        h = 0.5 * float(np.asarray(deflection).ravel()[0])
        return (np.cos(h), 0.0, np.sin(h), 0.0)

    ticks = int((args.duration or 15.0) / DT)
    print(f"{'t':>5} {'roll':>6} {'pitch':>6} {'yaw':>6} "
          f"{'depth':>7} {'dist':>6}")
    max_bank = 0.0
    for k in range(ticks):
        st = plant.state["sub"]
        flat = {f"sub.{n}": np.asarray(v, dtype=float).ravel()
                for n, v in st.items()}
        u = rt.tick(spec.pack_any(flat), GOAL)
        for _ in range(round(DT / PLANT_DT)):
            plant.step(PLANT_DT, u=u)

        q = np.asarray(st["orientation"], dtype=float).ravel()
        pos = np.asarray(st["position"], dtype=float).ravel()
        yaw_, pitch, roll = euler_zyx(q)
        max_bank = max(max_bank, abs(roll))
        t = (k + 1) * DT
        if pacer is not None:
            pacer.pace(t)
        if viz is not None and viz.due(t):
            viz.t(t)
            viz.pose("world/sub", pos, q)
            viz.pose("world/sub/elev_l", (-0.8, 0.30, 0.0),
                     hinge_quat(st["elev_l.deflection"]))
            viz.pose("world/sub/elev_r", (-0.8, -0.30, 0.0),
                     hinge_quat(st["elev_r.deflection"]))
            viz.trail("world/trail", pos)
        dist = np.linalg.norm(pos - GOAL)
        if k % 2 == 1:
            d = np.rad2deg
            print(f"{t:5.1f} {d(roll):+6.0f} {d(pitch):+6.0f} "
                  f"{d(wrap_pi(yaw_)):+6.0f} {pos[2]:+7.2f} "
                  f"{dist:6.2f}")
        if dist < 0.8:
            # a forward-only prop cannot hover — past the goal the
            # vehicle would simply orbit it; the demo ends at arrival
            break

    print(f"\nbank-and-yank: peak bank {np.rad2deg(max_bank):.0f}°, "
          f"final roll {np.rad2deg(roll):+.0f}° (righted), "
          f"arrived {dist:.2f} m from the goal at t={t:.1f} s")
    print("no yaw actuator ever existed — the turn is roll × pitch, "
          "discovered from the model.")


if __name__ == "__main__":
    main()
