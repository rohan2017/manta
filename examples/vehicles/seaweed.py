"""Seaweed — the recompile-on-damage doctrine, measured on one leg.

A finned submarine (forward prop, twin elevators, fixed tail fin) is
asked to fly the same 16 m leg at 2 m depth, three times, in three
side-by-side lanes:

  lane 1  clean vehicle, nominal controller     — the baseline
  lane 2  FOULED vehicle, nominal controller    — a 2 kg clump of
          seaweed hangs under the tail: more mass, more drag, a
          slight sink, and a speed-dependent trim moment. The stale
          model believes none of it. DEGRADED, NOT DANGEROUS: the
          vehicle plods along ~1.3 m below its commanded depth,
          level and stable — but it cannot close the leg at all
          (unmodeled drag slows it, fin authority is ½ρV²·A·CL, so
          slower means weaker, and the sag settles where the stale
          model's expectations keep failing). Meanwhile its tick
          cost J runs far above baseline: the model-health residual
          that triggers the refit on a real vehicle.
  lane 3  fouled vehicle, RECOMPILED controller — the model matches
          the water again: arrival within a quarter second of the
          clean baseline, depth held to centimetres. The seaweed is
          still attached; only the model changed.

The refit is stubbed — the rebuild is handed the true fouled world,
standing in for `manta.fit` run against flight telemetry (the reducer
arrives with the shiver integration; this demo measures what the
recompile buys once it exists). The rebuild itself is the real
on-board sequence: fresh `MPC(world)` + plan birth; generate + `cc
-O0` is ~a minute on a Pi-class board, and the stale controller —
lane 2, degraded but stable — is exactly what flies while it runs.

Run::

    .venv/bin/python -m examples.vehicles.seaweed            # rerun viz
    .venv/bin/python -m examples.vehicles.seaweed --no-viz   # headless
    .venv/bin/python -m examples.vehicles.seaweed --save out.rrd
"""

from __future__ import annotations

import math

import numpy as np

from manta import MPC, Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import (
    Aerofoil, AddedMass, ControlSurface, DragSurface, Mass, PointBuoy,
    Thruster,
)

from .._control import Pacer, common_args
from .._viz import Viz

RHO = 1025.0
MASS = 30.0
ZB = 0.02
DT, HORIZON = 0.25, 60
PLANT_DT = 0.05
DEPTH = -2.0
LEG = 16.0                     # the leg: 16 m straight at depth
LEG_CAP = 140                  # ticks (35 s) before a trial is abandoned
ARRIVE = 1.2                   # m — a fly-through gate, not a hover
LANE_DY = 3.0

#: the clump: ~2 kg of weed, nearly neutral (kelp floats), hanging
#: below and behind the hull — mass, drag, AND trim in one part set
WEED_MASS = 2.0
WEED_BUOY = 1.7e-3             # m³ → ~0.26 kg net sink
WEED_AT = (-0.5, 0.0, -0.18)


def build_world(*, seaweed: bool = False, lane_y: float = 0.0):
    c = Craft("sub")
    c.add(Mass("hull", mass=MASS, moi=(0.4, 4.0, 4.0)))
    c.add(AddedMass("am", translational=(3.0, 30.0, 30.0),
                    rotational=(0.05, 3.0, 3.0)))
    c.add(DragSurface.directional_quadratic(
        "drag", areas=(0.03, 0.35, 0.35), drag_coefficient=1.0))
    c.add(DragSurface("skin", force=(-4.0 / RHO, -10.0 / RHO,
                                     -10.0 / RHO)))
    c.add(DragSurface("spin", torque=(-2.5 / RHO, -8.0 / RHO,
                                      -8.0 / RHO)))
    c.add(PointBuoy("cb", volume=MASS / RHO,
                    mount_offset=(0.0, 0.0, ZB)))
    c.add(Thruster("prop", force=(50.0, 0.0, 0.0)))
    half = math.pi / 4.0
    c.add(Aerofoil("vfin", area=0.02, chord=0.1,
                   mount_offset=(-0.85, 0.0, 0.0),
                   mount_orientation=(math.cos(half), math.sin(half),
                                      0.0, 0.0)))
    tail = dict(area=0.012, chord=0.1, servo_gain=4.0,
                hinge_damping=0.4)
    c.add(ControlSurface("elev_l", mount_offset=(-0.8, 0.3, 0.0), **tail))
    c.add(ControlSurface("elev_r", mount_offset=(-0.8, -0.3, 0.0), **tail))

    if seaweed:
        # None of these carry state: the fouled craft has the SAME
        # state and input layout, so the controller Module shape is
        # unchanged and only the numbers differ — a parametric change,
        # handled by recompiling (the settled doctrine: a broken FIN
        # would be structural — a state leaves — and recompiles from
        # a world with the fin removed).
        c.add(Mass("weed", mass=WEED_MASS, moi=(0.01, 0.01, 0.01),
                   mount_offset=WEED_AT))
        c.add(PointBuoy("weed_buoy", volume=WEED_BUOY,
                        mount_offset=WEED_AT))
        c.add(DragSurface.directional_quadratic(
            "weed_drag", areas=(0.015, 0.05, 0.05), drag_coefficient=1.2,
            mount_offset=WEED_AT))

    w = (World(name="weedsub")
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=RHO)))
    w.add_craft(c, position=(0.0, lane_y, DEPTH))
    bounds = {"prop.throttle": (0.0, 1.0),
              "elev_l.deflection_cmd": (-0.4, 0.4),
              "elev_r.deflection_cmd": (-0.4, 0.4)}
    return w, bounds


def fly_trial(mpc, plant, goal, *, viz=None, pacer=None, t0=0.0,
              sub_path="world/sub0", trail="world/trail0",
              color=(80, 160, 255)):
    """One from-rest trial to arrival (or the cap): metrics + J trace."""
    spec = mpc.module().port("x").manifold
    rt = TargetNumpy(mpc)
    st = plant.state["sub"]
    x0 = spec.pack_any({f"sub.{n}": np.asarray(v, dtype=float).ravel()
                        for n, v in st.items()})
    rt.reset_plan(mpc.plan(x0, goal).U)

    zs, Js = [], []
    t = t0
    for k in range(LEG_CAP):
        st = plant.state["sub"]
        flat = {f"sub.{n}": np.asarray(v, dtype=float).ravel()
                for n, v in st.items()}
        u = rt.tick(spec.pack_any(flat), goal)
        Js.append(rt.J)
        for j in range(round(DT / PLANT_DT)):
            plant.step(PLANT_DT, u=u)
            t = t0 + k * DT + (j + 1) * PLANT_DT
            if pacer is not None:
                pacer.pace(t)
            if viz is not None and viz.due(t):
                st = plant.state["sub"]
                viz.t(t)
                viz.pose(sub_path,
                         np.asarray(st["position"], dtype=float).ravel(),
                         np.asarray(st["orientation"],
                                    dtype=float).ravel())
                viz.trail(trail,
                          np.asarray(st["position"],
                                     dtype=float).ravel(), color=color)
        pos = np.asarray(plant.state["sub"]["position"],
                         dtype=float).ravel()
        zs.append(pos[2])
        if np.linalg.norm(pos - goal) < ARRIVE:
            break
    zs = np.asarray(zs)
    return {
        "t": (k + 1) * DT,
        "arrived": np.linalg.norm(pos - goal) < ARRIVE,
        "cruise": float(np.abs(zs[len(zs) // 3:] - DEPTH).mean()),
        "sag": float(max(0.0, DEPTH - zs.min())),
        "J": float(np.median(Js)),
        "t_end": t,
    }


def main() -> None:
    p = common_args(__doc__)
    p.add_argument("--save", metavar="FILE.rrd", default=None,
                   help="record to a .rrd instead of a live viewer")
    args = p.parse_args()

    clean_w, bounds = build_world()
    mpc_nom = MPC(clean_w, u_bounds=bounds, horizon=HORIZON, dt=DT,
                  w_pos_terminal=120.0, w_vel_terminal=0.0)

    viz = (None if args.no_viz
           else Viz("manta/seaweed", addr=args.viz_addr, save=args.save))
    pacer = Pacer() if viz is not None and not args.save else None

    lanes = (("clean + nominal", False),
             ("FOULED + stale model", True),
             ("fouled + RECOMPILED", True))
    if viz is not None:
        for i, (_, fouled) in enumerate(lanes):
            sub = f"world/sub{i}"
            viz.box(f"{sub}/hull", (1.8, 0.24, 0.24),
                    color=(150, 150, 158))
            viz.box(f"{sub}/vfin", (0.10, 0.02, 0.28),
                    center=(-0.85, 0.0, 0.0), color=(120, 120, 130))
            if fouled:
                viz.box(f"{sub}/weed", (0.5, 0.3, 0.12), center=WEED_AT,
                        color=(90, 120, 50))
            viz.box(f"world/goal{i}", (0.2, 0.2, 0.2),
                    center=(LEG, i * LANE_DY, DEPTH), color=(90, 200, 90))

    print(f"the leg: {LEG:.0f} m straight at {-DEPTH:.0f} m depth — "
          f"flown three ways\n")
    colors = ((80, 160, 255), (230, 150, 60), (110, 220, 110))
    results = []
    t0 = 0.0
    for i, (label, fouled) in enumerate(lanes):
        if i == 2:
            print("\n*** refit + recompile: rebuilding MPC(world) "
                  "against the fouled model (fit stubbed with the true "
                  "fouled world; on a Pi: generate + cc -O0 ~1 min, "
                  "with lane 2's degraded-but-stable flight covering "
                  "the wait) ***\n")
            ctrl = MPC(build_world(seaweed=True)[0], u_bounds=bounds,
                       horizon=HORIZON, dt=DT,
                       w_pos_terminal=120.0, w_vel_terminal=0.0)
        else:
            ctrl = mpc_nom
        plant = TargetNumpy(Sim(
            build_world(seaweed=fouled, lane_y=i * LANE_DY)[0]))
        goal = np.array([LEG, i * LANE_DY, DEPTH])
        print(f"lane {i + 1}: {label}")
        r = fly_trial(ctrl, plant, goal, viz=viz, pacer=pacer, t0=t0,
                      sub_path=f"world/sub{i}", trail=f"world/trail{i}",
                      color=colors[i])
        results.append((label, r))
        t0 = r["t_end"] + 1.0
        if i == 1:
            print(f"  model-health residual: median tick J {r['J']:.1f} "
                  f"vs baseline {results[0][1]['J']:.1f} — "
                  f"{r['J'] / results[0][1]['J']:.1f}x elevated, "
                  f"sustained => refit")

    print(f"\n{'lane':<28}{'time':>8}{'cruise depth err':>18}"
          f"{'peak sag':>10}")
    for label, r in results:
        t_s = f"{r['t']:.1f} s" if r["arrived"] else "DNF"
        print(f"{label:<28}{t_s:>8}{r['cruise']:>16.2f} m"
              f"{r['sag']:>8.2f} m")
    print("\nsame leg, same seaweed on lanes 2 and 3: the only "
          "difference is whose model matches the water.")


if __name__ == "__main__":
    main()
