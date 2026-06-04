"""Airplane — a passively stable airframe flown on real control surfaces.

No IMU, no stabilizer loops: the airframe itself is stable, like a
trainer RC plane. The main `Naca00xx` wing is mounted at +10° incidence
with the ballast `Mass` hung below its quarter-chord; a neutral
horizontal tail weathervanes the nose back to trim, and the vertical fin
does the same in yaw. Every control is a real surface: small Naca panels
riding saturating `Joint` hinges — two ailerons at the wingtip trailing
edges, an elevator behind the tail, a rudder behind the fin. The keys
command a target deflection and a tiny P-servo (plus joint damping)
drives each hinge there; the aerodynamic hinge moment, the servo
reaction on the airframe, and the surface's lift all fall out of the
physics. A nose `Thruster` plays propeller, with a touch of reaction
counter-torque about the thrust axis.

Controls:  S/W elevator (pitch up/down)   A/D rudder (yaw left/right)
           Q/E ailerons (roll left/right)  X/Z throttle up/down

Run::

    .venv/bin/python -m examples.vehicles.airplane --keyboard
    .venv/bin/python -m examples.vehicles.airplane             # scripted
    .venv/bin/python -m examples.vehicles.airplane --no-viz    # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import DragSurface, Joint, Mass, Naca00xx, Thruster

from .._control import common_args, make_controller
from .._viz import Viz

# --- airframe geometry (m, kg; x forward, y left, z up) --------------------
MASS      = 3.0
CG        = (0.0, 0.0, -0.18)          # ballast below the wing quarter-chord
WING_INC  = np.radians(10.0)           # wing incidence (built-in AoA)
AIL_HINGE = 0.225                      # wing TE behind the quarter-chord mount
AIL_SPAN  = 1.0                        # aileron hinge station along the span
TAIL_X    = -1.2                       # horizontal tail quarter-chord
ELEV_X    = -1.32                      # elevator hinge (tail TE)
FIN_Z     = 0.12                       # vertical-stab mid-height
SURF_OFF  = (-0.05, 0.0, 0.0)          # surface panel centre behind its hinge

# --- control limits + servo --------------------------------------------------
MAX_AIL   = np.radians(8.0)
MAX_ELEV  = np.radians(20.0)
MAX_RUD   = np.radians(20.0)
THR_MAX   = 1.0
THR_RATE  = 0.4                        # throttle slew per second (X/Z held)
SERVO_KP  = 20.0                       # hinge servo: torque = KP·(target − angle)
AIL_TRIM  = np.radians(0.13)           # aileron trim vs. prop torque, per throttle


def _hinge(name: str, pos: tuple, axis: tuple) -> Joint:
    """A control-surface hinge: saturating Joint + a small rotor Mass so the
    DOF has inertia for the servo to push against."""
    j = Joint(name, mode="saturating", stall_torque=5.0, damping=0.3,
              axis=axis, transform=pos)
    j.add(Mass(f"{name}_m", mass=0.02, moi=(0.002, 0.002, 0.002),
               transform=SURF_OFF))
    return j


def build_world():
    a = Craft("plane")
    a.add(Mass("body", mass=MASS, moi=(0.4, 0.35, 0.7), transform=CG))

    # Main wing at +10° incidence: tilt the chord/normal pair in the
    # part frame so level flight already sees α = WING_INC.
    ci, si = np.cos(WING_INC), np.sin(WING_INC)
    a.add(Naca00xx("wing", area=0.72, CL_max=1.2, CD_0=0.02, induced_k=0.25,
                   chord_axis=(ci, 0.0, si), normal_axis=(-si, 0.0, ci)))

    # Ailerons: hinges at the wingtip trailing edges, spanwise (y) axis.
    for name, sy in (("ail_l", +1.0), ("ail_r", -1.0)):
        h = _hinge(name, (-AIL_HINGE, sy * AIL_SPAN, 0.0), (0.0, 1.0, 0.0))
        h.add(Naca00xx(f"{name}_s", area=0.04, CL_max=1.2, CD_0=0.01,
                       induced_k=0.1, transform=SURF_OFF))
        a.add(h)

    # Horizontal tail (neutral incidence) + elevator on its trailing edge.
    a.add(Naca00xx("tail", area=0.09, CL_max=1.2, CD_0=0.01, induced_k=0.1,
                   transform=(TAIL_X, 0.0, 0.0)))
    elev = _hinge("elev", (ELEV_X, 0.0, 0.0), (0.0, 1.0, 0.0))
    elev.add(Naca00xx("elev_s", area=0.05, CL_max=1.2, CD_0=0.01,
                      induced_k=0.1, transform=SURF_OFF))
    a.add(elev)

    # Vertical stabilizer above the tail + rudder on its trailing edge.
    # Same Naca panel, stood upright: lift acts along body y.
    a.add(Naca00xx("fin", area=0.06, CL_max=1.2, CD_0=0.01, induced_k=0.1,
                   chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 1.0, 0.0),
                   transform=(TAIL_X, 0.0, FIN_Z)))
    rud = _hinge("rud", (ELEV_X, 0.0, FIN_Z), (0.0, 0.0, 1.0))
    rud.add(Naca00xx("rud_s", area=0.03, CL_max=1.2, CD_0=0.01,
                     induced_k=0.1, chord_axis=(1.0, 0.0, 0.0),
                     normal_axis=(0.0, 1.0, 0.0), transform=SURF_OFF))
    a.add(rud)

    # Fuselage drag (at CG height, so it adds no pitch moment) + propeller:
    # thrust through the CG, with a small reaction torque about the axis.
    a.add(DragSurface.isotropic_quadratic("fuselage", area=0.02,
                                          drag_coefficient=0.4,
                                          transform=(0.0, 0.0, CG[2])))
    a.add(Thruster("prop", force=(12.0, 0.0, 0.0), torque=(-0.04, 0.0, 0.0),
                   transform=(0.45, 0.0, CG[2])))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=1.225)))
    w.add_craft(a, position=(0.0, 0.0, 60.0), velocity=(13.0, 0.0, 0.0))
    return w


def _quat(axis, angle):
    """wxyz quaternion for a rotation of `angle` about `axis` (viz only)."""
    h = 0.5 * angle
    ax = np.asarray(axis, dtype=float)
    return (np.cos(h), *(np.sin(h) * ax))


def _euler(q_wxyz):
    """ZYX yaw/pitch/roll (rad) from a wxyz quaternion."""
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    return yaw, pitch, roll


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.002
    duration = args.duration or (1e9 if args.keyboard else 24.0)

    sim = TargetNumpy(Sim(build_world()))

    # Scripted flight: throttle up and climb, bank into a right turn,
    # level the wings, then throttle off and glide.
    script = [(1.0, 2.0, {"x"}),       # throttle up → climb
              (5.0, 6.0, {"z"}),       # back to cruise power
              (8.0, 8.4, {"e"}),       # roll right into a bank
              (11.0, 11.3, {"q"}),     # roll back wings-level
              (14.0, 17.0, {"z"})]     # throttle to zero → glide
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  S/W pitch up/down   A/D yaw   Q/E roll   "
              "X/Z throttle   (Ctrl-C to quit)\n")

    FIXED, MOVING = (120, 150, 210), (240, 150, 60)
    viz = None if args.no_viz else Viz("manta/airplane")
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=300.0, color=(70, 110, 70, 160))
        viz.box("world/plane/fus", (0.95, 0.035, 0.035), center=(-0.45, 0, 0),
                color=(150, 150, 158))
        viz.box("world/plane/wing/panel", (0.15, 1.2, 0.008),
                center=(-0.075, 0, 0), color=FIXED)
        viz.pose("world/plane/wing", (0, 0, 0), _quat((0, 1, 0), -WING_INC))
        viz.box("world/plane/tail", (0.075, 0.3, 0.006),
                center=(TAIL_X - 0.0375, 0, 0), color=FIXED)
        viz.box("world/plane/fin", (0.075, 0.005, 0.12),
                center=(TAIL_X - 0.0375, 0, FIN_Z), color=FIXED)
        viz.box("world/plane/ail_l/s", (0.05, 0.2, 0.004), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/ail_r/s", (0.05, 0.2, 0.004), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/elev/s", (0.04, 0.3, 0.004), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/rud/s", (0.04, 0.003, 0.1), center=SURF_OFF,
                color=MOVING)

    # Hinge name → (viz path, hinge position, hinge axis) for the per-tick
    # deflection poses.
    hinges = {"ail_l": ((-AIL_HINGE, +AIL_SPAN, 0.0), (0, 1, 0)),
              "ail_r": ((-AIL_HINGE, -AIL_SPAN, 0.0), (0, 1, 0)),
              "elev":  ((ELEV_X, 0.0, 0.0), (0, 1, 0)),
              "rud":   ((ELEV_X, 0.0, FIN_Z), (0, 0, 1))}

    throttle = 0.3
    n = int(duration / dt)
    print(f"{'t (s)':>6} {'x (m)':>8} {'alt (m)':>8} {'|v| (m/s)':>9} "
          f"{'pitch°':>7} {'roll°':>7} {'head°':>7} {'thr':>5}")
    try:
        for i in range(n):
            t = i * dt
            ctrl.update(t)

            # Direct key → deflection mapping (no inner loops). Hinge sign
            # convention: +angle swings the panel's trailing edge toward
            # +axis×chord, so +elev = TE up = nose up, +rud = nose right,
            # +ail = that wing down.
            throttle = float(np.clip(
                throttle + (ctrl.held("x") - ctrl.held("z")) * THR_RATE * dt,
                0.0, THR_MAX))
            # A whisker of aileron trim cancels the propeller's torque roll
            # (exactly what a real plane's trim tab is set for).
            ail = MAX_AIL * (ctrl.held("q") - ctrl.held("e")) \
                - AIL_TRIM * throttle
            targets = {"elev": MAX_ELEV * (ctrl.held("s") - ctrl.held("w")),
                       "rud": MAX_RUD * (ctrl.held("d") - ctrl.held("a")),
                       "ail_l": +ail, "ail_r": -ail}

            st = sim.state["plane"]    # re-fetch: step() swaps the dict
            st["prop.throttle"] = throttle
            for name, target in targets.items():
                st[f"{name}.torque_cmd"] = \
                    SERVO_KP * (target - float(st[f"{name}.angle"]))
            sim.step(dt)

            st = sim.state["plane"]
            p = np.asarray(st["position"]).ravel()
            if viz is not None and i % 5 == 0:
                q = np.asarray(st["orientation"]).ravel()
                viz.t(t)
                viz.pose("world/plane", p, q)
                for name, (pos, axis) in hinges.items():
                    viz.pose(f"world/plane/{name}", pos,
                             _quat(axis, float(st[f"{name}.angle"])))
                viz.arrow("world/plane/thrust", (0.5, 0, CG[2]),
                          (0.6 * throttle, 0, 0), color=(235, 80, 80),
                          radius=0.02)
                viz.trail("world/plane/trail", p)

            if p[2] < 0:
                print(f"\nlanded at t = {t:.2f} s, range = "
                      f"{np.hypot(p[0], p[1]):.1f} m")
                break
            if (i + 1) % int(1.0 / dt) == 0:
                v = np.asarray(st["velocity"]).ravel()
                yaw, pitch, roll = _euler(np.asarray(st["orientation"]).ravel())
                print(f"{t:>6.2f} {p[0]:>8.1f} {p[2]:>8.1f} "
                      f"{np.linalg.norm(v):>9.2f} {np.degrees(pitch):>+7.1f} "
                      f"{np.degrees(roll):>+7.1f} {np.degrees(yaw):>+7.1f} "
                      f"{throttle:>5.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
