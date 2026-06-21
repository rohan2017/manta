"""Airplane — a passively stable airframe flown on real control surfaces.

No IMU, no stabilizer loops: the airframe itself is stable, like a
trainer RC plane. The main `Aerofoil` wing is mounted at +10° incidence
with the ballast `Mass` hung below its quarter-chord; a neutral
horizontal tail weathervanes the nose back to trim, and the vertical fin
does the same in yaw. Every control is a real `ControlSurface` — a wing
section with a deflectable trailing-edge flap, no hinge joint: two
ailerons at the wingtips, the elevator is the whole horizontal tail, the
rudder the whole vertical fin. The keys command a target deflection; each
surface's one-state servo (with its stall torque and aerodynamic hinge
moment — so a surface can blow back at speed) drives it there, and the
combined wing+flap lift falls out of the physics. Because the surfaces
are massless flaps rather than articulated joints, the craft stays a flat
rigid body — one deflection state per surface, not a joint angle+rate, and
no per-step articulated solve: 17 states instead of 21, and the sim step
and EKF predict run several times faster. A nose
`Thruster` plays propeller, with a touch of reaction counter-torque about
the thrust axis. Tricycle-gear `Collider`s rest the plane on a
`CollisionField` runway — it starts parked, and the wing's built-in
incidence floats it off the ground on its own once the takeoff roll
passes flying speed.

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
from manta.fields import CollisionField, FluidField, GravityField
from manta.parts import (
    Aerofoil, Collider, ControlSurface, DragSurface, Mass, Thruster,
)

from .._control import Pacer, TerminalController, common_args, make_controller
from .._viz import Viz

# --- airframe geometry (m, kg; x forward, y left, z up) --------------------
MASS      = 3.0
CG        = (0.0, 0.0, -0.18)          # ballast below the wing quarter-chord
WING_INC  = np.radians(4.0)            # wing incidence (built-in AoA); the
                                       # Aerofoil lift model is stronger than
                                       # the old one, so less incidence trims
                                       # the same cruise lift
WING_CHORD = 0.3                       # mean wing chord (Reynolds reference)
AIL_SPAN  = 1.0                        # aileron section station along the span
TAIL_X    = -1.25                      # horizontal tail quarter-chord
TAIL_CHORD = 0.18
FIN_Z     = 0.12                       # vertical-stab mid-height
FIN_CHORD = 0.18
SURF_OFF  = (-0.05, 0.0, 0.0)          # viz panel offset behind the aero centre
GEAR = {"nose":   (0.4, 0.0, -0.3),    # tricycle gear contact points
        "main_l": (-0.1, +0.25, -0.3),
        "main_r": (-0.1, -0.25, -0.3)}

# --- control limits ----------------------------------------------------------
MAX_AIL   = np.radians(8.0)
MAX_ELEV  = np.radians(20.0)
MAX_RUD   = np.radians(20.0)
THR_MAX   = 1.0
THR_RATE  = 0.4                        # throttle slew per second (X/Z held)
AIL_TRIM  = np.radians(0.045)          # aileron trim vs. prop torque, per throttle

# Deflection-command sign per surface, chosen so the documented keys give
# the intuitive response (S = nose up, D = nose right, Q = roll left).
ELEV_SIGN = -1.0                       # +cmd = TE down = nose down → negate
RUD_SIGN  = +1.0


def build_world():
    a = Craft("plane")
    a.add(Mass("body", mass=MASS, moi=(0.4, 0.35, 0.7), transform=CG))

    # Main wing at +10° incidence: tilt the chord/normal pair in the part
    # frame so level flight already sees α = WING_INC. The centre section
    # is a plain Aerofoil; the wingtips are aileron ControlSurfaces sharing
    # the same incidence — together they make up the full wing area.
    ci, si = np.cos(WING_INC), np.sin(WING_INC)
    wing_chord_axis  = (ci, 0.0, si)
    wing_normal_axis = (-si, 0.0, ci)
    a.add(Aerofoil("wing", area=0.6, chord=WING_CHORD,
                   CL_max=1.2, CD_0=0.02, induced_k=0.05,
                   chord_axis=wing_chord_axis, normal_axis=wing_normal_axis))

    # Ailerons: wingtip sections with trailing-edge flaps, at ±span.
    for name, sy in (("ail_l", +1.0), ("ail_r", -1.0)):
        a.add(ControlSurface(name, area=0.06, chord=WING_CHORD,
                             flap_chord_fraction=0.3,
                             CL_max=1.2, CD_0=0.02, induced_k=0.06,
                             chord_axis=wing_chord_axis,
                             normal_axis=wing_normal_axis,
                             stall_torque=5.0, hinge_damping=0.3,
                             servo_gain=40.0,
                             transform=(0.0, sy * AIL_SPAN, 0.0)))

    # Horizontal tail = elevator: the whole stabiliser is one control
    # surface (stab + elevator combined).
    a.add(ControlSurface("elev", area=0.14, chord=TAIL_CHORD,
                         flap_chord_fraction=0.4,
                         CL_max=1.2, CD_0=0.01, induced_k=0.04,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 0.0, 1.0),
                         stall_torque=5.0, hinge_damping=0.3, servo_gain=40.0,
                         transform=(TAIL_X, 0.0, 0.0)))

    # Vertical fin = rudder: stood upright so lift acts along body y.
    a.add(ControlSurface("rud", area=0.09, chord=FIN_CHORD,
                         flap_chord_fraction=0.4,
                         CL_max=1.2, CD_0=0.01, induced_k=0.04,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 1.0, 0.0),
                         stall_torque=5.0, hinge_damping=0.3, servo_gain=40.0,
                         transform=(TAIL_X, 0.0, FIN_Z)))

    # Fuselage drag (at CG height, so it adds no pitch moment) + propeller:
    # thrust through the CG, with a small reaction torque about the axis.
    a.add(DragSurface.isotropic_quadratic("fuselage", area=0.02,
                                          drag_coefficient=0.4,
                                          transform=(0.0, 0.0, CG[2])))
    a.add(Thruster("prop", force=(9.0, 0.0, 0.0), torque=(-0.04, 0.0, 0.0),
                   transform=(0.45, 0.0, CG[2])))

    # Tricycle landing gear: frictionless point contacts (free-rolling
    # wheels) on the ground plane.
    for name, pos in GEAR.items():
        a.add(Collider(name, stiffness=2000.0, damping=60.0, transform=pos))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=1.225))
         .add_field(CollisionField().add_half_space(origin=(0.0, 0.0, 0.0),
                                                    normal=(0.0, 0.0, 1.0))))
    w.add_craft(a, position=(0.0, 0.0, 0.30))    # parked on the runway
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
    duration = args.duration or (1e9 if args.keyboard else 36.0)

    sim = TargetNumpy(Sim(build_world()))

    # Scripted flight: full throttle down the runway (the wing incidence
    # lifts it off by itself), climb, bank into a right turn, level the
    # wings, then throttle off and glide back down to a landing.
    script = [(0.5, 3.0, {"x"}),       # full throttle → takeoff roll + climb
              (9.0, 10.6, {"z"}),      # back to cruise power
              (12.5, 12.9, {"e"}),     # roll right into a bank
              (15.5, 15.8, {"q"}),     # roll back wings-level
              (18.0, 21.0, {"z"})]     # throttle to zero → glide in
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  S/W pitch up/down   A/D yaw   Q/E roll   "
              "X/Z throttle   (Ctrl-C to quit)")
        if isinstance(ctrl, TerminalController):
            print("Type into THIS terminal; watch the viewer.\n")

    FIXED, MOVING = (120, 150, 210), (240, 150, 60)
    viz = None if args.no_viz else Viz("manta/airplane", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=300.0, color=(70, 110, 70, 160))
        viz.box("world/runway", (60.0, 3.0, 0.002), center=(55.0, 0.0, 0.0),
                color=(120, 120, 125))
        for pos in GEAR.values():
            viz.point(f"world/plane/gear/{pos[0]:.2f}_{pos[1]:.2f}", pos,
                      color=(40, 40, 45), radius=0.05)
        viz.box("world/plane/fus", (0.95, 0.035, 0.035), center=(-0.45, 0, 0),
                color=(150, 150, 158))
        viz.box("world/plane/wing/panel", (0.15, 1.2, 0.008),
                center=(-0.075, 0, 0), color=FIXED)
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
    hinges = {"ail_l": ((0.0, +AIL_SPAN, 0.0), (0, 1, 0)),
              "ail_r": ((0.0, -AIL_SPAN, 0.0), (0, 1, 0)),
              "elev":  ((TAIL_X, 0.0, 0.0), (0, 1, 0)),
              "rud":   ((TAIL_X, 0.0, FIN_Z), (0, 0, 1))}

    throttle = 0.0                     # parked, engine idle
    airborne = False
    hinge_logged: dict[str, float] = {}    # last viz-logged hinge angles
    thr_logged = -1.0                      # last viz-logged throttle
    # Live (keyboard or viewer attached) → hold the loop to real time;
    # uncapped the sim runs ~3× wall clock, which is unflyable and floods
    # the viewer's ingest channel. Headless runs stay full speed.
    pacer = Pacer() if (args.keyboard or viz is not None) else None
    # Fixed-timestep accumulator: physics always integrates at `dt` (the
    # gear contact and hinge servos are stiff — varying dt would
    # destabilize them), while input + viz run once per FRAME and the
    # pacer holds frame boundaries to the wall clock. If the machine
    # can't keep up, the sim slows down gracefully instead of exploding.
    FRAME = 0.02                       # input/viz frame (50 Hz)
    substeps = round(FRAME / dt)
    n_frames = int(duration / FRAME)
    t = 0.0
    print(f"{'t (s)':>6} {'x (m)':>8} {'alt (m)':>8} {'|v| (m/s)':>9} "
          f"{'pitch°':>7} {'roll°':>7} {'head°':>7} {'thr':>5}")
    try:
        for f in range(n_frames):
            if pacer is not None:
                pacer.pace(t)
            ctrl.update(t)

            # Direct key → deflection mapping (no inner loops). Hinge sign
            # convention: +angle swings the panel's trailing edge toward
            # +axis×chord, so +elev = TE up = nose up, +rud = nose right,
            # +ail = that wing down.
            throttle = float(np.clip(
                throttle + (ctrl.held("x") - ctrl.held("z")) * THR_RATE
                * FRAME, 0.0, THR_MAX))
            # A whisker of aileron trim cancels the propeller's torque roll
            # (exactly what a real plane's trim tab is set for).
            ail = MAX_AIL * (ctrl.held("q") - ctrl.held("e")) \
                + AIL_TRIM * throttle
            targets = {
                "elev": ELEV_SIGN * MAX_ELEV * (ctrl.held("s") - ctrl.held("w")),
                "rud":  RUD_SIGN  * MAX_RUD  * (ctrl.held("d") - ctrl.held("a")),
                "ail_l": +ail, "ail_r": -ail,
            }

            # Physics substeps: deflection commands are held over the frame;
            # each surface's own one-state servo drives it there (and lets
            # it blow back if the air loads exceed the servo's stall torque).
            for _ in range(substeps):
                st = sim.state["plane"]    # re-fetch: step() swaps the dict
                st["prop.throttle"] = throttle
                for name, target in targets.items():
                    st[f"{name}.deflection_cmd"] = target
                sim.step(dt)
                t += dt

            st = sim.state["plane"]
            p = np.asarray(st["position"]).ravel()
            if viz is not None and viz.due(t):    # throttle to ~30 Hz
                q = np.asarray(st["orientation"]).ravel()
                viz.t(t)
                viz.pose("world/plane", p, q)
                # The software-rendered WSLg viewer ingests slowly and the
                # SDK queue *blocks* rather than drops, so latency compounds
                # — only log what actually changed. Surfaces and throttle
                # sit still most of the flight.
                for name, (pos, axis) in hinges.items():
                    ang = float(st[f"{name}.deflection"])
                    if abs(ang - hinge_logged.get(name, 1e9)) > 0.002:
                        hinge_logged[name] = ang
                        viz.pose(f"world/plane/{name}", pos,
                                 _quat(axis, ang))
                if abs(throttle - thr_logged) > 0.005:
                    thr_logged = throttle
                    viz.arrow("world/plane/thrust", (0.5, 0, CG[2]),
                              (0.6 * throttle, 0, 0), color=(235, 80, 80),
                              radius=0.02)
                # Chase cam: 7 m behind the plane along its heading (yaw
                # only — no roll/pitch, so the horizon stays level), 2 m up.
                yaw = _euler(q)[0]
                eye = p + np.array(
                    [-7.0 * np.cos(yaw), -7.0 * np.sin(yaw), 2.0])
                viz.chase("world/chase", eye, p)
                if f == 0:
                    viz.track("world/chase")   # after the camera exists
                    # Wing incidence pose: logged here (not at init) so
                    # it exists on the sim timeline.
                    viz.pose("world/plane/wing", (0, 0, 0),
                             _quat((0, 1, 0), -WING_INC))
                # Sparse + capped: most calls are no-ops (min_dist), and
                # the re-sent polyline stays small. World-frame points, so
                # NOT a child of the posed plane.
                viz.trail("world/trail", p, max_len=300, min_dist=2.0)

            # Wheels-off / touchdown reporting (gear sits at z≈0.30 at rest).
            if not airborne and p[2] > 0.6:
                airborne = True
                print(f"      -- wheels off at t = {t:.2f} s, "
                      f"ground roll {p[0]:.1f} m")
            elif airborne and p[2] < 0.35:
                airborne = False
                v = np.asarray(st["velocity"]).ravel()
                print(f"      -- touchdown at t = {t:.2f} s, "
                      f"{np.linalg.norm(v):.1f} m/s, sink {-v[2]:.1f} m/s")
            if (f + 1) % round(1.0 / FRAME) == 0:
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
