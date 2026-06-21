"""Airplane — a Cessna 172 Skyhawk flown on real control surfaces.

Built from published 172-class figures: ~1043 kg, 10.97 m span, 16.2 m²
wing on a NACA 2412 section (via the `naca()` helper), a NACA 0012 tail,
and the real moments of inertia. No IMU, no stabilizer loops — the
airframe is statically stable on its own. As a high-wing aircraft the CG
hangs about a metre below the wing, and that pendulum is its real roll
stability; the horizontal tail and vertical fin weathervane it back to
trim in pitch and yaw.

Every control is a `ControlSurface` — a wing section with a deflectable
trailing-edge flap, no hinge joint: two ailerons outboard on the wing,
the elevator is the whole horizontal tail, the rudder the whole vertical
fin. The keys command a target deflection; each surface's one-state servo
(with its stall torque and aerodynamic hinge moment — so a surface can
blow back at speed) drives it there, and the combined wing+flap lift
falls out of the physics. Because the surfaces are massless flaps rather
than articulated joints, the craft stays a flat rigid body — one
deflection state per surface, not a joint angle+rate, and no per-step
articulated solve: 17 states instead of 21, and the sim step and EKF
predict run several times faster.

A nose `Thruster` plays propeller (constant thrust, with a reaction roll
torque the aileron trim cancels). Tricycle-gear `Colliders` rest it on a
`CollisionField` runway. Like a real 172, it is flown on trim: a standing
nose-up elevator trim rotates it at flying speed and holds the climb, so
the scripted flight needs only throttle and the occasional aileron.

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
    Collider, ControlSurface, DragSurface, Mass, Thruster, naca,
)

from .._control import Pacer, TerminalController, common_args, make_controller
from .._viz import Viz

# --- Cessna 172 Skyhawk (published 172N-class figures; m, kg, x fwd / y
# left / z up). Origin at the wing aerodynamic centre (quarter-chord). The
# high wing sits above the cabin, so the CG hangs ~1 m below it — that
# pendulum is the 172's real roll stability, not a modelling hack. ---------
MASS       = 1043.0                    # typical loaded mass (MTOW ~1111 kg)
MOI        = (1285.0, 1825.0, 2667.0)  # Ixx, Iyy, Izz (kg·m²)
CG_Z       = -1.0                      # CG below the high wing
CG         = (0.0, 0.0, CG_Z)
WING_AREA  = 16.2                      # m² total planform
WING_SPAN  = 10.97                     # m
WING_CHORD = 1.48                      # mean chord (= area / span)
WING_INC   = np.radians(1.5)           # wing rigging incidence
AIL_AREA   = 1.6                       # each outboard aileron section
AIL_Y      = 4.0                       # aileron section span station
TAIL_X     = -4.6                      # tail quarter-chord aft of the wing AC
HTAIL_AREA = 3.0                       # horizontal stabiliser + elevator
HTAIL_CHORD = 0.9
VTAIL_AREA = 1.1                       # vertical fin + rudder
VTAIL_CHORD = 1.0
FIN_Z      = 0.7                       # fin centre above the tail boom
SURF_OFF   = (-0.3, 0.0, 0.0)          # viz panel offset behind the aero centre
GEAR = {"nose":   (1.1, 0.0, -1.8),    # tricycle gear contact points
        "main_l": (-0.4, +1.3, -1.8),
        "main_r": (-0.4, -1.3, -1.8)}
START_Z    = 1.8                       # parked so the wheels rest on z=0

# --- propulsion (160 hp Lycoming, fixed-pitch prop, modelled as constant
# thrust) + control throws ---------------------------------------------------
THRUST     = 2600.0                    # N, static (sized for a brisk demo
                                       # climb; a stock 172 is ~1300 N)
PROP_TORQUE = -45.0                    # N·m reaction roll (countered by trim)
MAX_AIL    = np.radians(8.0)
MAX_ELEV   = np.radians(14.0)
MAX_RUD    = np.radians(22.0)
THR_MAX    = 1.0
THR_RATE   = 0.4                       # throttle slew per second (X/Z held)
AIL_TRIM   = np.radians(0.055)         # aileron trim vs. prop torque, per throttle
ELEV_TRIM  = np.radians(-4.5)          # standing nose-up elevator: rotates and
                                       # climbs hands-off at full power

# Full-size surfaces: servo authority sized so the surfaces track command
# in normal flight but can still blow back under extreme load.
SERVO = dict(stall_torque=600.0, hinge_damping=40.0, servo_gain=4000.0)

# Deflection-command sign per surface, chosen so the documented keys give
# the intuitive response (S = nose up, D = nose right, Q = roll left).
ELEV_SIGN = -1.0                       # +cmd = TE down = nose down → negate
RUD_SIGN  = +1.0


def build_world():
    a = Craft("plane")
    a.add(Mass("body", mass=MASS, moi=MOI, transform=CG))

    # Main wing — NACA 2412 (cambered), rigged at +1.5° incidence: tilt the
    # chord/normal pair so level flight already sees α = WING_INC. The
    # centre section is a plain Aerofoil; the outboard tips are aileron
    # ControlSurfaces sharing the same foil — together the full wing area.
    ci, si = np.cos(WING_INC), np.sin(WING_INC)
    wing_chord_axis  = (ci, 0.0, si)
    wing_normal_axis = (-si, 0.0, ci)
    ar = WING_SPAN**2 / WING_AREA
    wing = naca("2412", "wing", area=WING_AREA - 2 * AIL_AREA, chord=WING_CHORD,
                induced_k=1.0 / (np.pi * ar * 0.8),
                chord_axis=wing_chord_axis, normal_axis=wing_normal_axis)
    a.add(wing)

    # Ailerons: outboard 2412 sections with trailing-edge flaps, at ±span.
    foil = dict(alpha_0=wing.alpha_0, Cm_ac=wing.Cm_ac, CL_max=wing.CL_max,
                CD_0=wing.CD_0, induced_k=wing.induced_k)
    for name, sy in (("ail_l", +1.0), ("ail_r", -1.0)):
        a.add(ControlSurface(name, area=AIL_AREA, chord=WING_CHORD,
                             flap_chord_fraction=0.3, **foil,
                             chord_axis=wing_chord_axis,
                             normal_axis=wing_normal_axis, **SERVO,
                             transform=(0.0, sy * AIL_Y, 0.0)))

    # Horizontal tail = elevator (whole stabiliser), symmetric NACA 0012.
    t = naca("0012", area=1.0, chord=1.0)   # 0012 invariants (CL_max, CD_0)
    a.add(ControlSurface("elev", area=HTAIL_AREA, chord=HTAIL_CHORD,
                         flap_chord_fraction=0.4,
                         CL_max=t.CL_max, CD_0=t.CD_0, induced_k=0.06,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 0.0, 1.0),
                         **SERVO, transform=(TAIL_X, 0.0, 0.0)))

    # Vertical fin = rudder (whole fin), stood upright so lift acts along y.
    a.add(ControlSurface("rud", area=VTAIL_AREA, chord=VTAIL_CHORD,
                         flap_chord_fraction=0.4,
                         CL_max=t.CL_max, CD_0=t.CD_0, induced_k=0.08,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 1.0, 0.0),
                         **SERVO, transform=(TAIL_X, 0.0, FIN_Z)))

    # Fuselage parasite drag (at CG height, so no pitch moment) + propeller:
    # thrust along the centreline, with a reaction roll torque about the axis.
    a.add(DragSurface.isotropic_quadratic("fuselage", area=0.7,
                                          drag_coefficient=0.4,
                                          transform=(0.0, 0.0, CG_Z)))
    a.add(Thruster("prop", force=(THRUST, 0.0, 0.0),
                   torque=(PROP_TORQUE, 0.0, 0.0),
                   transform=(1.5, 0.0, CG_Z)))

    # Tricycle landing gear: frictionless point contacts (free-rolling
    # wheels) on the ground plane.
    for name, pos in GEAR.items():
        a.add(Collider(name, stiffness=1.5e5, damping=8000.0, transform=pos))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=1.225))
         .add_field(CollisionField().add_half_space(origin=(0.0, 0.0, 0.0),
                                                    normal=(0.0, 0.0, 1.0))))
    w.add_craft(a, position=(0.0, 0.0, START_Z))    # parked on the runway
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
    duration = args.duration or (1e9 if args.keyboard else 120.0)

    sim = TargetNumpy(Sim(build_world()))

    # Scripted flight, flown like a real 172: full power down the runway —
    # the elevator trim rotates it at flying speed and holds the climb —
    # then ease the power back and nose level for cruise, bank into a right
    # turn, roll level, and finally throttle off and glide down to land.
    script = [(0.5, 3.0, {"x"}),       # full throttle → takeoff roll + climb
              (38.0, 39.0, {"z"}),     # ease back to cruise power
              (44.0, 44.4, {"e"}),     # roll right into a bank
              (46.5, 46.9, {"q"}),     # roll back wings-level
              (54.0, 58.0, {"z"})]     # throttle to idle → glide down to land
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  S/W pitch up/down   A/D yaw   Q/E roll   "
              "X/Z throttle   (Ctrl-C to quit)")
        if isinstance(ctrl, TerminalController):
            print("Type into THIS terminal; watch the viewer.\n")

    FIXED, MOVING = (120, 150, 210), (240, 150, 60)
    viz = None if args.no_viz else Viz("manta/airplane", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=2000.0, color=(70, 110, 70, 160))
        viz.box("world/runway", (900.0, 25.0, 0.01), center=(420.0, 0.0, 0.0),
                color=(120, 120, 125))
        for pos in GEAR.values():
            viz.point(f"world/plane/gear/{pos[0]:.2f}_{pos[1]:.2f}", pos,
                      color=(40, 40, 45), radius=0.25)
        # Fuselage + fin post, at CG height; wing across the top (high wing).
        viz.box("world/plane/fus", (7.5, 0.5, 0.6), center=(-1.4, 0, CG_Z),
                color=(150, 150, 158))
        viz.box("world/plane/wing/panel",
                (WING_CHORD, WING_SPAN, 0.12), center=(-0.4, 0, 0), color=FIXED)
        # Moving surfaces: ailerons at the tips, elevator = whole tailplane,
        # rudder = whole fin (each posed by its deflection below).
        viz.box("world/plane/ail_l/s", (0.45, 2.0, 0.04), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/ail_r/s", (0.45, 2.0, 0.04), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/elev/s", (HTAIL_CHORD, 3.4, 0.05), center=SURF_OFF,
                color=MOVING)
        viz.box("world/plane/rud/s", (VTAIL_CHORD, 0.05, 1.4), center=SURF_OFF,
                color=MOVING)

    # Hinge name → (viz path, hinge position, hinge axis) for the per-tick
    # deflection poses.
    hinges = {"ail_l": ((0.0, +AIL_Y, 0.0), (0, 1, 0)),
              "ail_r": ((0.0, -AIL_Y, 0.0), (0, 1, 0)),
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
                "elev": ELEV_SIGN * MAX_ELEV * (ctrl.held("s") - ctrl.held("w"))
                        + ELEV_TRIM,
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
                    viz.arrow("world/plane/thrust", (1.5, 0, CG_Z),
                              (3.0 * throttle, 0, 0), color=(235, 80, 80),
                              radius=0.2)
                # Chase cam: 30 m behind the plane along its heading (yaw
                # only — no roll/pitch, so the horizon stays level), 8 m up.
                yaw = _euler(q)[0]
                eye = p + np.array(
                    [-30.0 * np.cos(yaw), -30.0 * np.sin(yaw), 8.0])
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

            # Wheels-off / touchdown reporting (origin rides at START_Z at rest).
            if not airborne and p[2] > START_Z + 0.6:
                airborne = True
                print(f"      -- wheels off at t = {t:.2f} s, "
                      f"ground roll {p[0]:.1f} m")
            elif airborne and p[2] < START_Z + 0.3:
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
