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

# --- Cessna 172 Skyhawk (published 172-class figures; m, kg, x fwd / y
# left / z up). Origin at the wing aerodynamic centre (quarter-chord, ≈ the
# CG longitudinally). The high wing sits ~2.4 m off the ground on the cabin
# roof, so the CG hangs ~1.2 m below it — that pendulum, with the tail/fin,
# is the airframe's stability. Stations below are scaled from the 172
# three-view; tell me the real numbers and I'll drop them straight in. -----
MASS       = 1043.0                    # typical loaded mass (MTOW ~1111 kg)
MOI        = (1285.0, 1825.0, 2667.0)  # Ixx, Iyy, Izz (kg·m²)
# Longitudinal stations (m, +fwd) from the wing quarter-chord:
PROP_X     = 2.30                      # propeller disc (nose)
WING_LE    = 0.37                      # wing leading edge (chord/4 ahead of AC)
WING_TE    = -1.10                     # wing trailing edge (hinge line of ailerons)
TAIL_X     = -4.55                     # tail aero centre (tail arm 4.55 m)
NOSE_GEAR_X = 1.45
MAIN_GEAR_X = -0.20                    # mains just aft of the CG (lets it rotate)
# Vertical stations (m, +up) from the wing:
CG_Z       = -1.20                     # CG below the high wing
THRUST_Z   = -1.05                     # propeller thrust line (engine height)
HSTAB_Z    = -0.70                     # horizontal tail height
FIN_BASE_Z = -0.70                     # fin root (= tailplane)
FIN_TOP_Z  = 0.35                      # fin tip (~2.75 m above ground)
FIN_Z      = 0.5 * (FIN_BASE_Z + FIN_TOP_Z)   # fin / rudder aero centre
GEAR_Z     = -2.40                     # wheel contacts (wing 2.4 m up, parked)
START_Z    = -GEAR_Z                   # parked so the wheels rest on z=0
CG         = (0.0, 0.0, CG_Z)
# Wing + surface planform:
WING_AREA  = 16.2                      # m² total planform
WING_SPAN  = 11.0                      # m
WING_CHORD = 1.47                      # mean chord (= area / span)
WING_INC   = np.radians(1.5)           # wing rigging incidence
AIL_AREA   = 1.6                       # each outboard aileron section
AIL_Y      = 4.3                       # aileron section span station
AIL_SPAN   = 2.0                       # aileron section span (viz)
HTAIL_AREA = 3.0                       # horizontal stabiliser + elevator
HTAIL_CHORD = 0.9
HTAIL_SPAN = 3.4
VTAIL_AREA = 1.1                       # vertical fin + rudder
VTAIL_CHORD = 1.0
GEAR_TRACK = 1.25                      # half the main-gear track
GEAR = {"nose":   (NOSE_GEAR_X, 0.0, GEAR_Z),    # tricycle gear contacts
        "main_l": (MAIN_GEAR_X, +GEAR_TRACK, GEAR_Z),
        "main_r": (MAIN_GEAR_X, -GEAR_TRACK, GEAR_Z)}

# --- propulsion (160 hp Lycoming O-320, fixed-pitch prop) + control throws --
THRUST     = 2600.0                    # N, static (sized for a brisk demo
                                       # climb; a stock 172 is ~1300 N)
PROP_TORQUE = -410.0                   # N·m reaction roll at full power
                                       # (160 hp / ~2700 rpm ≈ 420 N·m); rolls
                                       # left, held off with right-aileron trim
MAX_AIL    = np.radians(8.0)
MAX_ELEV   = np.radians(14.0)
MAX_RUD    = np.radians(22.0)
THR_MAX    = 1.0
THR_RATE   = 0.4                       # throttle slew per second (X/Z held)
AIL_TRIM   = np.radians(0.42)          # right-aileron trim vs. prop torque
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
                         **SERVO, transform=(TAIL_X, 0.0, HSTAB_Z)))

    # Vertical fin = rudder (whole fin), stood upright so lift acts along y.
    a.add(ControlSurface("rud", area=VTAIL_AREA, chord=VTAIL_CHORD,
                         flap_chord_fraction=0.4,
                         CL_max=t.CL_max, CD_0=t.CD_0, induced_k=0.08,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 1.0, 0.0),
                         **SERVO, transform=(TAIL_X, 0.0, FIN_Z)))

    # Fuselage parasite drag (at CG height, so no pitch moment) + propeller:
    # thrust along the centreline at the nose, with the engine's reaction
    # roll torque about the thrust axis.
    a.add(DragSurface.isotropic_quadratic("fuselage", area=0.7,
                                          drag_coefficient=0.4,
                                          transform=(0.0, 0.0, CG_Z)))
    a.add(Thruster("prop", force=(THRUST, 0.0, 0.0),
                   torque=(PROP_TORQUE, 0.0, 0.0),
                   transform=(PROP_X, 0.0, THRUST_Z)))

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

    FIXED, MOVING, METAL = (120, 150, 210), (240, 150, 60), (150, 150, 158)
    # Surface chords: the moving control is the rear fraction; the fixed
    # stabiliser/fin/wing is what's ahead of the hinge line.
    AIL_FLAP_C  = 0.30 * WING_CHORD
    ELEV_FLAP_C = 0.40 * HTAIL_CHORD
    RUD_FLAP_C  = 0.40 * VTAIL_CHORD
    HSTAB_FIX_C = HTAIL_CHORD - ELEV_FLAP_C
    FIN_FIX_C   = VTAIL_CHORD - RUD_FLAP_C
    # Hinge-line x of each control (rear edge of its fixed surface).
    AIL_HINGE_X  = WING_TE                      # wing trailing edge
    ELEV_HINGE_X = TAIL_X + 0.25 * HTAIL_CHORD - HSTAB_FIX_C
    RUD_HINGE_X  = TAIL_X + 0.25 * VTAIL_CHORD - FIN_FIX_C
    viz = None if args.no_viz else Viz("manta/airplane", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=2000.0, color=(70, 110, 70, 160))
        viz.box("world/runway", (1000.0, 30.0, 0.01), center=(450.0, 0.0, 0.0),
                color=(120, 120, 125))
        # --- fixed airframe -------------------------------------------------
        # Fuselage: nose (behind the prop) back to the tail cone, at CG height.
        fus_nose, fus_tail = PROP_X - 0.1, TAIL_X - 0.45
        viz.box("world/plane/fus", (fus_nose - fus_tail, 0.55, 0.72),
                center=(0.5 * (fus_nose + fus_tail), 0, CG_Z), color=METAL)
        # High wing across the top (posed with incidence at f == 0).
        viz.box("world/plane/wing/panel", (WING_CHORD, WING_SPAN, 0.13),
                center=(WING_LE - 0.5 * WING_CHORD, 0, 0), color=FIXED)
        # Fixed horizontal stabiliser + vertical fin (the elevator/rudder
        # are the moving flaps behind them).
        viz.box("world/plane/hstab",
                (HSTAB_FIX_C, HTAIL_SPAN, 0.08),
                center=(ELEV_HINGE_X + 0.5 * HSTAB_FIX_C, 0, HSTAB_Z),
                color=FIXED)
        viz.box("world/plane/fin",
                (FIN_FIX_C, 0.08, FIN_TOP_Z - FIN_BASE_Z),
                center=(RUD_HINGE_X + 0.5 * FIN_FIX_C, 0, FIN_Z), color=FIXED)
        # Landing gear: a strut down to each wheel.
        for nm, (gx, gy, gz) in GEAR.items():
            viz.box(f"world/plane/strut_{nm}",
                    (0.09, 0.09, (CG_Z - 0.3) - gz),
                    center=(gx, gy, 0.5 * ((CG_Z - 0.3) + gz)), color=(60, 60, 65))
            viz.point(f"world/plane/wheel_{nm}", (gx, gy, gz),
                      color=(30, 30, 35), radius=0.3)
        # Propeller disc at the nose.
        viz.box("world/plane/prop", (0.05, 0.12, 1.9),
                center=(PROP_X, 0, THRUST_Z), color=(45, 45, 50))
        # --- moving control flaps (posed by deflection below) --------------
        viz.box("world/plane/ail_l/s", (AIL_FLAP_C, AIL_SPAN, 0.04),
                center=(-0.5 * AIL_FLAP_C, 0, 0), color=MOVING)
        viz.box("world/plane/ail_r/s", (AIL_FLAP_C, AIL_SPAN, 0.04),
                center=(-0.5 * AIL_FLAP_C, 0, 0), color=MOVING)
        viz.box("world/plane/elev/s", (ELEV_FLAP_C, HTAIL_SPAN, 0.05),
                center=(-0.5 * ELEV_FLAP_C, 0, 0), color=MOVING)
        viz.box("world/plane/rud/s", (RUD_FLAP_C, 0.05, FIN_TOP_Z - FIN_BASE_Z),
                center=(-0.5 * RUD_FLAP_C, 0, 0), color=MOVING)

    # Control → (hinge-line position, hinge axis) for the per-tick flap pose.
    # The flap box (above) is drawn just behind the hinge and swung by δ.
    hinges = {"ail_l": ((AIL_HINGE_X, +AIL_Y, 0.0), (0, 1, 0)),
              "ail_r": ((AIL_HINGE_X, -AIL_Y, 0.0), (0, 1, 0)),
              "elev":  ((ELEV_HINGE_X, 0.0, HSTAB_Z), (0, 1, 0)),
              "rud":   ((RUD_HINGE_X, 0.0, FIN_Z), (0, 0, 1))}

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
                    viz.arrow("world/plane/thrust", (PROP_X, 0, THRUST_Z),
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
