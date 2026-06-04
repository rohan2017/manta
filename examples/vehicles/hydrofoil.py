"""Hydrofoil — climbs out of a waving Earth ocean onto its foils.

The world is a real-scale `Earth` (point-mass gravity, ocean +
atmosphere) centred at ``(0, 0, −R_EQ)`` so sea level is the world
``z ≈ 0`` plane, with `SeaWaves` riding the boundary: the water/air
interface is a moving sinusoid carrying deep-water orbital velocity, so
the hull genuinely bobs at rest and the foils feel the moving water.

The hull is nine `PointBuoy` + nine `DragSurface` elements on the
corners and long-edge midpoints of an inverted triangular prism. Each
samples the fluid at its own point, so buoyancy and hull drag fade
element-by-element as the boat rises — `Earth(surface_smoothing=…)`
turns the water/air switch into a smooth band, which both stabilizes
the floating draft and gives the foils a lift-vs-ride-height slope
(the surface-piercing self-regulation of a real foiler).

Foils: a front V-pair (dihedral) on a strut under the CoM carries most
of the lift; a rear split foil rides two `Joint` hinges as **elevons**
— a pitch PID + a roll PID mix onto them. A large rear vertical
stabilizer weathervanes yaw; yaw COMMANDS vector the stern thruster,
which hangs on a yaw `Joint`. A two-axis laser gimbal (nested pan/tilt
`Joint`s, one PID each) rides the foredeck, keeping its laser locked on
a fixed buoy through the whole run — takeoff, waves, and carves.

Controls:  X/Z throttle up/down   A/D vector thrust (yaw left/right)

Run::

    .venv/bin/python -m examples.vehicles.hydrofoil --keyboard
    .venv/bin/python -m examples.vehicles.hydrofoil            # scripted
    .venv/bin/python -m examples.vehicles.hydrofoil --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import PID, Craft, Sim, TargetNumpy, World
from manta.parts import DragSurface, Joint, Mass, Naca00xx, PointBuoy, Thruster
from manta.planets import Earth, SeaWaves

from .._control import Pacer, common_args, make_controller
from .._viz import Viz

# --- sea state --------------------------------------------------------------
WAVES   = SeaWaves(amplitude=0.12, wavelength=10.0)
SMOOTH  = 0.25                       # water/air blend band (m)

# --- hull (m, kg; x forward, y left, z up; origin at the waterline) ---------
MASS    = 60.0
STATIONS = (1.2, 0.0, -1.2)          # bow / midship / stern triangles
DECK_Y, DECK_Z, KEEL_Z = 0.5, 0.05, -0.30
BUOY_V  = 0.0093                     # m³ per element (deck edges sit awash)

# --- foils ------------------------------------------------------------------
FRONT_X, FRONT_Z = 0.0, -0.50        # V-foil: directly under the CoM
FOIL_Y  = 0.40                       # V panels out on the rising arms
ELEV_Y  = 0.35                       # elevon halves: wide roll lever
DIHEDRAL = np.radians(20.0)
FRONT_INC = np.radians(10.0)
REAR_X, REAR_Z = -1.3, -0.95         # elevon foil: DEEPER than the front
                                     # so it never ventilates first
REAR_INC = np.radians(4.0)

# --- control ----------------------------------------------------------------
THRUST    = 350.0                    # N at full throttle
THR_RATE  = 0.3                      # throttle slew per second (X/Z held)
MAX_VEC   = np.radians(14.0)         # thrust-vector deflection
BANK_GAIN = 1.9                      # coordinated-turn roll per vector rad
VEC_RATE  = np.radians(20.0)         # vector slew, rad/s
MAX_ELEV  = np.radians(22.0)         # elevon deflection clamp
SERVO_KP  = 8.0                      # hinge servo torque per rad of error

# --- laser gimbal -------------------------------------------------------------
GIMBAL    = np.array([0.8, 0.0, 0.45])   # base on the foredeck
TARGET    = np.array([130.0, 20.0, 4.0]) # the buoy the laser tracks
LASER_LEN = 25.0


def _tilt(incidence: float, dihedral: float):
    """(chord_axis, normal_axis) for a foil with `incidence` about y and
    `dihedral` roll about x (positive = lift tilted toward −y)."""
    ci, si = np.cos(incidence), np.sin(incidence)
    chord = np.array([ci, 0.0, si])
    normal = np.array([-si, 0.0, ci])
    cd, sd = np.cos(dihedral), np.sin(dihedral)
    R_x = np.array([[1, 0, 0], [0, cd, -sd], [0, sd, cd]])
    return tuple(chord), tuple(R_x @ normal)


def _hinge(name: str, pos: tuple, axis: tuple) -> Joint:
    j = Joint(name, mode="saturating", stall_torque=40.0, damping=0.8,
              axis=axis, transform=pos)
    j.add(Mass(f"{name}_m", mass=0.15, moi=(0.004, 0.004, 0.004),
               transform=(-0.03, 0.0, 0.0)))
    return j


def build_world():
    b = Craft("boat")
    b.add(Mass("hull", mass=MASS, moi=(15.0, 40.0, 45.0),
               transform=(0.0, 0.0, -0.2)))

    # Hull: 9 buoy + 9 drag elements on the triangular-prism edges.
    for x in STATIONS:
        for tag, (y, z) in (("pl", (+DECK_Y, DECK_Z)),
                            ("pr", (-DECK_Y, DECK_Z)),
                            ("kl", (0.0, KEEL_Z))):
            b.add(PointBuoy(f"b_{tag}{x:+.0f}", volume=BUOY_V,
                            transform=(x, y, z)))
            b.add(DragSurface.isotropic_quadratic(
                f"d_{tag}{x:+.0f}", area=0.006, drag_coefficient=0.6,
                transform=(x, y, z)))

    # Superstructure air drag — the speed-dependent term that gives the
    # foiling regime a thrust/drag equilibrium (a submerged foil's drag
    # is ~weight/(L/D), nearly speed-independent: faster just means it
    # rides higher on a smaller wetted patch). Mounted above the splash
    # band so it only ever sees air.
    b.add(DragSurface.isotropic_quadratic("air", area=0.8,
                                          drag_coefficient=0.9,
                                          transform=(0.0, 0.0, 0.5)))

    # Front V-foil: two panels with dihedral on a small vertical strut.
    # Each panel's lift leans toward the centreline (a true V): sideslip
    # then produces a restoring roll moment instead of a divergent one.
    for tag, sy in (("l", +1.0), ("r", -1.0)):
        chord, normal = _tilt(FRONT_INC, sy * DIHEDRAL)
        b.add(Naca00xx(f"foil_{tag}", area=0.035, CL_max=1.1, CD_0=0.01,
                       induced_k=0.1, chord_axis=chord, normal_axis=normal,
                       transform=(FRONT_X, sy * FOIL_Y, FRONT_Z)))
    # Strut sample points sit LOW so they stay wetted at foiling ride
    # height — they are the only yaw stiffness once the hull is dry.
    b.add(Naca00xx("strut_f", area=0.025, CL_max=1.1, CD_0=0.01,
                   induced_k=0.1, chord_axis=(1, 0, 0), normal_axis=(0, 1, 0),
                   transform=(FRONT_X, 0.0, -0.55)))

    # Rear split foil: each half is an elevon on a spanwise hinge.
    chord, normal = _tilt(REAR_INC, 0.0)
    for tag, sy in (("l", +1.0), ("r", -1.0)):
        h = _hinge(f"elev_{tag}", (REAR_X, sy * ELEV_Y, REAR_Z), (0, 1, 0))
        h.add(Naca00xx(f"elev_{tag}_s", area=0.012, CL_max=1.1, CD_0=0.01,
                       induced_k=0.1, chord_axis=chord, normal_axis=normal,
                       transform=(-0.03, 0.0, 0.0)))
        b.add(h)
    # Rear vertical stabilizer: much bigger than the front strut.
    b.add(Naca00xx("strut_r", area=0.08, CL_max=1.1, CD_0=0.01,
                   induced_k=0.1, chord_axis=(1, 0, 0), normal_axis=(0, 1, 0),
                   transform=(REAR_X, 0.0, -0.70)))

    # Vectoring thruster: a yaw Joint at the stern carries the prop.
    tv = _hinge("tvec", (-1.45, 0.0, -0.5), (0, 0, 1))
    tv.add(Thruster("prop", force=(THRUST, 0.0, 0.0)))
    b.add(tv)

    # Two-axis laser gimbal on the foredeck: a pan Joint (yaw) hosting a
    # tilt Joint (pitch); a PID per axis keeps the laser on the buoy no
    # matter what the boat does.
    pan = Joint("pan", mode="saturating", stall_torque=8.0, damping=0.3,
                axis=(0.0, 0.0, 1.0), transform=tuple(GIMBAL))
    pan.add(Mass("pan_motor", mass=0.3, moi=(0.002, 0.002, 0.003)))
    tilt = Joint("tilt", mode="saturating", stall_torque=6.0, damping=0.3,
                 axis=(0.0, 1.0, 0.0), transform=(0.0, 0.0, 0.12))
    tilt.add(Mass("laser", mass=0.2, moi=(0.001, 0.001, 0.001)))
    pan.add(tilt)
    b.add(pan)

    earth = Earth(position=(0.0, 0.0, -Earth.R_EQ), waves=WAVES,
                  surface_smoothing=SMOOTH)
    w = World().add_planet(earth)
    w.add_craft(b, position=(0.0, 0.0, 0.02))
    return w


def _quat(axis, angle):
    h = 0.5 * angle
    return (np.cos(h), *(np.sin(h) * np.asarray(axis, dtype=float)))


def _R(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _euler(q_wxyz):
    """ZYX yaw/pitch/roll (rad); pitch + = nose down (about +y = left)."""
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    return yaw, pitch, roll


def _eta(x, y, t):
    """Wave elevation at (x, y) — same planar sinusoid the Earth uses."""
    k = 2 * np.pi / WAVES.wavelength
    g0 = Earth.MU / Earth.R_EQ**2
    omega = k * np.sqrt(g0 * WAVES.wavelength / (2 * np.pi))
    return WAVES.amplitude * np.cos(k * x - omega * t)


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.002
    duration = args.duration or (1e9 if args.keyboard else 32.0)

    sim = TargetNumpy(Sim(build_world()))

    # Scripted: bob for a while, throttle up onto the foils, carve a
    # right then a left turn at speed.
    script = [(5.0, 9.0, {"x"}),       # ramp throttle → climb onto foils
              (10.0, 12.0, {"z"}),     # settle to cruise power
              (15.0, 18.0, {"d"}),     # carve right
              (21.0, 24.0, {"a"}),     # carve left
              (25.0, 26.5, {"x"})]     # power out of the carve
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  X/Z throttle   A/D vector thrust   "
              "(Ctrl-C to quit)")

    FIXED, MOVING, HULL = (120, 150, 210), (240, 150, 60), (200, 190, 160)
    viz = None if args.no_viz else Viz("manta/hydrofoil", addr=args.viz_addr)
    if viz is not None:
        # Static geometry only here — local POSES are logged on the first
        # frame (after viz.t), or they never exist on the sim timeline.
        # The sea is a disc that follows the boat (posed per frame), so
        # you can never drive off its edge.
        viz.disc("world/sea/disc", 60.0, color=(15, 35, 60, 255))
        # Hull: rectangular prism above the waterline, V-keel triangular
        # prism below (two slanted panels meeting at the keel line, base
        # flush with the box bottom). The buoy/drag points sit exactly on
        # the prism's corner + midpoint edges.
        viz.box("world/boat/deck", (1.5, DECK_Y, 0.15),
                center=(0, 0, DECK_Z + 0.15), color=HULL)
        half_w = 0.5 * np.hypot(DECK_Y, DECK_Z - KEEL_Z)
        viz.box("world/boat/side_l/s", (1.5, half_w, 0.015), color=HULL)
        viz.box("world/boat/side_r/s", (1.5, half_w, 0.015), color=HULL)
        # Struts + front V panels (fixed), elevons + thruster (moving).
        viz.box("world/boat/strut_f", (0.04, 0.005, 0.20),
                center=(FRONT_X, 0, -0.45), color=FIXED)
        viz.box("world/boat/strut_r", (0.06, 0.006, 0.23),
                center=(REAR_X, 0, -0.52), color=FIXED)
        for tag in ("l", "r"):
            viz.box(f"world/boat/foil_{tag}/s", (0.05, 0.16, 0.004),
                    color=FIXED)
            viz.box(f"world/boat/elev_{tag}/s", (0.04, 0.14, 0.003),
                    center=(-0.03, 0, 0), color=MOVING)
        viz.box("world/boat/pan/tilt/laser_body", (0.10, 0.04, 0.04),
                color=(60, 60, 70))

    # One PID per stabilized axis (mixed onto the elevons) + one per
    # gimbal joint (commanding joint torque directly).
    pid = {ax: TargetNumpy(PID(kp=kp, ki=ki, kd=kd, integral_limit=lim))
           for ax, (kp, ki, kd, lim) in
           {"pitch": (2.5, 0.5, 0.6, 0.2), "roll": (2.2, 0.6, 0.35, 0.2),
            "pan": (6.0, 8.0, 0.5, 2.0), "tilt": (6.0, 8.0, 0.5, 2.0)
            }.items()}

    throttle, vec = 0.0, 0.0
    FRAME = 0.02
    substeps = round(FRAME / dt)
    pacer = Pacer() if (args.keyboard or viz is not None) else None
    t = 0.0
    print(f"{'t (s)':>6} {'x (m)':>7} {'v (m/s)':>8} {'ride (m)':>8} "
          f"{'pitch°':>7} {'roll°':>7} {'head°':>7} {'thr':>5} {'aim°':>6}")
    try:
        for f in range(int(duration / FRAME)):
            if pacer is not None:
                pacer.pace(t)
            ctrl.update(t)

            throttle = float(np.clip(
                throttle + (ctrl.held("x") - ctrl.held("z")) * THR_RATE
                * FRAME, 0.0, 1.0))
            # Slew the vector command (a step at speed kicks the bank
            # loop too hard).
            vec_cmd = MAX_VEC * (ctrl.held("d") - ctrl.held("a"))
            vec = float(np.clip(vec_cmd, vec - VEC_RATE * FRAME,
                                vec + VEC_RATE * FRAME))

            st = sim.state["boat"]
            yaw, pitch, roll = _euler(np.asarray(st["orientation"]).ravel())
            v_now = float(np.linalg.norm(np.asarray(st["velocity"]).ravel()))
            # Elevon authority grows with q ∝ v²: schedule the PID
            # output back down so the loops stay tuned at speed.
            gain = min(1.5, max(0.45, (8.0 / max(v_now, 4.0)) ** 2))
            # Elevon mixing: PIDs hold the boat level. +deflection = TE
            # up = local lift down (= stern down = nose up).
            cmd_p = gain * pid["pitch"].step(FRAME, setpoint=0.0,
                                             measurement=pitch)["command"]
            # Coordinated turn: a yaw command also banks the boat into
            # the turn (a flat skid at speed trips the foils sideways).
            cmd_r = gain * pid["roll"].step(FRAME,
                                            setpoint=BANK_GAIN * vec,
                                            measurement=roll)["command"]
            targets = {
                "elev_l": float(np.clip(-cmd_p - cmd_r,
                                        -MAX_ELEV, MAX_ELEV)),
                "elev_r": float(np.clip(-cmd_p + cmd_r,
                                        -MAX_ELEV, MAX_ELEV)),
                "tvec": vec,
            }

            # Laser gimbal: desired pan/tilt from the buoy bearing in the
            # hull frame; each PID outputs a joint torque directly.
            p = np.asarray(st["position"]).ravel()
            q = np.asarray(st["orientation"]).ravel()
            R = _R(q)
            to_tgt = R.T @ (TARGET - (p + R @ GIMBAL))
            des = {"pan": np.arctan2(to_tgt[1], to_tgt[0]),
                   "tilt": -np.arctan2(to_tgt[2], np.hypot(*to_tgt[:2]))}

            for _ in range(substeps):
                st = sim.state["boat"]    # re-fetch: step() swaps the dict
                st["prop.throttle"] = throttle
                for name, target in targets.items():
                    st[f"{name}.torque_cmd"] = \
                        SERVO_KP * (target - float(st[f"{name}.angle"]))
                # Gimbal PIDs run at the physics rate: the bearing slews
                # ~1 rad/s at speed, too fast for frame-rate stepping.
                for j, d in des.items():
                    ang = float(st[f"{j}.angle"])
                    st[f"{j}.torque_cmd"] = pid[j].step(
                        dt, setpoint=ang + _wrap(d - ang),
                        measurement=ang)["command"]
                sim.step(dt)
                t += dt

            st = sim.state["boat"]
            p = np.asarray(st["position"]).ravel()
            if viz is not None and f % 2 == 0:
                q = np.asarray(st["orientation"]).ravel()
                viz.t(t)
                if f == 0:
                    # Static local poses, logged once ON the sim timeline.
                    slant = np.arctan2(DECK_Y, DECK_Z - KEEL_Z)
                    for tag, sy in (("l", +1.0), ("r", -1.0)):
                        viz.pose(f"world/boat/side_{tag}",
                                 (0, sy * DECK_Y / 2, (DECK_Z + KEEL_Z) / 2),
                                 _quat((1, 0, 0), sy * (np.pi / 2 - slant)))
                        viz.pose(f"world/boat/foil_{tag}",
                                 (FRONT_X, sy * FOIL_Y, FRONT_Z),
                                 _quat((1, 0, 0), sy * DIHEDRAL))
                    viz.point("world/target", TARGET,
                              color=(90, 230, 120), radius=0.5)
                    # Laser ray along +x of the tilt frame.
                    viz.arrow("world/boat/pan/tilt/laser", (0, 0, 0),
                              (LASER_LEN, 0, 0), color=(255, 60, 60),
                              radius=0.02)
                viz.pose("world/boat", p, q)
                for name in ("elev_l", "elev_r"):
                    sy = +1.0 if name.endswith("l") else -1.0
                    viz.pose(f"world/boat/{name}",
                             (REAR_X, sy * ELEV_Y, REAR_Z),
                             _quat((0, 1, 0), float(st[f"{name}.angle"])))
                viz.pose("world/boat/tvec", (-1.45, 0, -0.5),
                         _quat((0, 0, 1), float(st["tvec.angle"])))
                viz.arrow("world/boat/tvec/thrust", (0, 0, 0),
                          (-0.8 * throttle, 0, 0), color=(235, 80, 80),
                          radius=0.02)
                viz.pose("world/boat/pan", tuple(GIMBAL),
                         _quat((0, 0, 1), float(st["pan.angle"])))
                viz.pose("world/boat/pan/tilt", (0, 0, 0.12),
                         _quat((0, 1, 0), float(st["tilt.angle"])))
                if f == 0:
                    # Eye-track the boat: the orbit centre follows it and
                    # the user orbits/zooms normally.
                    viz.track("world/boat")
                viz.trail("world/trail", p, max_len=300, min_dist=2.0)
                viz.pose("world/sea", (p[0], p[1], -1.8))
                if f % 10 == 0:      # 5 Hz animated wave strips, boat-centred
                    xs = p[0] + np.arange(-12.0, 12.5, 1.5)
                    for j, dy in enumerate(np.arange(-8.0, 8.5, 2.0)):
                        y = p[1] + dy
                        pts = np.column_stack(
                            [xs, np.full_like(xs, y), _eta(xs, y, t)])
                        viz.line(f"world/waves/{j}", pts,
                                 color=(90, 160, 220), radius=0.015)

            if (f + 1) % round(1.0 / FRAME) == 0:
                v = np.asarray(st["velocity"]).ravel()
                q = np.asarray(st["orientation"]).ravel()
                yaw, pitch, roll = _euler(q)
                # Laser pointing error: actual ray direction vs. bearing.
                R = _R(q)
                pan_a, tilt_a = (float(st["pan.angle"]),
                                 float(st["tilt.angle"]))
                d_world = R @ np.array(
                    [np.cos(pan_a) * np.cos(tilt_a),
                     np.sin(pan_a) * np.cos(tilt_a), -np.sin(tilt_a)])
                bearing = TARGET - (p + R @ GIMBAL)
                bearing = bearing / np.linalg.norm(bearing)
                aim = np.degrees(np.arccos(np.clip(d_world @ bearing,
                                                   -1.0, 1.0)))
                print(f"{t:>6.2f} {p[0]:>7.1f} {np.linalg.norm(v):>8.2f} "
                      f"{p[2]:>8.2f} {np.degrees(pitch):>+7.1f} "
                      f"{np.degrees(roll):>+7.1f} {np.degrees(yaw):>+7.1f} "
                      f"{throttle:>5.2f} {aim:>6.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
