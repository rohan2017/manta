"""Airplane — a Cessna 172 Skyhawk flown on real control surfaces.

Geometry is a re-measured 172 airframe survey (coordinate origin at the
propeller plane): 11.0 m span / 16.2 m² wing on a NACA 2412 section (via the
`naca()` helper), a NACA 0012 tail 4.6 m aft, gear and surface stations to
scale, plus 1043 kg gross and the real moments of inertia. No IMU, no stabilizer
loops — the airframe is statically stable on its own: wing dihedral and the
high-wing-over-CG pendulum give roll stability, and the horizontal tail and
vertical fin weathervane it back to trim in pitch and yaw.

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
from .._geom import euler_zyx, quat
from .._viz import Viz

# --- Cessna 172 Skyhawk, from a re-measured airframe survey. Coordinate
# origin is the PROPELLER PLANE (spinner tip) on the thrust line; +x forward
# (so the airframe lies at negative x), +y left, +z up. Aerodynamic parts sit
# at their quarter-chord (where lift acts); measured c/4 stations map to them.
MASS       = 1043.0                    # kg, max gross (2300 lb); empty ~745 kg
MOI        = (1742.0, 2475.0, 3616.0)  # Ixx, Iyy, Izz (kg·m²) at gross
#                                        (Roskam 1285/1825/2667 slug·ft² ×1.356)

# CG — at gross, standard C172 reference.
CG         = (-2.45, 0.0, -0.20)

# Main wing — NACA 2412, rectangular, +1.5° rigging incidence, 1.7° dihedral.
WING_AREA  = 16.2; WING_SPAN = 11.0; WING_CHORD = 1.49
WING_INC   = np.radians(1.5)
WING_QC_X  = -2.44                      # quarter-chord (LE −2.07, TE −3.56)
WING_Z     = 0.85
DIHEDRAL   = np.radians(1.7)

# Ailerons — outboard wing sections (span 1.8 m, centred y ±3.9), each a
# ControlSurface (full-chord wing section + trailing-edge flap). The flap is
# the measured 0.46 m aileron chord.
AIL_Y      = 3.9                        # section centre span station
AIL_SPAN   = 1.8
AIL_AREA   = AIL_SPAN * WING_CHORD       # ≈ 2.68 m² outboard section
AIL_FLAP   = 0.46 / WING_CHORD           # aileron is 31% of the chord

# Horizontal tail — NACA 0012, rectangular, span 3.40, chord 0.60; the whole
# stabiliser is the elevator (flap = the rear 0.27 m → 45% of the chord).
HTAIL_QC_X = -7.05; HTAIL_Z = 0.25       # c/4 (LE −6.90, TE −7.50)
HTAIL_AREA = 2.0; HTAIL_CHORD = 0.60
ELEV_FLAP  = 0.27 / HTAIL_CHORD

# Vertical tail — NACA 0012, height ~1.2 (z 0.30–1.49), mean chord 0.95; the
# whole fin is the rudder (flap = the rear 0.50 m → 53% of the chord). The fin
# is tapered, so its MAC c/4 sits aft of the measured root c/4 (−6.60): anchor
# the rectangular surface to the real TE (−7.50) so the rudder hinge lands on
# the measured −7.00 (QC = −7.50 + 0.75·0.95).
VTAIL_QC_X = -6.79; VTAIL_Z = 0.90       # MAC c/4 (TE −7.50, hinge −7.00)
VTAIL_AREA = 1.1; VTAIL_CHORD = 0.95
RUD_FLAP   = 0.50 / VTAIL_CHORD

# Landing gear — wheel contact points (track 2.54 m, wheelbase 1.56 m).
GEAR_Z     = -1.23
GEAR = {"nose":   (-1.10, 0.0, GEAR_Z),
        "main_l": (-2.66, +1.27, GEAR_Z),
        "main_r": (-2.66, -1.27, GEAR_Z)}
START_Z    = -GEAR_Z                     # parked so the wheels rest on z=0
# Wheels roll freely fore–aft (body x) and vertically, but grip sideways
# (body y), so the airframe tracks where it points: a yaw (rudder) on the
# ground becomes a turn instead of a frictionless sideways skid.
GEAR_FRICTION = (0.0, 6000.0, 0.0)      # N·s/m per body axis (lateral grip)

# Fuselage (decorative viz only — physics is a point DragSurface at the CG):
# three measured rectangular prisms, each center=(x,y,z), size=(L,W,H).
FUSELAGE_BOXES = dict(
    nose  = dict(center=(-0.50, 0.0, -0.10), size=(1.00, 0.90, 0.95)),
    cabin = dict(center=(-2.30, 0.0,  0.10), size=(2.60, 1.05, 1.40)),
    tail  = dict(center=(-5.55, 0.0,  0.00), size=(3.90, 0.55, 0.75)),
)

# --- propulsion (160 hp Lycoming O-320, fixed-pitch prop) + control throws --
THRUST     = 1300.0                    # N, static — a stock 172 at full power
PROP_TORQUE = -410.0                   # N·m reaction roll at full power
                                       # (160 hp / ~2700 rpm ≈ 420 N·m); rolls
                                       # left, held off with right-aileron trim
# Fuselage drag coefficients, applied per box face (the face area normal to
# each body axis is that axis's reference area). Anisotropic — both physically
# correct and what the slender shape needs: LOW nose-on (a fuselage is
# streamlined fore–aft) and HIGH broadside, where the large side/top faces well
# aft of the CG give the body its weathervane stability + yaw damping, the
# authority a single point drag source at the CG could not provide. The tandem
# boxes' frontal faces are de-shadowed in build_world (a box behind a larger
# one adds no nose-on area), so axial drag ≈ the max cross-section, not the sum.
FUSELAGE_CD_AXIAL = 0.10
FUSELAGE_CD_SIDE  = 0.60
MAX_AIL    = np.radians(12.0)
MAX_ELEV   = np.radians(14.0)
MAX_RUD    = np.radians(22.0)
THR_MAX    = 1.0
THR_RATE   = 0.4                       # throttle slew per second (X/Z held)
AIL_TRIM   = np.radians(0.28)          # right-aileron trim vs. prop torque
                                       # (sized for the ~43 m/s climb speed)
ELEV_TRIM  = np.radians(-4.5)          # standing nose-up elevator: at the
                                       # reduced power it gently rotates the
                                       # plane near 44 m/s and holds the climb

# Full-size surfaces: servo authority sized so the surfaces track command
# in normal flight but can still blow back under extreme load.
SERVO = dict(stall_torque=600.0, hinge_damping=40.0, servo_gain=4000.0)

# Deflection-command sign per surface, chosen so the documented keys give
# the intuitive response (S = nose up, D = nose right, Q = roll left).
ELEV_SIGN = -1.0                       # +cmd = TE down = nose down → negate
RUD_SIGN  = +1.0


def build_world():
    a = Craft("plane")
    a.add(Mass("body", mass=MASS, moi=MOI, mount_offset=CG))

    # TODO(mount_orientation): this model predates static mount rotations
    # and encodes its INSTALLATION geometry — rigging incidence, dihedral,
    # and the fin's 90° roll — in the chord/normal axis pair. The current
    # convention is the opposite: leave chord_axis/normal_axis canonical
    # ((1,0,0) / (0,0,1)) and put installation angles in
    # `mount_orientation` (see the Aerofoil docstring for why). Converting
    # this example would replace `wing_chord_axis` and the ∓sd dihedral
    # component with one roll/pitch quaternion per panel, and make the
    # left/right mirror a sign on the roll rather than on a vector
    # component. Left as-is for now because it is live on the site as a
    # WASM bundle and the change, while behaviour-preserving, is not
    # risk-free.
    #
    # Main wing — NACA 2412 (cambered), rigged at +1.5° incidence: tilt the
    # chord/normal pair so level flight already sees α = WING_INC. The wing is
    # unswept and rectangular (chord WING_CHORD, tips at ±WING_SPAN/2), split
    # spanwise into CONTIGUOUS panels that tile the full span so each carries
    # the real dihedral and the ailerons sit at their measured station. Per
    # side, root→tip: an inboard fixed panel, the aileron ControlSurface (a
    # wing section + trailing-edge flap), and a fixed tip panel. Every panel
    # shares the wing quarter-chord x and chord-plane height; dihedral enters
    # through normal_axis (y-component ∓sd → lift tilts inboard), since a single
    # panel spanning both sides could not tilt left and right oppositely.
    ci, si = np.cos(WING_INC), np.sin(WING_INC)
    sd = np.sin(DIHEDRAL)
    wing_chord_axis = (ci, 0.0, si)
    ar = WING_SPAN**2 / WING_AREA
    wing_ik = 1.0 / (np.pi * ar * 0.8)
    tip = 0.5 * WING_SPAN                             # wingtip station (+5.50)
    ail_in, ail_out = AIL_Y - 0.5 * AIL_SPAN, AIL_Y + 0.5 * AIL_SPAN   # 3.0/4.8

    def panel(name, sy, y0, y1):
        """A fixed rectangular wing panel spanning y in [y0, y1] on side sy,
        placed at its span centroid with the wing's quarter-chord and dihedral.
        """
        return naca("2412", name, area=(y1 - y0) * WING_CHORD, chord=WING_CHORD,
                    induced_k=wing_ik, chord_axis=wing_chord_axis,
                    normal_axis=(-si, -sy * sd, ci),
                    mount_offset=(WING_QC_X, sy * 0.5 * (y0 + y1), WING_Z))

    wing = None
    for s, sy in (("l", +1.0), ("r", -1.0)):
        wing = panel(f"wing_in_{s}", sy, 0.0, ail_in)     # root → aileron
        a.add(wing)
        a.add(panel(f"wing_tip_{s}", sy, ail_out, tip))   # aileron → tip

    # Ailerons: the outboard wing section (root chord, span AIL_SPAN) with a
    # trailing-edge flap, at the wing dihedral so lift tilts inboard (sideslip
    # is self-correcting). Placed at the aileron's span centroid AIL_Y.
    foil = dict(alpha_0=wing.alpha_0, Cm_ac=wing.Cm_ac, CL_max=wing.CL_max,
                CD_0=wing.CD_0, induced_k=wing.induced_k)
    for name, sy in (("ail_l", +1.0), ("ail_r", -1.0)):
        a.add(ControlSurface(name, area=AIL_AREA, chord=WING_CHORD,
                             flap_chord_fraction=AIL_FLAP, **foil,
                             chord_axis=wing_chord_axis,
                             normal_axis=(-si, -sy * sd, ci), **SERVO,
                             mount_offset=(WING_QC_X, sy * AIL_Y, WING_Z)))

    # Horizontal tail = elevator (whole stabiliser), symmetric NACA 0012.
    t = naca("0012", area=1.0, chord=1.0)   # 0012 invariants (CL_max, CD_0)
    a.add(ControlSurface("elev", area=HTAIL_AREA, chord=HTAIL_CHORD,
                         flap_chord_fraction=ELEV_FLAP,
                         CL_max=t.CL_max, CD_0=t.CD_0, induced_k=0.06,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 0.0, 1.0),
                         **SERVO, mount_offset=(HTAIL_QC_X, 0.0, HTAIL_Z)))

    # Vertical fin = rudder (whole fin), stood upright so lift acts along y.
    a.add(ControlSurface("rud", area=VTAIL_AREA, chord=VTAIL_CHORD,
                         flap_chord_fraction=RUD_FLAP,
                         CL_max=t.CL_max, CD_0=t.CD_0, induced_k=0.08,
                         chord_axis=(1.0, 0.0, 0.0), normal_axis=(0.0, 1.0, 0.0),
                         **SERVO, mount_offset=(VTAIL_QC_X, 0.0, VTAIL_Z)))

    # Fuselage aerodynamics: one anisotropic drag box per FUSELAGE_BOXES
    # prism, mounted at the box centre. Drag along each body axis scales with
    # the area of the box face normal to it — so the slender body is low-drag
    # nose-on but has large side/top area. The aft tail box sits well behind
    # the CG, so in a sideslip its side drag weathervanes the nose into the
    # relative wind and damps yaw rate (giving the rudder a heading to hold
    # against) — a single point drag source at the CG could do neither.
    # Nose-on flow shadows the tandem boxes: a box behind a larger one sees no
    # clean frontal flow, so the axial (x) reference area is only the GROWTH in
    # frontal silhouette front-to-back (summing to ≈ the max cross-section, not
    # all three faces). Side/top faces are normal to the tandem axis — never
    # shadowed — so they use the full face.
    front_seen = 0.0
    for nm, b in FUSELAGE_BOXES.items():     # ordered nose → cabin → tail
        lx, ly, lz = b["size"]
        frontal = max(0.0, ly * lz - front_seen)   # un-shadowed frontal area
        front_seen = max(front_seen, ly * lz)
        # F_i = -½·ρ·Cd_i·A_i · v_i·|v_i|: Cd low fore–aft (x), high broadside.
        A2 = -0.5 * np.diag([FUSELAGE_CD_AXIAL * frontal,
                             FUSELAGE_CD_SIDE * lx * lz,
                             FUSELAGE_CD_SIDE * lx * ly])
        a.add(DragSurface(f"fus_{nm}",
                          force_tensors=[np.zeros((3, 3)), A2],
                          mount_offset=b["center"]))

    # Propeller: thrust along the centreline at the hub, with the engine's
    # reaction roll torque about the thrust axis.
    a.add(Thruster("prop", force=(THRUST, 0.0, 0.0),
                   torque=(PROP_TORQUE, 0.0, 0.0),
                   mount_offset=(0.0, 0.0, 0.0)))

    # Tricycle landing gear: frictionless point contacts (free-rolling
    # wheels) on the ground plane.
    for name, pos in GEAR.items():
        a.add(Collider(name, stiffness=1.5e5, damping=8000.0,
                       friction=GEAR_FRICTION, mount_offset=pos))

    w = (World()
         .add_field(GravityField().add_uniform((0.0, 0.0, -9.81)))
         .add_field(FluidField().add_uniform(density=1.225))
         .add_field(CollisionField().add_half_space(origin=(0.0, 0.0, 0.0),
                                                    normal=(0.0, 0.0, 1.0))))
    w.add_craft(a, position=(0.0, 0.0, START_Z))    # parked on the runway
    return w


def draw_airframe(viz, craft):
    """Render the airframe straight from the SIM parts — nothing is
    duplicated, so what you see is the geometry the physics uses. Each
    aerofoil / control surface is drawn at its part transform (the
    quarter-chord) with its real chord and span (= area / chord); the prop
    and gear at their transforms. Returns the ``{name: (hinge_pos, axis)}``
    map used to swing the moving control flaps each frame.

    Only the fuselage is decorative (it is a point drag source, not a
    shaped part): it is drawn from the measured FUSELAGE_BOXES prisms.
    """
    from manta.parts import Aerofoil, ControlSurface, Collider
    FIXED, MOVING, METAL = (120, 150, 210), (240, 150, 60), (150, 150, 158)
    hinges = {}
    for p in craft.parts:
        T = [float(v) for v in p.mount_offset]
        if isinstance(p, (Aerofoil, ControlSurface)):
            c = float(p.chord)
            span = float(p.area) / c
            nax = [float(v) for v in p.normal_axis]
            vertical = abs(nax[1]) > abs(nax[2])        # fin/rudder
            le_x, te_x = T[0] + 0.25 * c, T[0] - 0.75 * c

            def box(name, chord_len, cx, color):
                size = ((chord_len, 0.07, span) if vertical
                        else (chord_len, span, 0.07))
                viz.box(name, size, center=(cx, T[1], T[2]), color=color)

            if isinstance(p, ControlSurface):
                flap_c = float(p.flap_chord_fraction) * c
                fixed_c = c - flap_c
                box(f"world/plane/{p.name}_fix", fixed_c,
                    le_x - 0.5 * fixed_c, FIXED)
                flap_size = ((flap_c, 0.05, span) if vertical
                             else (flap_c, span, 0.05))
                viz.box(f"world/plane/{p.name}/s", flap_size,
                        center=(-0.5 * flap_c, 0, 0), color=MOVING)
                ax = np.cross([float(v) for v in p.chord_axis], nax)
                ax = ax / np.linalg.norm(ax)
                hinges[p.name] = ((te_x + flap_c, T[1], T[2]),
                                  tuple(round(v) for v in ax))
            else:
                box(f"world/plane/{p.name}", c, T[0] - 0.25 * c, FIXED)
        elif isinstance(p, Collider):
            viz.box(f"world/plane/strut_{p.name}", (0.09, 0.09, -0.35 - T[2]),
                    center=(T[0], T[1], 0.5 * (-0.35 + T[2])), color=(60, 60, 65))
            viz.point(f"world/plane/wheel_{p.name}", tuple(T),
                      color=(30, 30, 35), radius=0.3)
    # Fuselage (decorative): three measured prisms. The cabin is the entity
    # the camera tracks.
    for nm, b in FUSELAGE_BOXES.items():
        viz.box(f"world/plane/fus_{nm}", b["size"], center=b["center"],
                color=METAL)
    viz.track("world/plane/fus_cabin")
    return hinges


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.002
    duration = args.duration or (1e9 if args.keyboard else 130.0)

    world = build_world()
    craft = world.crafts[0]
    sim = TargetNumpy(Sim(world))

    # Scripted flight, flown like a real 172 at its true ~1300 N power: a long
    # full-throttle takeoff roll (the elevator trim rotates it near 44 m/s and
    # holds the climb), then a gentle left turn, roll level, and an idle glide
    # back down to land. The reduced power makes the takeoff roll ~1 km, so the
    # timings are far later than a brisk demo would use.
    script = [(0.5, 4.0, {"x"}),       # full throttle → long takeoff roll, climb
              (60.0, 60.5, {"q"}),     # bank left into a gentle climbing turn
              (66.0, 66.5, {"e"}),     # roll back to wings-level
              (74.0, 110.0, {"z"})]    # throttle to idle → glide down and land
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  S/W pitch up/down   A/D yaw   Q/E roll   "
              "X/Z throttle   (Ctrl-C to quit)")
        if isinstance(ctrl, TerminalController):
            print("Type into THIS terminal; watch the viewer.\n")

    viz = None if args.no_viz else Viz("manta/airplane", addr=args.viz_addr)
    hinges = {}
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=2000.0, color=(70, 110, 70, 160))
        viz.box("world/runway", (1400.0, 30.0, 0.01), center=(650.0, 0.0, 0.0),
                color=(120, 120, 125))
        # The whole airframe is drawn from the sim parts — see draw_airframe.
        hinges = draw_airframe(viz, craft)

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

            # Direct key → deflection mapping (no inner loops). A positive
            # ControlSurface command is trailing-edge-DOWN, which ADDS lift, so
            # the raw senses are:
            #   +ail_l  raises the LEFT wing  → rolls RIGHT
            #   +elev   lifts the tail        → pitches nose DOWN
            #   +rud    pushes the fin to +y  → yaws nose RIGHT
            # ELEV_SIGN and the (e−q) aileron order flip these so the documented
            # keys read intuitively (S = nose up, Q = roll left, D = nose right).
            throttle = float(np.clip(
                throttle + (ctrl.held("x") - ctrl.held("z")) * THR_RATE
                * FRAME, 0.0, THR_MAX))
            # A whisker of aileron trim cancels the propeller's torque roll
            # (exactly what a real plane's trim tab is set for): +trim lifts the
            # left wing, rolling right against the prop's left-roll torque.
            ail = MAX_AIL * (ctrl.held("e") - ctrl.held("q")) \
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
                                 quat(axis, ang))
                if abs(throttle - thr_logged) > 0.005:
                    thr_logged = throttle
                    viz.arrow("world/plane/thrust", (0, 0, 0),
                              (3.0 * throttle, 0, 0), color=(235, 80, 80),
                              radius=0.2)
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
                yaw, pitch, roll = euler_zyx(np.asarray(st["orientation"]).ravel())
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
