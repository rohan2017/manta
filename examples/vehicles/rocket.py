"""Rocket — thrust-vector control via a two-axis gimbal, flown by LQR.

A tall body with a single engine hung BELOW it on a stacked two-axis
gimbal (`RevoluteJoint` x over `RevoluteJoint` y). The engine is not an
abstraction: a `Mass` rides the inner gimbal next to the `Thruster`, so
vectoring the nozzle slings real mass around — the reaction wiggles the
whole rocket, and the gimbal torque commands fight the engine's own
inertia. All of that comes from the joint-space block solve; the model
just declares the parts.

The LQR is the interesting bit:

  * Its inputs are the two gimbal TORQUES plus throttle — it has to
    swing the engine to steer the thrust, the actual TVC control
    problem (gimbal angle/rate are tracked states, not commands).
  * Roll about the thrust axis is genuinely unactuated, so the roll
    tangent directions get zero `Q` weight: their gain rows come out
    zero and the Riccati solve converges around them.
  * It is built at the CONTROL rate (50 Hz), not the physics rate: the
    landing-leg contact springs need 500 Hz integration, but at
    dt=0.002 the discrete plant's poles crowd the unit circle and the
    Riccati fixpoint iteration would need ~1e5 sweeps. Two rates, one
    model — the controller holds each command over 10 physics substeps.

Three `Collider` feet in a triangle let it SIT on the pad: the mission
starts engine-off on the legs, ignites, climbs to altitude, translates
sideways, descends, and cuts the engine at touchdown to settle back
onto them.

With ``--keyboard``: WASD moves the horizontal setpoint, space/shift
changes altitude, ``g`` toggles the engine (cut it in the air if you
want to watch it fall over).

Run::

    .venv/bin/python -m examples.vehicles.rocket             # scripted mission
    .venv/bin/python -m examples.vehicles.rocket --keyboard
    .venv/bin/python -m examples.vehicles.rocket --no-viz    # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, LQR, Sim, TargetNumpy, World
from manta.fields import CollisionField, FluidField, GravityField, HalfSpace
from manta.parts import Collider, DragSurface, Mass, RevoluteJoint, Thruster

from .._control import Pacer, common_args, make_controller
from .._viz import Viz

G = 9.81
M_BODY, M_ENG = 8.0, 1.5           # body / gimballed-engine mass (kg)
MAXT = 250.0                       # full-throttle thrust (N): T/W ≈ 2.7
GIM_Z = -0.8                       # gimbal pivot, below the craft origin
ENG_Z = -0.18                      # engine COM + nozzle, below the pivot
LEG_R, LEG_Z = 0.45, -1.05         # landing-leg triangle radius / foot z
BODY_R, BODY_L = 0.16, 2.0         # hull cylinder (viz + the slender moi)
Z_PAD = 1.06                       # craft-origin height standing on the legs
DT_SIM, SUBSTEPS = 0.002, 10       # contact physics rate / control divider
DT_CTRL = DT_SIM * SUBSTEPS        # 50 Hz — the rate the LQR is solved at
APEX = np.array([0.0, 0.0, 10.0])  # climb target
PAD_B = np.array([6.0, 0.0, Z_PAD])  # landing pad


def build_world():
    c = Craft("rocket")
    # Slender-body inertia; the COM sits high so the engine hangs well
    # below it (lever arm ≈ 0.9 m for the thrust-vector torque).
    c.add(Mass("body", mass=M_BODY, moi=(2.8, 2.8, 0.35),
               transform=(0, 0, 0.3)))
    c.add(DragSurface.isotropic_quadratic("aero", area=0.07,
                                          drag_coefficient=0.8))

    # Two-axis gimbal: outer x-joint, inner y-joint, both torque-commanded
    # ("saturating"). The engine Mass + Thruster ride the inner frame, so
    # the thrust direction AND the engine's inertia track the gimbal.
    gx = RevoluteJoint("gimbal_x", axis=(1, 0, 0), mode="saturating",
                       stall_torque=30.0, damping=1.0,
                       transform=(0, 0, GIM_Z))
    gy = RevoluteJoint("gimbal_y", axis=(0, 1, 0), mode="saturating",
                       stall_torque=30.0, damping=1.0)
    gy.add(Mass("engine", mass=M_ENG, moi=(0.02, 0.02, 0.01),
                transform=(0, 0, ENG_Z)))
    gy.add(Thruster("main", force=(0, 0, MAXT), transform=(0, 0, ENG_Z)))
    gx.add(gy)
    c.add(gx)

    # Three feet in a triangle: spring-damper contact plus viscous
    # tangential friction so the rocket grips the pad instead of skating.
    for k, name in enumerate(("leg_a", "leg_b", "leg_c")):
        th = np.pi / 2 + k * 2 * np.pi / 3
        c.add(Collider(name, stiffness=1.5e4, damping=400.0, friction=150.0,
                       transform=(LEG_R * np.cos(th), LEG_R * np.sin(th),
                                  LEG_Z)))

    cf = CollisionField()
    cf.add(HalfSpace(origin=(0, 0, 0), normal=(0, 0, 1)))
    w = (World()
         .add_field(GravityField(g=(0, 0, -G)))
         .add_field(FluidField().add_uniform(density=1.225))
         .add_field(cf))
    w.add_craft(c, position=(0, 0, Z_PAD))
    return w, c


def gimbal_quat(gx: float, gy: float) -> np.ndarray:
    """wxyz quaternion of the inner gimbal frame in body coords:
    outer rotation about x, then inner about y (q = qx ⊗ qy)."""
    cx, sx = np.cos(gx / 2), np.sin(gx / 2)
    cy, sy = np.cos(gy / 2), np.sin(gy / 2)
    return np.array([cx * cy, sx * cy, cx * sy, -sx * sy])


def main() -> None:
    args = common_args(__doc__).parse_args()
    duration = args.duration or (1e9 if args.keyboard else 60.0)
    m_tot = M_BODY + M_ENG
    hover = m_tot * G / MAXT

    w, _ = build_world()

    # Tracked tangent order (16): position(3) orientation(3) velocity(3)
    # angular_velocity(3) gimbal_x.angle/rate gimbal_y.angle/rate. Roll
    # about the thrust axis — orientation z and angular-velocity z — is
    # unactuated (a single gimballed engine has no roll authority), so
    # its Q weights are ZERO: the gain rows for those directions vanish
    # and nothing in the symmetric model ever excites them.
    Q = np.diag([2.0, 2.0, 6.0,       # position
                 6.0, 6.0, 0.0,       # orientation   (roll: unactuated)
                 4.0, 4.0, 4.0,       # velocity
                 1.0, 1.0, 0.0,       # angular rate  (roll: unactuated)
                 0.5, 0.05,           # gimbal_x angle, rate
                 0.5, 0.05])          # gimbal_y angle, rate
    R = np.diag([2.0, 2.0, 0.4])      # [gimbal_x τ, gimbal_y τ, throttle]
    lqr_t = LQR(
        w, x_ref={"rocket": {"position": tuple(APEX)}},
        u_ref={"main.throttle": hover},
        track=["rocket.position", "rocket.velocity",
               "rocket.orientation", "rocket.angular_velocity",
               "rocket.gimbal_x.angle", "rocket.gimbal_x.rate",
               "rocket.gimbal_y.angle", "rocket.gimbal_y.rate"],
        Q=Q, R=R, dt=DT_CTRL)
    lqr = TargetNumpy(lqr_t)
    # The unit-circle closed-loop modes are exactly the zero-weighted
    # roll pair; everything actuated is inside.
    cl = np.sort(np.abs(lqr_t.closed_loop_eigs))
    print(f"closed-loop |eig|: actuated max {cl[-3]:.4f} (stable < 1), "
          f"roll pair {cl[-2]:.4f}, {cl[-1]:.4f} (unactuated)")

    sim = TargetNumpy(Sim(w))

    # Mission state machine. Scripted: sit → ignite → climb → translate →
    # descend → cut at touchdown → settle. Keyboard: fly the setpoint.
    # slew = setpoint rate limit (m/s); None = step the setpoint. The
    # translate leg is a deliberate 6 m step: watch the gimbal slam over
    # and the engine's reaction wiggle the hull before the rocket leans
    # into the move.
    phases = [("pad",   np.array([0.0, 0.0, Z_PAD]), 0.0),
              ("climb", APEX,                        2.0),
              ("xlate", np.array([6.0, 0.0, 10.0]),  None),
              ("land",  PAD_B - (0.0, 0.0, 0.10),    0.7)]
    ph, engine_on, touchdown_t = 0, False, None
    setpoint = np.array([0.0, 0.0, Z_PAD])
    u0 = {n: 0.0 for n in lqr_t.input_names}
    u = dict(u0)

    ctrl = make_controller(args.keyboard)
    if args.keyboard:
        engine_on = True
        print("\nControls:  W/S north/south   A/D east/west   "
              "space/shift up/down   G engine on/off   (Ctrl-C to quit)\n")

    viz = None if args.no_viz else Viz("manta/rocket", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=30.0, color=(60, 70, 80, 160))
        for tag, ctr in (("a", (0, 0, 0)), ("b", PAD_B * (1, 1, 0))):
            viz.disc(f"world/pad_{tag}", 0.8, center=ctr,
                     color=(110, 110, 120, 255), thickness=0.01)
        viz.split_cylinder("world/rocket/hull", BODY_R, BODY_L,
                           colors=((220, 222, 228), (180, 60, 50)))
        viz.pose("world/rocket/hull", (0, 0, GIM_Z + BODY_L / 2))
        for k, tag in enumerate("abc"):       # legs: hull rim → feet
            th = np.pi / 2 + k * 2 * np.pi / 3
            viz.line(f"world/rocket/leg_{tag}",
                     [(BODY_R * np.cos(th), BODY_R * np.sin(th), GIM_Z + 0.4),
                      (LEG_R * np.cos(th), LEG_R * np.sin(th), LEG_Z)],
                     color=(140, 140, 150), radius=0.02)
        viz.box("world/rocket/gimbal/nozzle", (0.07, 0.07, 0.11),
                center=(0, 0, ENG_Z + 0.02), color=(70, 70, 75))

    pacer = Pacer() if (args.keyboard or viz is not None) else None
    n = int(duration / DT_CTRL)
    sp_rate = 1.5                                  # keyboard setpoint slew
    print(f"\n{'t (s)':>6}  {'phase':>6}  {'pos':>22}  "
          f"{'gimbal (deg)':>14}  {'thr':>5}")
    try:
        for i in range(n):
            t = i * DT_CTRL
            if pacer is not None:
                pacer.pace(t)
            st = sim.state["rocket"]
            pos = np.asarray(st["position"]).ravel()
            vel = np.asarray(st["velocity"]).ravel()

            if args.keyboard:
                ctrl.update(t)
                setpoint += np.array([
                    ctrl.held("w") - ctrl.held("s"),
                    ctrl.held("d") - ctrl.held("a"),
                    ctrl.held("space") - ctrl.held("shift"),
                ], dtype=float) * sp_rate * DT_CTRL
                if ctrl.pressed("g"):
                    engine_on = not engine_on
                    print(f"   engine {'ON' if engine_on else 'CUT'}")
                name = "fly"
            else:
                name, target, slew = phases[ph]
                if name == "pad":                  # sit, then ignite
                    if t >= 1.5:
                        ph, engine_on = 1, True
                        print("   ignition")
                elif name == "land":
                    if (engine_on and pos[2] < Z_PAD + 0.04
                            and abs(vel[2]) < 0.35):
                        engine_on, touchdown_t = False, t
                        print(f"   touchdown at t={t:.2f} s — engine cut")
                elif (np.linalg.norm(pos - target) < 0.3
                        and np.linalg.norm(vel) < 0.4):
                    ph += 1
                d = target - setpoint              # slew-limited setpoint
                dist = np.linalg.norm(d)
                step = np.inf if slew is None else slew * DT_CTRL
                setpoint = (target.copy() if dist <= step
                            else setpoint + d / dist * step)

            if engine_on:
                # Same gain everywhere: K retargets exactly under the
                # translation (see LQR docstring); the throttle clip is
                # the engine's [0, MAXT] envelope.
                lqr.retarget({"rocket": {"position": setpoint}})
                u = lqr.control({"rocket": st})
                u["rocket.main.throttle"] = float(
                    np.clip(u["rocket.main.throttle"], 0.0, 1.0))
            else:
                u = dict(u0)                       # ballistic; gimbal limp
            for name_u, val in u.items():
                sim.command(name_u).set(float(val))
            for _ in range(SUBSTEPS):              # contact physics @ 500 Hz
                sim.step(DT_SIM)

            gx = float(np.asarray(st["gimbal_x.angle"]).ravel()[0])
            gy = float(np.asarray(st["gimbal_y.angle"]).ravel()[0])
            if viz is not None:
                quat = np.asarray(st["orientation"]).ravel()
                thr = u["rocket.main.throttle"]
                viz.t(t)
                viz.pose("world/rocket", pos, quat)
                viz.pose("world/rocket/gimbal", (0, 0, GIM_Z),
                         gimbal_quat(gx, gy))
                viz.arrow("world/rocket/gimbal/plume", (0, 0, ENG_Z),
                          (0, 0, -2.2 * thr / hover),   # 2.2 m at hover
                          color=(255, 150, 50), radius=0.05 + 0.04 * thr)
                viz.trail("world/trail", pos, max_len=3000, min_dist=0.05)
                viz.point("world/setpoint", setpoint,
                          color=(90, 230, 120), radius=0.15)

            if (i + 1) % int(2.0 / DT_CTRL) == 0:
                print(f"{t:>6.2f}  {name:>6}  {np.round(pos, 2) + 0.0!s:>22}"
                      f"  ({np.degrees(gx):+5.1f}, {np.degrees(gy):+5.1f})"
                      f"  {u['rocket.main.throttle']:>5.2f}")
            if touchdown_t is not None and t > touchdown_t + 4.0:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()

    st = sim.state["rocket"]
    pos = np.asarray(st["position"]).ravel()
    q = np.asarray(st["orientation"]).ravel()
    tilt = np.degrees(2 * np.arccos(min(1.0, abs(float(q[0])))))
    vmag = float(np.linalg.norm(np.asarray(st["velocity"]).ravel()))
    print(f"\nfinal: pos {np.round(pos, 3) + 0.0}  tilt {tilt:.2f} deg  "
          f"|v| {vmag:.4f} m/s")
    if not args.keyboard:
        err = np.linalg.norm(pos[:2] - PAD_B[:2])
        print(f"landed {err:.3f} m from the pad center, on its legs"
              if tilt < 5.0 and abs(pos[2] - Z_PAD) < 0.05
              else "did NOT land cleanly")


if __name__ == "__main__":
    main()
