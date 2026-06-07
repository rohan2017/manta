"""Rocket — ascent burn then a hoverslam, steered by gimbal-only LQR.

A tall body with a single engine hung BELOW it on a stacked two-axis
gimbal (`RevoluteJoint` x over `RevoluteJoint` y). The engine is not an
abstraction: a `Mass` rides the inner gimbal next to the `Thruster`, so
vectoring the nozzle slings real mass around — the reaction wiggles the
whole rocket, and the gimbal torque commands fight the engine's own
inertia. All of that comes from the joint-space block solve; the model
just declares the parts. Three `Collider` feet in a triangle let it sit
on the pad engine-off.

**The throttle follows a curve the loop commands, not the controller.**
The flight is two burns with a coast between — no continuous burn down:

    ascent   : uniform thrust, 6 s          (a = 1.3 g)
    coast    : engine OFF, up over apex and falling
    hoverslam: a single suicide burn, ~4 s, that nulls velocity AT the
               deck — thrust is the textbook law ``F = m·(v²/2h + g)``
               recomputed each tick from the descent speed v and the
               height-to-go h. It self-corrects for drag (a frozen
               thrust sized at ignition would stop metres high), so the
               rocket settles onto its legs instead of dropping.

A command port IS the curve sampler: the loop just writes
``throttle = F/MAXT`` each tick (the `Thruster` is linear), no model
change. The slam DURATION is set by WHERE it ignites: triggering at the
constant-decel stopping distance ``h = v·T_slam/2`` makes the burn last
about ``T_slam``.

With the throttle open-loop, the LQR's only real authority is the two
gimbal TORQUES (gimbal angle/rate are tracked states — the actual TVC
problem). Three control facts the demo leans on:

  * Roll about the thrust axis is genuinely unactuated → zero `Q`
    weight on the roll tangent pair; its gain rows come out zero.
  * The gimbal-to-torque gain scales with instantaneous thrust, but the
    ascent and slam thrusts are close (~0.30 vs ~0.33 throttle), so ONE
    gain set is stable on both — solved once at the ascent trim. (A very
    different burn, e.g. a 3x pulse, would need its own gain.)
  * It is solved at the CONTROL rate (50 Hz), not the physics rate: the
    leg springs need 500 Hz integration, but at dt = 0.002 the discrete
    poles crowd the unit circle and the Riccati fixpoint won't converge.
    Two rates, one model.

During the coast there is no thrust to vector — the position/velocity
references are pinned to the live state so the LQR only stabilizes
attitude, which it does by swinging the engine mass (momentum
exchange: the gimbal as a crude reaction wheel).

With ``--keyboard``: ``g`` fires the next burn (ignition on the pad;
the slam while falling — time it yourself), WASD steers the aim point.

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
M_TOT = M_BODY + M_ENG
MAXT = 400.0                       # full-throttle thrust (N): T/W ≈ 4.3
GIM_Z = -0.8                       # gimbal pivot, below the craft origin
ENG_Z = -0.18                      # engine COM + nozzle, below the pivot
LEG_R, LEG_Z = 0.45, -1.05         # landing-leg triangle radius / foot z
BODY_R, BODY_L = 0.16, 2.0         # hull cylinder (viz + the slender moi)
Z_PAD = 1.06                       # craft-origin height standing on the legs
DT_SIM, SUBSTEPS = 0.002, 10       # contact physics rate / control divider
DT_CTRL = DT_SIM * SUBSTEPS        # 50 Hz — the rate the LQR is solved at
PAD_B = np.array([6.0, 0.0, Z_PAD])  # landing pad (lateral aim point)

T_BURN1 = 6.0                      # ascent burn duration
THR_BURN1 = M_TOT * 1.3 * G / MAXT  # uniform ascent thrust (a = 1.3 g)
T_SLAM = 4.0                       # target hoverslam duration


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


def slam_throttle(v_down: float, height: float) -> float:
    """Closed-loop suicide-burn throttle: the constant-deceleration that
    brings descent speed ``v_down`` to zero over the remaining height,
    plus weight support — ``F = m·(v²/2h + g)``, clipped to [0, 1]. Run
    every tick it lands at the deck regardless of drag."""
    a_req = v_down * v_down / (2.0 * max(height, 0.05))
    return float(np.clip(M_TOT * (a_req + G) / MAXT, 0.0, 1.0))


def gimbal_quat(gx: float, gy: float) -> np.ndarray:
    """wxyz quaternion of the inner gimbal frame in body coords:
    outer rotation about x, then inner about y (q = qx ⊗ qy)."""
    cx, sx = np.cos(gx / 2), np.sin(gx / 2)
    cy, sy = np.cos(gy / 2), np.sin(gy / 2)
    return np.array([cx * cy, sx * cy, cx * sy, -sx * sy])


def tilt_from_vertical(q: np.ndarray) -> float:
    """Degrees between body z and world z. (NOT the total rotation
    angle — the unactuated roll accumulates harmlessly and would
    dominate that number.)"""
    w, x, y, z = q
    return float(np.degrees(np.arccos(
        np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def main() -> None:
    args = common_args(__doc__).parse_args()
    duration = args.duration or (1e9 if args.keyboard else 60.0)

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
    # One gain set, solved at the ascent thrust trim. The throttle row of
    # K is never used — the curve owns the throttle; only the two gimbal
    # rows steer. Ascent and slam thrust are close, so this gain is
    # stable on both (the unit-circle modes are exactly the zero-weighted
    # roll pair).
    lqr_t = LQR(
        w, x_ref={"rocket": {"position": tuple(PAD_B)}},
        u_ref={"main.throttle": THR_BURN1},
        track=["rocket.position", "rocket.velocity",
               "rocket.orientation", "rocket.angular_velocity",
               "rocket.gimbal_x.angle", "rocket.gimbal_x.rate",
               "rocket.gimbal_y.angle", "rocket.gimbal_y.rate"],
        Q=Q, R=R, dt=DT_CTRL)
    lqr = TargetNumpy(lqr_t)
    cl = np.sort(np.abs(lqr_t.closed_loop_eigs))
    print(f"closed-loop |eig|: actuated max {cl[-3]:.4f} (stable < 1), "
          f"roll pair {cl[-2]:.4f}, {cl[-1]:.4f} (unactuated, by design)")

    sim = TargetNumpy(Sim(w))

    ctrl = make_controller(args.keyboard)
    if args.keyboard:
        print("\nControls:  G fire the next burn (ignition / slam)   "
              "WASD steer the aim point   (Ctrl-C to quit)\n")

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

    # Mission state machine: pad → burn1 → coast → slam → down. The
    # throttle is always the curve's value; the LQR only ever contributes
    # the gimbal torques.
    phase, t_ignite = "pad", None
    aim = PAD_B[:2].copy()                 # lateral aim point
    apex, slam_used, touchdown_t = 0.0, 0.0, None
    u0 = {n: 0.0 for n in ("rocket.gimbal_x.torque_cmd",
                           "rocket.gimbal_y.torque_cmd",
                           "rocket.main.throttle")}

    pacer = Pacer() if (args.keyboard or viz is not None) else None
    n = int(duration / DT_CTRL)
    print(f"\nascent {T_BURN1:.0f} s @ thr {THR_BURN1:.2f}, then coast, "
          f"then a ~{T_SLAM:.0f} s hoverslam (thrust nulls v at the deck)")
    print(f"\n{'t (s)':>6}  {'phase':>5}  {'pos':>22}  {'vz':>6}  "
          f"{'gimbal (deg)':>14}  {'thr':>5}")
    try:
        for i in range(n):
            t = i * DT_CTRL
            if pacer is not None:
                pacer.pace(t)
            st = sim.state["rocket"]
            pos = np.asarray(st["position"]).ravel()
            vel = np.asarray(st["velocity"]).ravel()
            apex = max(apex, float(pos[2]))
            h = float(pos[2]) - Z_PAD

            if args.keyboard:
                ctrl.update(t)
                aim += np.array([ctrl.held("w") - ctrl.held("s"),
                                 ctrl.held("d") - ctrl.held("a")],
                                dtype=float) * 1.5 * DT_CTRL
                fire = ctrl.pressed("g")
            else:
                # Ignite on the pad; trigger the slam at the constant-decel
                # stopping distance for a T_SLAM-long burn (h = v·T/2).
                fire = ((phase == "pad" and t >= 1.5)
                        or (phase == "coast" and vel[2] < 0
                            and h <= -vel[2] * T_SLAM / 2))

            # --- phase transitions ---------------------------------------
            if phase == "pad" and fire:
                phase, t_ignite = "burn1", t
                print(f"   ignition — {T_BURN1:.0f} s ascent burn")
            elif phase == "burn1" and t - t_ignite >= T_BURN1:
                phase = "coast"
                print(f"   burnout at z={pos[2]:.1f} m, "
                      f"vz=+{vel[2]:.1f} m/s — coasting")
            elif phase == "coast" and fire:
                phase, t_ignite = "slam", t
                print(f"   SLAM ignition: z={pos[2]:.1f} m, "
                      f"vz={vel[2]:.1f} m/s")
            elif phase == "slam":
                tau = t - t_ignite
                if (h < 0.03 and abs(vel[2]) < 0.6) or tau >= T_SLAM + 1.0:
                    phase, touchdown_t, slam_used = "down", t, tau
                    print(f"   touchdown after {tau:.2f} s — z={pos[2]:.2f}, "
                          f"vz={vel[2]:+.2f} m/s")

            # --- throttle from the curve, gimbal from the LQR ------------
            if phase == "burn1":
                thr = THR_BURN1
            elif phase == "slam":
                thr = slam_throttle(-vel[2], h)   # closed-loop suicide burn
            else:                                 # pad / coast / down
                thr = 0.0

            engine_armed = phase in ("burn1", "coast", "slam")
            if engine_armed:
                if phase == "coast":
                    # No thrust to vector: pin the position/velocity refs
                    # to the live state so only attitude+gimbal regulate
                    # (by waving the engine — momentum exchange).
                    ref_p, ref_v = pos, vel
                else:
                    # Steer to the pad; z is the curve's business, so its
                    # error is zeroed (the throttle row is discarded and
                    # the gimbal rows are z-blind by symmetry anyway).
                    ref_p = np.array([aim[0], aim[1], pos[2]])
                    ref_v = np.zeros(3)
                lqr.retarget({"rocket": {"position": ref_p,
                                         "velocity": ref_v}})
                u = lqr.control({"rocket": st})
            else:
                u = dict(u0)
            u["rocket.main.throttle"] = thr

            for name_u, val in u.items():
                sim.command(name_u).set(float(val))
            for _ in range(SUBSTEPS):              # contact physics @ 500 Hz
                sim.step(DT_SIM)

            gx = float(np.asarray(st["gimbal_x.angle"]).ravel()[0])
            gy = float(np.asarray(st["gimbal_y.angle"]).ravel()[0])
            if viz is not None:
                quat = np.asarray(st["orientation"]).ravel()
                viz.t(t)
                viz.pose("world/rocket", pos, quat)
                viz.pose("world/rocket/gimbal", (0, 0, GIM_Z),
                         gimbal_quat(gx, gy))
                viz.arrow("world/rocket/gimbal/plume", (0, 0, ENG_Z),
                          (0, 0, -3.0 * thr),       # plume ∝ thrust
                          color=(255, 150, 50), radius=0.05 + 0.04 * thr)
                viz.trail("world/trail", pos, max_len=3000, min_dist=0.05)
                viz.point("world/aim", (aim[0], aim[1], 0.02),
                          color=(90, 230, 120), radius=0.15)

            if (i + 1) % int(2.0 / DT_CTRL) == 0:
                print(f"{t:>6.2f}  {phase:>5}  {np.round(pos, 1) + 0.0!s:>22}"
                      f"  {vel[2]:>+6.1f}  ({np.degrees(gx):+5.1f}, "
                      f"{np.degrees(gy):+5.1f})  {thr:>5.2f}")
            if touchdown_t is not None and t > touchdown_t + 4.0:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()

    st = sim.state["rocket"]
    pos = np.asarray(st["position"]).ravel()
    tilt = tilt_from_vertical(np.asarray(st["orientation"]).ravel())
    vmag = float(np.linalg.norm(np.asarray(st["velocity"]).ravel()))
    print(f"\napex {apex:.1f} m, hoverslam {slam_used:.2f} s")
    print(f"final: pos {np.round(pos, 3) + 0.0}  tilt {tilt:.2f} deg  "
          f"|v| {vmag:.4f} m/s")
    if not args.keyboard:
        err = np.linalg.norm(pos[:2] - PAD_B[:2])
        print(f"landed {err:.3f} m from the pad center, on its legs"
              if tilt < 5.0 and abs(pos[2] - Z_PAD) < 0.05
              else "did NOT land cleanly")


if __name__ == "__main__":
    main()
