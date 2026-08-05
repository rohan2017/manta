"""Quadcopter — the full estimate→control stack, flown to a setpoint.

This is the payoff demo for manta's sibling-transform architecture. One
model is consumed by three transforms that snap together over the shared
`Linearization` seam:

    Sim(world)   — truth: a 4-rotor quad integrated forward, with process
                   noise (thruster `force_noise`) jittering the real motion
    EKF(world)   — state estimate (position, attitude, velocity, rates, and
                   the IMU gyro bias) from a noisy GPS fix + gyro
    LQR(world)   — a full-state hover regulator; feedback on the *estimate*

The runtimes are plumbed by hand in the loop: the sim's noisy GPS + gyro
``reading``s are folded into the EKF with ``update``, the LQR's ``control``
commands are passed to ``step(dt, u=...)`` on the sim (truth) and to
``predict`` on the EKF (known input). The control loop is then just
``control → step → update → predict`` — you own the order, the same shape
in every backend.

Both noise covariances are **model-derived** from one σ each: GPS
`position_noise` → the EKF's R; thruster `force_noise` → the process noise
Q (it enters the wrench, so the EKF reads it off ∂f/∂noise). A
`NoiseDriver` draws those same σ to jitter the truth, so the filter's
assumptions and the world it estimates are consistent by construction.

You fly it: WASD moves the horizontal setpoint, space/shift change
altitude — the LQR chases the moving target. Without ``--keyboard`` the
demo flies itself: straight-line trapezoidal velocity ramps to 10 random
waypoints in a 40 m cube. The LQR regulates about the moving reference —
``retarget`` feeds it the trajectory's position AND velocity — so the
quad sprints, flips over, and brakes onto each target (the cheap end of
MPC: precomputed feedforward + a constant-gain tracking regulator).

Run::

    .venv/bin/python -m examples.vehicles.quadcopter --keyboard
    .venv/bin/python -m examples.vehicles.quadcopter            # scripted
    .venv/bin/python -m examples.vehicles.quadcopter --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import (
    Craft, EKF, LQR, NoiseDriver, Sim, TargetNumpy, World,
)
from manta.fields import GravityField
from manta.parts import IMU, Mass, PositionSensor, Thruster

from .._control import Pacer, common_args, make_controller
from .._viz import Viz

L, M, G = 0.25, 1.0, 9.81          # arm length, mass, gravity
MAXT, KYAW = 6.0, 0.05             # per-rotor max thrust (N), yaw coeff
SIGMA_FORCE = 0.15                 # thruster force σ (N)  → process noise Q
SIGMA_GPS = 0.04                   # GPS σ (m)             → measurement R
SIGMA_GYRO = 0.01                  # gyro σ (rad/s)        → measurement R
ROTORS = ["front", "right", "back", "left"]
ARM = {"front": (+L, 0, 0), "right": (0, +L, 0),
       "back": (-L, 0, 0), "left": (0, -L, 0)}
PROP_R = 0.11                      # prop disc radius (viz)
COLD, HOT = np.array([50.0, 105.0, 235.0]), np.array([235.0, 60.0, 40.0])
# Trajectory ramps: A_MAX < g so a braking climb (thrust < weight) stays
# feasible in any direction; V_MAX sets the cruise. Both inside the
# rotors' authority, so the tracking LQR stays linear by construction.
A_MAX, V_MAX = 8.0, 12.0           # leg accel (m/s²), cruise speed (m/s)


def build_world():
    c = Craft("quad")
    c.add(Mass("body", mass=M, moi=(0.01, 0.01, 0.02)))
    F = (0.0, 0.0, MAXT)
    chir = {"front": +1, "right": -1, "back": +1, "left": -1}   # CW/CCW
    for name in ROTORS:
        c.add(Thruster(name, force=F,
                       torque=(0, 0, chir[name] * KYAW * MAXT),
                       mount_offset=ARM[name], force_noise_sigma=SIGMA_FORCE))
    c.add(IMU("imu", gyro_noise_sigma=SIGMA_GYRO, gyro_bias_sigma=2e-4))
    c.add(PositionSensor("gps", position_noise_sigma=SIGMA_GPS))
    w = World().add_field(GravityField(g=(0.0, 0.0, -G)))
    w.add_craft(c, position=(0, 0, 2.0))
    return w, c


def ramp_traj(p_a, p_b, a: float = A_MAX, v_max: float = V_MAX):
    """A straight-line leg ``p_a → p_b`` flown as a trapezoidal velocity
    ramp: accelerate at ``a``, cruise at ``v_max``, decelerate to rest
    (triangular when the leg is too short to reach cruise).

    Returns ``(sample, T)`` — ``sample(tau) -> (position, velocity,
    acceleration)`` for leg time ``tau``, clamped to the endpoints;
    ``T`` the leg duration.
    """
    p_a = np.asarray(p_a, dtype=float).copy()
    p_b = np.asarray(p_b, dtype=float).copy()
    d = float(np.linalg.norm(p_b - p_a))
    if d < 1e-9:
        return (lambda tau: (p_b, np.zeros(3), np.zeros(3))), 0.0
    u = (p_b - p_a) / d
    t_a = min(v_max / a, np.sqrt(d / a))     # accel time (≤ → triangular)
    v_pk = a * t_a
    t_c = (d - v_pk * t_a) / v_pk            # cruise time (0 if triangular)
    T = 2.0 * t_a + t_c

    def sample(tau: float):
        if tau <= 0.0:
            s, v, acc = 0.0, 0.0, 0.0
        elif tau < t_a:
            s, v, acc = 0.5 * a * tau * tau, a * tau, a
        elif tau < t_a + t_c:
            s, v, acc = 0.5 * a * t_a * t_a + v_pk * (tau - t_a), v_pk, 0.0
        elif tau < T:
            tr = T - tau
            s, v, acc = d - 0.5 * a * tr * tr, a * tr, -a
        else:
            s, v, acc = d, 0.0, 0.0
        return p_a + u * s, u * v, u * acc

    return sample, T


def tilt_q(acc):
    """The attitude feedforward (differential flatness): the wxyz
    quaternion tilting body-z onto the thrust direction that produces
    ``acc`` — specific force ``acc + g·ẑ`` — with no yaw. ``A_MAX < g``
    keeps the z-component positive, so this never degenerates."""
    d = np.asarray(acc, dtype=float) + np.array([0.0, 0.0, G])
    d = d / np.linalg.norm(d)
    axis = np.cross([0.0, 0.0, 1.0], d)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * np.arctan2(s, d[2])
    return np.concatenate([[np.cos(half)], axis / s * np.sin(half)])


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.01
    duration = args.duration or (1e9 if args.keyboard else 240.0)
    p0 = np.array([0.0, 0.0, 2.0])         # LQR operating point (hover here)
    hover = M * G / 4.0 / MAXT             # throttle per rotor at trim

    w, c = build_world()

    # Truth, with a noise driver so process + sensor noise are real. The
    # quad starts below and to the side of the hover point and flies up to
    # it under control.
    start = np.array([0.2, -0.15, 1.85])
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=11))
    sim.state["quad"]["position"] = start.copy()

    # Estimator — seeded at the (known) launch pose; R and Q are
    # model-derived. A tight initial P reflects that we know where we
    # started; the filter's job is then to reject the GPS/gyro noise while
    # the LQR flies on its estimate.
    ekf = TargetNumpy(EKF(w))
    ekf.reset(state={"quad": c.initial_state(position=tuple(start))},
              P=np.eye(ekf.spec.tangent_dim) * 0.2)

    # Full-state hover regulator (position + attitude + their rates).
    Q = np.diag([8, 8, 8, 0.5, 0.5, 0.5, 2, 2, 1, 0.2, 0.2, 0.2])
    lqr_t = LQR(
        w, x_ref={"quad": {"position": tuple(p0)}},
        u_ref={f"{r}.throttle": hover for r in ROTORS},
        regulate=["quad.position", "quad.velocity",
               "quad.orientation", "quad.angular_velocity"],
        Q=Q, R=np.eye(4) * 0.5, dt=dt)
    lqr = TargetNumpy(lqr_t)
    print(f"closed-loop |eig|max = "
          f"{np.max(np.abs(lqr_t.closed_loop_eigs)):.4f}  (stable < 1)")

    # GPS + gyro + accel feed the EKF each step (gated on freshness). The
    # accelerometer sees the gravity direction, which is what makes the
    # quad's roll/pitch observable (GPS + gyro alone leave attitude weakly
    # observable, and a full-state regulator needs it).
    EKF_SENSORS = ["quad.gps.position", "quad.imu.gyro", "quad.imu.accel"]

    # Keyboard: WASD = horizontal, space/shift = up/down. Without it the
    # demo flies an autonomous waypoint tour: a `ramp_traj` leg to each of
    # N_JUMPS random points in a cube (xy ∈ ±20 m, z ∈ 0–40 m), planning
    # the next leg once the quad has settled on the current target.
    ctrl = make_controller(args.keyboard)
    if args.keyboard:
        print("\nControls:  W/S forward/back   A/D left/right   "
              "space/shift up/down   (Ctrl-C to quit)\n")
    N_JUMPS = 10
    rng = np.random.default_rng(7)
    jumps = rng.uniform([-20, -20, 0], [20, 20, 40], (N_JUMPS, 3))
    visited = 0
    traj, t_leg, leg_t0 = None, 0.0, 0.0   # active leg (sample fn, T, start)
    THR_LIM = 2.5         # per-rotor overdrive clip (×MAXT → T/W ≈ 6)

    viz = None if args.no_viz else Viz("manta/quadcopter", addr=args.viz_addr)
    if viz is not None:
        viz.plane("world/ground", z=0.0, size=50.0, color=(60, 70, 80, 160))
        # Airframe: two crossed arms spanning the motors + a checkered
        # target-marker hub. The prop discs are re-logged per frame
        # (their colour tracks throttle).
        viz.box("world/quad/arm_x", (L, 0.018, 0.008), color=(115, 115, 115))
        viz.box("world/quad/arm_y", (0.018, L, 0.008), color=(115, 115, 115))
        viz.checker_sphere("world/quad/hub", 0.065)
    pos_off = ekf.spec.slot("quad.position").tangent_offset

    setpoint = p0.copy()
    sp_rate = 0.8                          # m/s setpoint slew while held
    n = int(duration / dt)
    # Live runs must hold sim time to the wall clock: uncapped, the loop
    # integrates several× real time and floods the viewer's channel —
    # latency then grows without bound and keyboard flying is hopeless.
    pacer = Pacer() if (args.keyboard or viz is not None) else None
    print(f"\n{'t (s)':>6}  {'true pos':>22}  {'setpoint':>18}  {'est err':>8}")
    try:
        for i in range(n):
            t = i * dt
            if pacer is not None:
                pacer.pace(t)
            est = ekf.state_dict()["quad"]
            ep = np.asarray(est["position"]).ravel()
            p_ref, v_ref, a_ref = setpoint, np.zeros(3), np.zeros(3)
            if args.keyboard:
                ctrl.update(t)
                move = np.array([
                    (ctrl.held("w") - ctrl.held("s")),
                    (ctrl.held("d") - ctrl.held("a")),
                    (ctrl.held("space") - ctrl.held("shift"))], dtype=float)
                setpoint += move * sp_rate * dt
            else:
                # Waypoint tour: ride the active leg's velocity ramp;
                # when its profile has played out and the estimate has
                # settled on the target, plan the next leg — through all
                # N_JUMPS, then stop.
                if traj is not None:
                    if t - leg_t0 >= t_leg:
                        traj = None        # profile done: hold the endpoint
                    else:
                        p_ref, v_ref, a_ref = traj(t - leg_t0)
                ev = np.asarray(est["velocity"]).ravel()
                if traj is None and (np.linalg.norm(ep - setpoint) < 0.25
                                     and np.linalg.norm(ev) < 0.5):
                    if visited == N_JUMPS:
                        break
                    nxt = jumps[visited]
                    visited += 1
                    traj, t_leg = ramp_traj(setpoint, nxt)
                    leg_t0 = t
                    setpoint = nxt.copy()
                    print(f"   → waypoint {visited}/{N_JUMPS}: "
                          f"{np.round(setpoint, 1) + 0.0}")

            # Regulate about the MOVING reference: x_ref carries the
            # trajectory's position, velocity AND the tilt that produces
            # its acceleration (`tilt_q`, the flatness feedforward) — so
            # the flip-to-brake is commanded by the plan at zero error,
            # not built up from lag. K is never re-solved: with no drag,
            # A, B are position- and velocity-independent (those moves
            # are exact); attitude is NOT an invariant direction, so the
            # tilted reference leans on LQR's gain margin — solid to ~40°
            # here. Feedforward from the plan, feedback on the deviation:
            # the cheap end of MPC.
            lqr.retarget({"quad": {"position": p_ref, "velocity": v_ref,
                                   "orientation": tilt_q(a_ref)}})
            u = lqr.control({"quad": est})

            # Attitude-priority mixer instead of a bare clip: saturating
            # any rotor flattens the DIFFERENTIAL commands (the torques)
            # and tumbles the quad, while clipping the COLLECTIVE only
            # costs a moment of altitude tracking. So split the four
            # throttles into collective + differential, shrink the
            # differential only if it can't fit the [0, THR_LIM] band at
            # all, and slide the collective to keep it intact.
            u_vec = np.array([u[f"quad.{r}.throttle"] for r in ROTORS])
            coll = u_vec.mean()
            diff = u_vec - coll
            span = float(diff.max() - diff.min())
            if span > THR_LIM:
                diff *= THR_LIM / span
            coll = float(np.clip(coll, -diff.min(), THR_LIM - diff.max()))
            for r, val in zip(ROTORS, coll + diff):
                u[f"quad.{r}.throttle"] = float(val)   # viz shows the mixed u

            sim.step(dt, u=u)              # apply commands; realize GPS+gyro
            for nm in EKF_SENSORS:         # fold each reading (all ungated)...
                ekf.update(nm, sim.reading(nm), u=u)
            ekf.predict(dt, u=u)           # ...then predict (you own the order)

            if viz is not None and viz.due(t):   # throttle to ~30 Hz
                tp = np.asarray(sim.state["quad"]["position"]).ravel()
                tq = np.asarray(sim.state["quad"]["orientation"]).ravel()
                viz.t(t)
                viz.pose("world/quad", tp, tq)
                viz.trail("world/trail", tp,   # world coords: not under the posed quad
                          max_len=2000, min_dist=0.1)
                viz.point("world/setpoint", setpoint, color=(90, 230, 120),
                          radius=0.2)
                viz.point("world/ref", p_ref,      # the ramp's moving target
                          color=(200, 255, 210), radius=0.07)
                for r in ROTORS:               # prop discs + thrust arrows
                    thr = float(np.clip(u[f"quad.{r}.throttle"],
                                        0.0, THR_LIM))
                    frac = thr / THR_LIM       # colour spans the full clip
                    col = tuple(np.rint((1 - frac) * COLD + frac * HOT)
                                .astype(int))
                    viz.disc(f"world/quad/{r}/prop", PROP_R,
                             center=(ARM[r][0], ARM[r][1], 0.02),
                             color=col + (185,), thickness=0.004,
                             static=False)
                    viz.arrow(f"world/quad/{r}/thrust", ARM[r],
                              (0, 0, 0.4 * thr / hover),
                              color=(255, 120, 60), radius=0.015)
                # The estimate, drawn as its own confidence: a 3σ ellipsoid
                # of the position covariance — a small sphere riding the
                # hub when the filter is sharp, ballooning when it isn't.
                ep = np.asarray(
                    ekf.state_dict()["quad"]["position"]).ravel()
                cov = ekf.P[pos_off:pos_off + 3, pos_off:pos_off + 3]
                viz.cov_ellipsoid("world/quad_est", ep, cov, nsigma=3.0,
                                  color=(235, 235, 255, 70), min_half=0.03)

            if (i + 1) % 100 == 0:
                tp = np.asarray(sim.state["quad"]["position"]).ravel()
                ep = np.asarray(ekf.state_dict()["quad"]["position"]).ravel()
                print(f"{t:>6.2f}  {np.round(tp, 2) + 0.0!s:>22}  "
                      f"{np.round(setpoint, 2) + 0.0!s:>18}  "
                      f"{np.linalg.norm(tp - ep):>8.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()

    final = np.asarray(sim.state["quad"]["position"]).ravel()
    print(f"\nfinal pos {np.round(final, 2) + 0.0}"
          f"  setpoint {np.round(setpoint, 2) + 0.0}"
          f"  (err {np.linalg.norm(final - setpoint):.3f} m)")


if __name__ == "__main__":
    main()
