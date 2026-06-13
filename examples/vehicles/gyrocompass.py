"""Gyrocompass — the EKF finds true north from Earth's rotation, automatically.

A gyrocompass needs no magnetometer and no GPS: it finds north by *sensing the
planet spin*. The world frame is inertial and the Earth turns in it, so a
strapdown IMU reads the body's INERTIAL angular rate — a body fixed to the
Earth therefore measures the spin vector Ω (7.29e-5 rad/s). At a latitude φ
that vector has a horizontal component Ω·cos φ pointing at TRUE NORTH; the
accelerometer measures gravity (down). Together they fix the full attitude —
including heading.

The point of this example is that **nobody writes that logic**. The instrument
is a heavily-damped buoy floating in a calm sea at 45°N, co-rotating with the
Earth, carrying only an IMU. The EKF is auto-derived from the same world
model as the truth — the planet's spin is in its *process model* — so heading
information reaches the filter on its own, through two indirect channels:

  * the drag dynamics predict the buoy's angular rate co-rotating with the sea
    *as seen from the estimated attitude*, R(q̂)ᵀΩ, while the gyro keeps
    measuring R(q_true)ᵀΩ — a persistent innovation whenever heading is wrong;
  * a heading error rotates into a tilt error at the rate Ω·cos φ (the classic
    INS alignment coupling), and the accelerometer sees tilt at gain g.

Neither sensor measures heading directly — the gyro's measurement Jacobian
w.r.t. orientation is exactly zero — which is why the `observability()` rank
test calls heading unobservable here. It is *weakly* observable, through slow
dynamics; `sigma_horizon()` (printed at startup) resolves it, showing heading
σ collapsing over the alignment window. From a 35° initial heading error the
filter pulls most of it out in the first seconds, then grinds the rest out
Earth-rate slow: ~1.5° after five minutes and still closing (real marine
gyrocompasses take minutes-to-hours for the same reason — the north signal
is only Ω·cos φ ≈ 5e-5 rad/s).

A tiny `ProcessNoise` part declares the unmodeled buffeting (micro-chop): the
same channel jitters the truth and auto-builds the filter's Q, keeping the
covariance honest so the slow Earth-rate channel stays weighted in.

Run::

    .venv/bin/python -m examples.vehicles.gyrocompass            # default 35° start
    .venv/bin/python -m examples.vehicles.gyrocompass --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, EKF, NoiseDriver, Sim, TargetNumpy, World
from manta.parts import DragSurface, IMU, Mass, PointBuoy, ProcessNoise
from manta.planets import Earth, SeaWaves

from .._control import Pacer, common_args
from .._viz import Viz

RHO = 1025.0
MASS = 120.0
LAT = np.radians(45.0)                       # latitude (sets the north signal)
EARTH_AXIS = (float(np.cos(LAT)), 0.0, float(np.sin(LAT)))   # north = +x
OMEGA = Earth.SIDEREAL * np.array(EARTH_AXIS)                # world-frame spin
GYRO_SIGMA = 1.0e-5                           # FOG/RLG-grade: resolves 7e-5 rad/s


def build_world(heading_deg: float):
    """A north-seeking buoy at LAT, started pointing `heading_deg` off north."""
    b = Craft("gyro")
    # Low CoM under a buoy line → strong pendulum levelling, and heavy
    # top/bottom drag → it settles dead-still fast (a moving accel/gyro would
    # corrupt the gravity + spin references).
    b.add(Mass("hull", mass=MASS, moi=(8.0, 30.0, 30.0), transform=(0, 0, -0.5)))
    for i, x in enumerate(np.linspace(-1.0, 1.0, 7)):
        b.add(PointBuoy(f"buoy{i}", volume=MASS / RHO / 7, transform=(float(x), 0, 0)))
        b.add(DragSurface.isotropic_quadratic(f"du{i}", area=0.6, drag_coefficient=3.0,
                                              transform=(float(x), 0, 0.3)))
        b.add(DragSurface.isotropic_quadratic(f"dl{i}", area=0.6, drag_coefficient=3.0,
                                              transform=(float(x), 0, -0.3)))
    b.add(IMU("imu", gyro_noise_sigma=GYRO_SIGMA, accel_noise_sigma=1e-3))
    # Unmodeled buffeting: jitters the truth AND auto-builds the EKF's Q
    # (zero Q ⇒ the covariance collapses and the slow Earth-rate channel
    # gets down-weighted into a stall).
    b.add(ProcessNoise("pn", force_noise_sigma=0.02, torque_noise_sigma=1e-3))

    earth = Earth(position=(0, 0, -Earth.R_EQ), rotation_rate=Earth.SIDEREAL,
                  rotation_axis=EARTH_AXIS, waves=SeaWaves(amplitude=0.0, wavelength=12.0),
                  surface_smoothing=0.3)
    w = World().add_planet(earth)
    # Co-rotate (so the IMU senses Ω) AND co-orbit (v = Ω×r, so the buoy is at
    # rest relative to the sea — no current).
    r0 = np.array([0, 0, -0.2]) - np.array([0, 0, -Earth.R_EQ])
    v_orbit = np.cross(OMEGA, r0)
    h = np.radians(heading_deg)
    w.add_craft(b, position=(0, 0, -0.2), velocity=tuple(v_orbit),
                angular_velocity=tuple(OMEGA),
                orientation=(np.cos(h / 2), 0, 0, np.sin(h / 2)))
    return w, b, v_orbit


def _heading(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> None:
    p = common_args(__doc__)
    p.add_argument("--heading", type=float, default=35.0,
                   help="true heading off north (deg) — the filter starts at 0")
    args = p.parse_args()
    dt = 0.02
    duration = args.duration or 300.0

    w, c, v_orbit = build_world(args.heading)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=1))

    # The filter: same world model, gyro + accel only. Coarse-alignment
    # prior — position/velocity/Ω known (moored at a known latitude),
    # heading NOT (orientation starts at identity, prior σ ~40°).
    ekf_ir = EKF(w, sensors=["gyro.imu.gyro", "gyro.imu.accel"])
    ekf = TargetNumpy(ekf_ir)
    P0 = np.diag([0.5] * 3 + [0.5] * 3 + [0.1] * 3 + [1e-4] * 3)
    ekf.reset(state={"gyro": c.initial_state(
                  position=(0, 0, -0.2), velocity=tuple(v_orbit),
                  angular_velocity=tuple(OMEGA),
                  orientation=(1.0, 0.0, 0.0, 0.0))},
              P=P0)
    IMU_SENSORS = ["gyro.imu.gyro", "gyro.imu.accel"]

    # The rank test calls heading unobservable — the Earth-rate channel is
    # 5 orders below its threshold. The σ-horizon recursion resolves it.
    rep = ekf_ir.observability()
    unobs = ", ".join(n.split(".")[-1] for n, _ in rep.unobservable) or "none"
    print(f"  rank test: {rep.rank}/{rep.tangent_dim} "
          f"(calls unobservable: {unobs})")
    sh = ekf_ir.sigma_horizon(horizon=60.0, dt=dt, P0=P0)
    s0, sT = sh.sigma0("gyro.orientation"), sh.sigmaT("gyro.orientation")
    print(f"  σ-horizon: orientation σ {np.degrees(s0):.0f}° → "
          f"{np.degrees(sT):.2f}° in 60 s — heading is (weakly) observable\n")
    print(f"  latitude {np.degrees(LAT):.0f}°N, Ω={Earth.SIDEREAL:.2e} rad/s "
          f"(north component {Earth.SIDEREAL*np.cos(LAT):.2e})")
    print(f"  true heading {args.heading:.1f}°; filter starts at 0°\n")

    viz = None if args.no_viz else Viz("manta/gyrocompass", addr=args.viz_addr)
    if viz is not None:
        viz.box("world/buoy/hull", (2.0, 0.4, 0.4), color=(210, 200, 90))
        viz.plane("world/sea", z=0.0, size=12.0, color=(40, 90, 160, 120))
        viz.arrow("world/true_north", (0, 0, 1.2), (3, 0, 0), color=(90, 230, 120),
                  radius=0.05)               # +x = true north (fixed)

    pacer = Pacer() if (args.keyboard or viz is not None) else None
    print(f"{'t (s)':>6} {'est °':>8} {'true °':>8} {'err °':>7} {'σ_yaw °':>8}")
    for i in range(int(duration / dt)):
        t = i * dt
        if pacer is not None:
            pacer.pace(t)
        sim.step(dt)
        for nm in IMU_SENSORS:
            ekf.update(nm, sim.reading(nm))
        ekf.predict(dt)

        eq = np.asarray(ekf.state_dict()["gyro"]["orientation"]).ravel()
        est_heading = _heading(eq)
        tp = np.asarray(sim.state["gyro"]["position"]).ravel()
        tq = np.asarray(sim.state["gyro"]["orientation"]).ravel()
        if viz is not None and viz.due(t):
            viz.t(t)
            viz.pose("world/buoy", tp, tq)
            # Needle: where the filter currently thinks north is.
            a = np.radians(-est_heading)
            viz.arrow("world/buoy/needle", (0, 0, 0.6),
                      (3 * np.cos(a), 3 * np.sin(a), 0), color=(235, 80, 80),
                      radius=0.05)
        if (i + 1) % int(10 / dt) == 0:
            true = _heading(tq)
            err = abs((est_heading - true + 180) % 360 - 180)
            sig_yaw = np.degrees(np.sqrt(np.asarray(ekf.P)[5, 5]))
            print(f"{t+dt:>6.0f} {est_heading:>8.2f} {true:>8.2f} "
                  f"{err:>7.2f} {sig_yaw:>8.2f}")


if __name__ == "__main__":
    main()
