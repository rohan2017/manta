"""Gyrocompass — finding true north from Earth's rotation alone.

A gyrocompass needs no magnetometer and no GPS: it finds north by *sensing the
planet spin*. The world frame is inertial and the Earth turns in it, so a
strapdown IMU reads the body's INERTIAL angular rate — a body fixed to the
Earth therefore measures the spin vector Ω (7.29e-5 rad/s). At a latitude φ
that vector has a horizontal component Ω·cos φ pointing at TRUE NORTH; the
accelerometer measures gravity (down). Two known world vectors (Ω, g) seen in
the body frame fix the full attitude — including heading — by TRIAD.

The catch is that Ω is tiny: a single gyro sample is almost all noise. So the
instrument *averages* the gyro over an alignment interval to pull Ω out of the
noise (this is why real gyrocompasses take minutes to settle), then TRIADs the
averaged gyro + accel against the known Ω and gravity.

Here the instrument is a heavily-damped buoy floating in a calm sea at 45°N,
co-rotating (and co-orbiting, `v = Ω×r`, so it feels no current) with the
Earth. From a wrong initial heading it aligns to < 1° in ~30 s.

Run::

    .venv/bin/python -m examples.vehicles.gyrocompass            # default 35° start
    .venv/bin/python -m examples.vehicles.gyrocompass --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, NoiseDriver, Sim, TargetNumpy, World
from manta.parts import DragSurface, IMU, Mass, PointBuoy
from manta.planets import Earth, SeaWaves

from .._control import Pacer, common_args
from .._viz import Viz

RHO = 1025.0
MASS = 120.0
LAT = np.radians(45.0)                       # latitude (sets the north signal)
EARTH_AXIS = (float(np.cos(LAT)), 0.0, float(np.sin(LAT)))   # north = +x
OMEGA = Earth.SIDEREAL * np.array(EARTH_AXIS)                # world-frame spin
G_WORLD = np.array([0.0, 0.0, -9.81])        # gravity at the instrument (down)
SETTLE = 15.0                                # let the buoy still itself first
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
    return w, b


def _triad(b1, b2, r1, r2):
    """Attitude R (body→world) from two vector pairs, body bᵢ = Rᵀ·world rᵢ."""
    def frame(u, v):
        x = u / np.linalg.norm(u)
        z = np.cross(u, v); z = z / np.linalg.norm(z)
        return np.column_stack([x, np.cross(z, x), z])
    return frame(r1, r2) @ frame(b1, b2).T


def _heading(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> None:
    p = common_args(__doc__)
    p.add_argument("--heading", type=float, default=35.0,
                   help="initial heading off north (deg)")
    args = p.parse_args()
    dt = 0.02
    duration = args.duration or 90.0

    w, c = build_world(args.heading)
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=1))

    viz = None if args.no_viz else Viz("manta/gyrocompass", addr=args.viz_addr)
    if viz is not None:
        viz.box("world/buoy/hull", (2.0, 0.4, 0.4), color=(210, 200, 90))
        viz.plane("world/sea", z=0.0, size=12.0, color=(40, 90, 160, 120))
        viz.arrow("world/true_north", (0, 0, 1.2), (3, 0, 0), color=(90, 230, 120),
                  radius=0.05)               # +x = true north (fixed)

    pacer = Pacer() if (args.keyboard or viz is not None) else None
    gyro_avg = np.zeros(3)
    accel_avg = np.zeros(3)
    n = 0
    est_heading = float("nan")
    print(f"  latitude {np.degrees(LAT):.0f}°N, Ω={Earth.SIDEREAL:.2e} rad/s "
          f"(north component {Earth.SIDEREAL*np.cos(LAT):.2e})")
    print(f"  start heading {args.heading:.1f}° off north; {SETTLE:.0f}s settle, "
          f"then align\n")
    print(f"{'t (s)':>6} {'align (s)':>9} {'est °':>8} {'true °':>8} {'err °':>7}")
    for i in range(int(duration / dt)):
        t = i * dt
        if pacer is not None:
            pacer.pace(t)
        sim.step(dt)
        o = sim.outputs()["gyro"]
        if t >= SETTLE:                       # average only once it's still
            n += 1
            gyro_avg += (np.asarray(o["imu.gyro"]).ravel() - gyro_avg) / n
            accel_avg += (np.asarray(o["imu.accel"]).ravel() - accel_avg) / n
            # Two world refs (Ω, gravity) seen in the body frame → attitude.
            R = _triad(gyro_avg, -accel_avg, OMEGA, G_WORLD)
            est_heading = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        tp = np.asarray(sim.state["gyro"]["position"]).ravel()
        tq = np.asarray(sim.state["gyro"]["orientation"]).ravel()
        if viz is not None and viz.due(t):
            viz.t(t)
            viz.pose("world/buoy", tp, tq)
            if not np.isnan(est_heading):
                # Needle: where the instrument currently thinks north is.
                a = np.radians(-est_heading)
                viz.arrow("world/buoy/needle", (0, 0, 0.6),
                          (3 * np.cos(a), 3 * np.sin(a), 0), color=(235, 80, 80),
                          radius=0.05)
        if (i + 1) % int(5 / dt) == 0 and t >= SETTLE:
            true = _heading(tq)
            err = abs((est_heading - true + 180) % 360 - 180)
            print(f"{t+dt:>6.0f} {t+dt-SETTLE:>9.0f} {est_heading:>8.2f} "
                  f"{true:>8.2f} {err:>7.2f}")


if __name__ == "__main__":
    main()
