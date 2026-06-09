"""Submarine — a neutrally-buoyant AUV that surfaces to fix, dives to run.

A cylindrical AUV in a real Earth ocean (`Earth` + `SeaWaves`): a line of
`PointBuoy`s along the hull rides the waves when surfaced, the centre of mass
sits below the centre of buoyancy so the hull is **passively pendulum-stable**
in roll and pitch, and the roll moment of inertia is much smaller than the
pitch one (a cylinder) so it holds a pitch attitude without constant thrust.

Thrust is a 5-thruster vector layout:
  * one powerful stern thruster — surge (+x);
  * two horizontal (fore + aft) — together: sway (±y); differential: yaw;
  * two vertical (fore + aft)   — together: heave (±z); differential: pitch.

Navigation is the point. Two GPS receivers (a custom water-gated
`PositionSensor` defined here) sit on the fore and aft deck: each only
resolves when its antenna is ABOVE the wave surface (it samples the fluid and
suppresses its reading underwater). Surfaced, the two fixes give absolute
position AND heading (the fore/aft baseline); submerged, the EKF
dead-reckons on the IMU gyro + DVL body-velocity until the next surfacing.

Controls:  W/S surge   A/D yaw   space/shift heave (rise/dive)   Q/E sway

Run::

    .venv/bin/python -m examples.vehicles.submarine --keyboard
    .venv/bin/python -m examples.vehicles.submarine            # scripted
    .venv/bin/python -m examples.vehicles.submarine --no-viz   # headless
"""

from __future__ import annotations

import numpy as np

from manta import Craft, EKF, NoiseDriver, Sim, TargetNumpy, World, wire
from manta.fields import FluidField
from manta.ir.frames import WorldFrame
from manta.ir.types import Scalar
from manta.parts import DVL, DragSurface, IMU, Mass, PointBuoy, PositionSensor, Thruster
from manta.parts.base import Output
from manta.planets import Earth, SeaWaves

from .._control import Pacer, common_args, make_controller
from .._viz import Viz

# --- sea state + hull -------------------------------------------------------
RHO = 1025.0                      # seawater density
MASS = 120.0
WAVES = SeaWaves(amplitude=0.18, wavelength=12.0)
SMOOTH = 0.20                     # water/air blend band (m)
HULL_L, HULL_R = 2.4, 0.30        # cylinder length / radius
N_BUOY = 7
BUOY_FAC = 1.15                   # total displaced volume / (MASS/RHO): slight
                                  # positive trim → it floats to the surface
COM_Z = -0.20                     # CoM below the buoy line (z=0) → pendulum
DECK_Z = 0.70                     # GPS mast: antennas well clear of the wave band

# --- thrust (N, N) ----------------------------------------------------------
SURGE, SWAY, HEAVE, YAW, PITCH = 900.0, 350.0, 450.0, 350.0, 120.0


# ---------------------------------------------------------------------------
# A water-gated GPS: a PositionSensor that only resolves above the surface.
# It samples the FluidField at its own mount point and reports a `wet` flag
# (≈1 submerged, ≈0 in air); the loop folds the fix only when it's dry.
# ---------------------------------------------------------------------------

class SurfaceGPS(PositionSensor):
    """`PositionSensor` that also reports whether its antenna is underwater.

    `wet` is a smooth 0..1 from the local fluid density (seawater → 1, air →
    0), so the example can gate the position fix on surfacing."""

    wet = Output()

    def update(self, ctx):
        out = super().update(ctx)
        rho = ctx.field(FluidField).value_at_sym(
            ctx.position[WorldFrame], ctx.t).density
        out.outputs["wet"] = ctx.sample(Scalar(rho * (1.0 / RHO)), rate=self.rate)
        return out


def build_world():
    sub = Craft("sub")
    # Cylinder: long in x. Roll MoI (about x) ≪ pitch/yaw (about y, z); CoM
    # below the buoy line for pendulum stability in roll + pitch.
    sub.add(Mass("hull", mass=MASS, moi=(5.0, 60.0, 60.0), transform=(0, 0, COM_Z)))

    # A line of buoys along the hull axis (z = 0): rides the wave profile when
    # surfaced. Co-located axial drag fades with submersion the same way.
    xs = np.linspace(-1.2, 1.2, N_BUOY)
    v_each = BUOY_FAC * MASS / RHO / N_BUOY
    for i, x in enumerate(xs):
        sub.add(PointBuoy(f"buoy{i}", volume=v_each, transform=(float(x), 0, 0)))
        sub.add(DragSurface.isotropic_quadratic(f"hd{i}", area=0.05,
                                                drag_coefficient=0.8,
                                                transform=(float(x), 0, 0)))
    # Top/bottom panels at three stations: these are OFF the roll axis, so
    # they damp roll + pitch rate (the axial line above cannot).
    for j, x in enumerate((1.0, 0.0, -1.0)):
        sub.add(DragSurface.isotropic_quadratic(f"top{j}", area=0.07,
                                                drag_coefficient=1.0,
                                                transform=(x, 0, HULL_R)))
        sub.add(DragSurface.isotropic_quadratic(f"bot{j}", area=0.07,
                                                drag_coefficient=1.0,
                                                transform=(x, 0, -HULL_R)))

    # 5-thruster vector layout.
    sub.add(Thruster("surge", force=(1, 0, 0), transform=(-1.2, 0, 0),
                     force_noise_sigma=3.0))
    sub.add(Thruster("h_fore", force=(0, 1, 0), transform=(1.0, 0, 0)))
    sub.add(Thruster("h_aft", force=(0, 1, 0), transform=(-1.0, 0, 0)))
    sub.add(Thruster("v_fore", force=(0, 0, 1), transform=(1.0, 0, 0)))
    sub.add(Thruster("v_aft", force=(0, 0, 1), transform=(-1.0, 0, 0)))

    # Sensors: IMU + DVL (dead-reckoning) + two water-gated GPS (fore/aft).
    sub.add(IMU("imu", gyro_noise_sigma=0.01, gyro_bias_sigma=5e-4))
    sub.add(DVL("dvl", velocity_noise_sigma=0.05))
    sub.add(SurfaceGPS("gps_fore", position_noise_sigma=0.1, transform=(1.0, 0, DECK_Z)))
    sub.add(SurfaceGPS("gps_aft", position_noise_sigma=0.1, transform=(-1.0, 0, DECK_Z)))

    earth = Earth(position=(0, 0, -Earth.R_EQ), waves=WAVES, surface_smoothing=SMOOTH)
    w = World().add_planet(earth)
    w.add_craft(sub, position=(0, 0, -0.2))
    return w, sub


def mix(surge, sway, heave, yaw, pitch):
    """5 DOF commands → 5 thruster throttles (N)."""
    return {
        "surge.throttle": SURGE * surge,
        "h_fore.throttle": SWAY * sway + YAW * yaw,
        "h_aft.throttle": SWAY * sway - YAW * yaw,
        "v_fore.throttle": HEAVE * heave + PITCH * pitch,
        "v_aft.throttle": HEAVE * heave - PITCH * pitch,
    }


def _eta(x, y, t):
    """Wave elevation at (x, y, t) — the same planar sinusoid the Earth uses."""
    k = 2 * np.pi / WAVES.wavelength
    g0 = Earth.MU / Earth.R_EQ**2
    omega = k * np.sqrt(g0 * WAVES.wavelength / (2 * np.pi))
    return WAVES.amplitude * np.cos(k * x - omega * t)


def _heading(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.02
    duration = args.duration or (1e9 if args.keyboard else 48.0)

    w, c = build_world()
    sim = TargetNumpy(Sim(w))
    sim.attach_driver(NoiseDriver(seed=5))

    sensors = ["sub.imu.gyro", "sub.dvl.velocity",
               "sub.gps_fore.position", "sub.gps_aft.position"]
    ekf_ir = EKF(w, sensors=sensors)
    ekf = TargetNumpy(ekf_ir)
    ekf.reset(state={"sub": c.initial_state(position=(0.0, 0.0, -0.2))},
              P=np.eye(ekf.spec.tangent_dim) * 0.1)
    # Always-on inertial sensors + the thruster commands ride the bus; the two
    # GPS are folded by hand, only when their antenna is out of the water.
    wire(sim.out("sub.imu.gyro"), ekf.meas("sub.imu.gyro"))
    wire(sim.out("sub.dvl.velocity"), ekf.meas("sub.dvl.velocity"))

    full = ekf_ir.observability()
    dr = ekf_ir.observability(sensors=["sub.imu.gyro", "sub.dvl.velocity"])
    print(f"observability — surfaced (2 GPS): {full.rank}/{full.tangent_dim}; "
          f"submerged (gyro+DVL): {dr.rank}/{dr.tangent_dim} "
          f"(dead-reckons {', '.join(n.split('.')[-1] for n, _ in dr.unobservable)})\n")

    # Mission: cruise on the surface (GPS lock), dive and run a dog-leg blind,
    # then surface to re-fix. (surge, sway, heave, yaw, pitch)
    script = [(2.0, 6.0, {"w"}),                 # cruise out on the surface (GPS lock)
              (6.0, 10.0, {"shift"}),            # dive setpoint → ~12 m
              (6.0, 36.0, {"w"}),                # long blind transit at depth
              (14.0, 18.0, {"d"}), (24.0, 28.0, {"a"}),   # two blind dog-legs
              (37.0, 43.0, {"space"}),           # surface setpoint → 0, re-fix
              (45.0, 48.0, {"w"})]
    ctrl = make_controller(args.keyboard, script)
    if args.keyboard:
        print("Controls:  W/S surge   A/D yaw   space/shift heave   Q/E sway\n")

    viz = None if args.no_viz else Viz("manta/submarine", addr=args.viz_addr)
    if viz is not None:
        viz.split_cylinder("world/sub/hull", HULL_R, HULL_L,
                           colors=((210, 200, 90), (180, 140, 40)))
        viz.pose("world/sub/hull", (0, 0, 0), (np.cos(np.pi/4), 0, np.sin(np.pi/4), 0))
        viz.box("world/sub_est/hull", (HULL_L, 0.2, 0.2), color=(120, 120, 120, 130))
        viz.plane("world/seabed", z=-25.0, size=80.0, color=(30, 40, 35, 255))

    n = int(duration / dt)
    pacer = Pacer() if (args.keyboard or viz is not None) else None
    # Fly-by-wire: space/shift drive a DEPTH SETPOINT, a depth-hold loop on the
    # (pressure-accurate) depth flies heave to it; surge + yaw are direct. A
    # positive-trim AUV can't hold depth on open-loop thrust.
    depth_sp = -0.2
    print(f"{'t':>5} {'depth':>7} {'mode':>5} {'pos err':>8} {'head err°':>9}")
    try:
        for i in range(n):
            t = i * dt
            if pacer is not None:
                pacer.pace(t)
            ctrl.update(t)
            depth_sp = float(np.clip(
                depth_sp + (ctrl.held("space") - ctrl.held("shift")) * 3.0 * dt,
                -18.0, 0.0))
            z = float(np.asarray(sim.state["sub"]["position"]).ravel()[2])
            vz = float(np.asarray(sim.state["sub"]["velocity"]).ravel()[2])
            heave = float(np.clip(0.4 * (depth_sp - z) - 0.8 * vz, -1.0, 1.0))
            cmd = mix(surge=ctrl.held("w") - ctrl.held("s"),
                      sway=ctrl.held("e") - ctrl.held("q"),
                      heave=heave,
                      yaw=ctrl.held("a") - ctrl.held("d"), pitch=0.0)
            for name, val in cmd.items():
                sim.command(name).set(val)
                ekf.command(name).set(val)

            sim.step(dt)
            # Gated GPS: fold each fix only while its antenna is dry.
            out = sim.outputs()["sub"]
            surfaced = False
            for g in ("gps_fore", "gps_aft"):
                if float(np.asarray(out[f"{g}.wet"]).ravel()[0]) < 0.5:
                    ekf.update(f"sub.{g}.position", out[f"{g}.position"])
                    surfaced = True
            ekf.step(dt)              # fold gyro + DVL, predict forward

            tp = np.asarray(sim.state["sub"]["position"]).ravel()
            tq = np.asarray(sim.state["sub"]["orientation"]).ravel()
            est = ekf.state_dict()["sub"]
            ep = np.asarray(est["position"]).ravel()
            eq = np.asarray(est["orientation"]).ravel()
            if viz is not None and viz.due(t):
                viz.t(t)
                viz.pose("world/sub", tp, tq)
                viz.pose("world/sub_est", ep, eq)
                viz.trail("world/trail", tp, max_len=2000, min_dist=0.1,
                          color=(120, 230, 255) if surfaced else (90, 130, 180))
                # Sub-centred wave surface (same η the physics uses) + a deep
                # seabed plane for depth reference.
                xs = tp[0] + np.linspace(-12, 12, 12)
                ys = tp[1] + np.linspace(-12, 12, 12)
                Xg, Yg = np.meshgrid(xs, ys)
                viz.heightfield("world/waves", Xg, Yg, _eta(Xg, Yg, t),
                                color=(40, 110, 165))
                viz.pose("world/seabed", (tp[0], tp[1], 0.0))
            if (i + 1) % 50 == 0:
                he = abs((_heading(tq) - _heading(eq) + 180) % 360 - 180)
                print(f"{t:>5.1f} {tp[2]:>7.2f} {'GPS' if surfaced else ' DR':>5} "
                      f"{np.linalg.norm(tp - ep):>8.3f} {he:>9.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
