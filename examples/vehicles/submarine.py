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

from manta import Craft, EKF, NoiseDriver, Sim, TargetNumpy, World
from manta.fields import FluidField
from manta.ir.frames import WorldFrame
from manta.ir.types import Scalar
from manta.parts import (
    DragSurface, IMU, Mass, Output, PointBuoy, PositionSensor, Thruster,
    VelocitySensor,
)
from manta.planets import Earth, SeaWaves

from .._control import Pacer, common_args, make_controller
from .._geom import wave_elevation, yaw
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

# --- Earth + scene ----------------------------------------------------------
# The world frame is INERTIAL and the Earth spins in it (sidereal by default),
# so the IMU gyro (= angular_velocity[WorldFrame]) reads the body's INERTIAL
# rate with NO Earth-rate term in the part model: a sub co-rotating with the
# Earth senses the planet's spin. We anchor a local `Scene` at the north pole
# and work in its small ENU coordinates (placement, depth-hold, rendering all
# scene-relative), leaving the planet at its true scale at the world origin.
# At the pole the spin axis is local up, so the gyro reads Ω on its vertical
# axis AND the co-rotating ocean is locally still (Ω×r = 0 on the axis). A
# mid-latitude scene would put a NORTH component on the gyro — the gyrocompass
# signal — but the co-rotating ocean would then sweep past at Ω·R·cos(lat).
ANCHOR = (0.0, 0.0, Earth.R_EQ)       # north-pole surface point (planet frame)
START = (0.0, 0.0, -0.2)              # scene-local: 0.2 m below the surface

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

    requires_fields = [FluidField]

    wet = Output()

    def update(self, ctx):
        out = super().update(ctx)
        rho = ctx.field(FluidField).value_at_sym(
            ctx.position[WorldFrame], ctx.t).density
        out.outputs["wet"] = Scalar(rho * (1.0 / RHO))
        out.rates["wet"] = self.rate
        return out


def build_world():
    sub = Craft("sub")
    # Cylinder: long in x. Roll MoI (about x) ≪ pitch/yaw (about y, z); CoM
    # below the buoy line for pendulum stability in roll + pitch.
    sub.add(Mass("hull", mass=MASS, moi=(5.0, 60.0, 60.0), mount_offset=(0, 0, COM_Z)))

    # A line of buoys along the hull axis (z = 0): rides the wave profile when
    # surfaced. Co-located axial drag fades with submersion the same way.
    xs = np.linspace(-1.2, 1.2, N_BUOY)
    v_each = BUOY_FAC * MASS / RHO / N_BUOY
    for i, x in enumerate(xs):
        sub.add(PointBuoy(f"buoy{i}", volume=v_each, mount_offset=(float(x), 0, 0)))
        sub.add(DragSurface.isotropic_quadratic(f"hd{i}", area=0.05,
                                                drag_coefficient=0.8,
                                                mount_offset=(float(x), 0, 0)))
    # Top/bottom panels at three stations: these are OFF the roll axis, so
    # they damp roll + pitch rate (the axial line above cannot).
    for j, x in enumerate((1.0, 0.0, -1.0)):
        sub.add(DragSurface.isotropic_quadratic(f"top{j}", area=0.07,
                                                drag_coefficient=1.0,
                                                mount_offset=(x, 0, HULL_R)))
        sub.add(DragSurface.isotropic_quadratic(f"bot{j}", area=0.07,
                                                drag_coefficient=1.0,
                                                mount_offset=(x, 0, -HULL_R)))

    # 5-thruster vector layout.
    sub.add(Thruster("surge", force=(1, 0, 0), mount_offset=(-1.2, 0, 0),
                     force_noise_sigma=3.0))
    sub.add(Thruster("h_fore", force=(0, 1, 0), mount_offset=(1.0, 0, 0)))
    sub.add(Thruster("h_aft", force=(0, 1, 0), mount_offset=(-1.0, 0, 0)))
    sub.add(Thruster("v_fore", force=(0, 0, 1), mount_offset=(1.0, 0, 0)))
    sub.add(Thruster("v_aft", force=(0, 0, 1), mount_offset=(-1.0, 0, 0)))

    # Sensors: IMU (gyro carries the body's inertial rate, so on the spinning
    # Earth it reads Ω) + DVL (dead-reckoning, modeled with the simple
    # VelocitySensor) + two water-gated GPS (fore/aft).
    sub.add(IMU("imu", gyro_noise_sigma=0.01, gyro_bias_sigma=5e-4))
    sub.add(VelocitySensor("dvl", velocity_noise_sigma=0.05))
    sub.add(SurfaceGPS("gps_fore", position_noise_sigma=0.1, mount_offset=(1.0, 0, DECK_Z)))
    sub.add(SurfaceGPS("gps_aft", position_noise_sigma=0.1, mount_offset=(-1.0, 0, DECK_Z)))

    earth = Earth(waves=WAVES, surface_smoothing=SMOOTH)
    w = World().add_planet(earth)
    scene = earth.scene_at(ANCHOR)
    # Co-rotating with the Earth: the IMU then reads Earth's spin (a real
    # Earth-fixed sub does). Over the run this turns the sub ~0.2°, which the
    # filter tracks; the point is the steady inertial rate on the gyro.
    # `scene.at_rest` supplies the body spin rate + orbital velocity and
    # places the sub by its small scene-local coordinate.
    w.add_craft(sub, **scene.at_rest(START))
    return w, sub, earth, scene


def mix(surge, sway, heave, yaw, pitch):
    """5 DOF commands → 5 thruster throttles (N)."""
    return {
        "surge.throttle": SURGE * surge,
        "h_fore.throttle": SWAY * sway + YAW * yaw,
        "h_aft.throttle": SWAY * sway - YAW * yaw,
        "v_fore.throttle": HEAVE * heave + PITCH * pitch,
        "v_aft.throttle": HEAVE * heave - PITCH * pitch,
    }


def main() -> None:
    args = common_args(__doc__).parse_args()
    dt = 0.02
    duration = args.duration or (1e9 if args.keyboard else 48.0)

    w, c, earth, scene = build_world()
    sim = TargetNumpy(Sim(w))
    # The IMU gyro reads angular_velocity[WorldFrame] — the body's INERTIAL
    # rate — so co-rotating with the spinning Earth it senses Earth's spin with
    # NO Earth-rate term in the part model. Confirm it on a noiseless step.
    sim.step(1e-3)
    g0 = np.asarray(sim.outputs()["sub"]["imu.gyro"]).ravel()
    print(f"IMU senses Earth's spin (inertial rate, no part code): gyro = "
          f"{np.array2string(g0, precision=2)} rad/s, |gyro| = "
          f"{np.linalg.norm(g0):.2e} vs Ω = {Earth.SIDEREAL:.2e}")
    sim.attach_driver(NoiseDriver(seed=5))

    sensors = ["sub.imu.gyro", "sub.dvl.velocity",
               "sub.gps_fore.position", "sub.gps_aft.position"]
    ekf_ir = EKF(w, sensors=sensors)
    ekf = TargetNumpy(ekf_ir)
    ekf.reset(state={"sub": c.initial_state(**scene.at_rest(START))},
              P=np.eye(ekf.spec.tangent_dim) * 0.1)
    # Always-on inertial sensors fed every step; the two GPS are folded by
    # hand, only when their antenna is out of the water.
    INERTIAL = ["sub.imu.gyro", "sub.dvl.velocity"]

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

    # Everything lives under `world/scene` — the local ENU frame at the pole.
    # The scene's (planet-radius) world pose is logged on that entity each
    # frame; children stay in small scene coordinates and the view tracks it.
    viz = None if args.no_viz else Viz("manta/submarine", addr=args.viz_addr)
    if viz is not None:
        viz.split_cylinder("world/scene/sub/hull", HULL_R, HULL_L,
                           colors=((210, 200, 90), (180, 140, 40)))
        viz.pose("world/scene/sub/hull", (0, 0, 0),
                 (np.cos(np.pi/4), 0, np.sin(np.pi/4), 0))
        viz.box("world/scene/sub_est/hull", (HULL_L, 0.2, 0.2),
                color=(120, 120, 120, 130))
        viz.plane("world/scene/seabed", z=-25.0, size=80.0, color=(30, 40, 35, 255))
        viz.track("world/scene")

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
            pre = scene.relative(sim.state["sub"], t)   # depth-hold in scene coords
            z = float(pre["position"][2])
            vz = float(pre["velocity"][2])
            heave = float(np.clip(0.4 * (depth_sp - z) - 0.8 * vz, -1.0, 1.0))
            cmd = mix(surge=ctrl.held("w") - ctrl.held("s"),
                      sway=ctrl.held("e") - ctrl.held("q"),
                      heave=heave,
                      yaw=ctrl.held("a") - ctrl.held("d"), pitch=0.0)
            sim.step(dt, u=cmd)
            out = sim.outputs()["sub"]
            for nm in INERTIAL:               # always-on inertial sensors
                ekf.update(nm, sim.reading(nm), u=cmd)
            # Gated GPS: fold each fix only while its antenna is dry.
            surfaced = False
            for g in ("gps_fore", "gps_aft"):
                if float(np.asarray(out[f"{g}.wet"]).ravel()[0]) < 0.5:
                    ekf.update(f"sub.{g}.position", out[f"{g}.position"], u=cmd)
                    surfaced = True
            ekf.predict(dt, u=cmd)   # then predict forward (you own the order)

            trel = scene.relative(sim.state["sub"], t)
            erel = scene.relative(ekf.state_dict()["sub"], t)
            tp = np.asarray(trel["position"]); tq = np.asarray(trel["orientation"])
            ep = np.asarray(erel["position"]); eq = np.asarray(erel["orientation"])
            if viz is not None and viz.due(t):
                viz.t(t)
                viz.pose("world/scene", *scene.world_pose(t))
                viz.pose("world/scene/sub", tp, tq)
                viz.pose("world/scene/sub_est", ep, eq)
                viz.trail("world/scene/trail", tp, max_len=2000, min_dist=0.1,
                          color=(120, 230, 255) if surfaced else (90, 130, 180))
                # Sub-centred wave surface (same η the physics uses, in scene
                # coords) + a deep seabed plane for depth reference.
                xs = tp[0] + np.linspace(-12, 12, 12)
                ys = tp[1] + np.linspace(-12, 12, 12)
                Xg, Yg = np.meshgrid(xs, ys)
                viz.heightfield("world/scene/waves", Xg, Yg,
                                wave_elevation(Xg, t, WAVES),
                                color=(40, 110, 165))
                viz.pose("world/scene/seabed", (tp[0], tp[1], 0.0))
            if (i + 1) % 50 == 0:
                he = abs((np.degrees(yaw(tq)) - np.degrees(yaw(eq))
                          + 180) % 360 - 180)
                print(f"{t:>5.1f} {tp[2]:>7.2f} {'GPS' if surfaced else ' DR':>5} "
                      f"{np.linalg.norm(tp - ep):>8.3f} {he:>9.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
