"""Analytic coordinate-test fixtures; not a hardware runner."""
import casadi as ca
from manta import Craft, World
from manta.fields import FluidField, gravity_at
from manta.types import PartFrame, PlanetFrame, WorldFrame, Scalar, Vec3, Wrench
from manta.parts import Part, PartRole, Parameter, State, WhiteNoise, Output, PartUpdate


class Body(Part):
    contributes_inertia = True
    mass = Parameter(20.)
    moi = Parameter((1., 2., 3.), manifold="R3", frame=PartFrame)

    def update(self, ctx):
        gravity = ctx.orientation.conjugate().apply(gravity_at(ctx, ctx.position[WorldFrame]))
        return Wrench(force=gravity*self.mass, torque=Vec3[PartFrame].constant((0, 0, 0)))


class Sensor(Part):
    role = PartRole.SENSOR
    rate = Parameter(None)

    def readings(self, **outputs):
        return PartUpdate(wrench=Wrench.zero(PartFrame), outputs=outputs,
                          rates={name: self.rate for name in outputs})


class IMU(Sensor):
    gyro_noise = WhiteNoise("R3", frame=PartFrame, sigma=1e-4)  # rad/s per acquisition
    accel_noise = WhiteNoise("R3", frame=PartFrame, sigma=.002)  # m/s² per acquisition
    gyro_bias = State((0., 0., 0.), manifold="R3", frame=PartFrame)
    accel_bias = State((0., 0., 0.), manifold="R3", frame=PartFrame)
    gyro = Output()
    accel = Output()

    def update(self, ctx):
        sensor_from_world = ctx.orientation.conjugate()
        gravity = gravity_at(ctx, ctx.position[WorldFrame])
        return self.readings(
            gyro=sensor_from_world.apply(ctx.angular_velocity[WorldFrame])
                 + self.gyro_bias + self.gyro_noise,
            accel=sensor_from_world.apply(ctx.acceleration[WorldFrame]-gravity)
                  + self.accel_bias + self.accel_noise)


class DVL(Sensor):
    velocity_noise = WhiteNoise("R3", frame=PartFrame, sigma=.01)
    velocity = Output()

    def __init__(self, name, *, planet, **kwargs):
        self.planet = planet
        super().__init__(name, **kwargs)

    def update(self, ctx):
        radius = ctx.position[WorldFrame].mx - ca.DM(self.planet.center)
        ground_velocity = ca.cross(ca.DM(self.planet.omega_vec_world()), radius)
        relative = Vec3[WorldFrame].from_mx(ctx.velocity[WorldFrame].mx-ground_velocity)
        return self.readings(velocity=ctx.orientation.conjugate().apply(relative)+self.velocity_noise)


class GPS(Sensor):
    position_noise = WhiteNoise("R3", frame=PlanetFrame, sigma=.5)
    position = Output()

    def __init__(self, name, *, planet, **kwargs):
        self.planet = planet
        super().__init__(name, **kwargs)

    def update(self, ctx):
        radius = ctx.position[WorldFrame].mx - ca.DM(self.planet.center)
        rotation = self.planet.R_world_from_planet_sym(ctx.t.mx)
        return self.readings(position=Vec3[PlanetFrame].from_mx(rotation.T@radius)+self.position_noise)


class Barometer(Sensor):
    requires_fields = [FluidField]
    pressure_noise = WhiteNoise("R1", sigma=20.)  # Pa
    pressure = Output()

    def update(self, ctx):
        # The planet's fluid field supplies the atmosphere/ocean pressure law.
        fluid = ctx.field(FluidField).value_at_sym(ctx.position[WorldFrame], ctx.t)
        return self.readings(pressure=Scalar(fluid.pressure)+self.pressure_noise)


def make_world(planet, initial_state):
    craft = Craft("vehicle")
    craft.add(Body("body"))
    craft.add(IMU("imu", rate=100))
    craft.add(DVL("dvl", planet=planet, rate=5))
    craft.add(GPS("gps", planet=planet, rate=1))
    craft.add(Barometer("baro", rate=10))
    world = World("navigation").add_planet(planet)
    world.add_craft(craft, **initial_state)
    return world

import numpy as np
from manta.ir._rotation import quat_from_rotmat_np

def trajectory(planet, t, speed=2.):
    """Independent circular-path oracle in inertial Cartesian coordinates."""
    latitude = np.deg2rad(45.)
    radius = planet.R_EQ + 10.  # Air track: GPS and barometer are both available.
    relative_rate = speed/(radius*np.cos(latitude))
    longitude = np.deg2rad(179.9999) + (planet.omega+relative_rate)*t
    s, c = np.sin(longitude), np.cos(longitude)
    up = np.array([np.cos(latitude)*c, np.cos(latitude)*s, np.sin(latitude)])
    east = np.array([-s, c, 0.])
    north = np.cross(up, east)
    rotation = np.column_stack((east, north, up))
    p = radius*up
    omega = np.array([0., 0., planet.omega+relative_rate])
    v = np.cross(omega, p)
    a = np.cross(omega, v)
    return dict(position=p, velocity=v, orientation=quat_from_rotmat_np(rotation),
                angular_velocity=rotation.T@omega), rotation, a
