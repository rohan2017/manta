"""Global coordinates and user-defined sensor models; no local-frame resets."""

import casadi as ca
import numpy as np
import pytest
from manta import INS, EKF, UKF, IMUPreintegrator, PreintegratedIMUPacket, TargetNumpy
from manta.fields import GravityField, FluidField
from manta.ir import Vec3, WorldFrame
from manta.planets import Earth
from tests._ins_global_models import make_world, IMU as UserIMU
from tests._ins_global_models import trajectory


@pytest.mark.parametrize("estimator", [INS, EKF, UKF])
@pytest.mark.parametrize("at_pole", [False, True])
def test_same_runtime_and_global_sensor_models_at_nonzero_epoch(estimator, at_pole):
    planet = Earth(flattening=0.0, include_j2=False)
    t = 3600.0
    truth, R, acceleration = trajectory(planet, t)
    if at_pole:
        R = np.eye(3)
        truth = dict(
            position=np.array([0.0, 0.0, planet.R_EQ + 10]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=planet.omega_vec_world(),
        )
    world = make_world(planet, truth)
    # World initial-state conversion happens at t=0; these are already global
    # inertial coordinates at the desired epoch, so no scene conversion occurs.
    options = {"imu": "imu"} if estimator is INS else {}
    runtime = TargetNumpy(
        estimator(
            world, sensors=["dvl.velocity", "gps.position", "baro.pressure"], **options
        )
    )
    runtime.reset(t=t, P=np.eye(runtime.spec.tangent_dim) * 1e-8)
    position = truth["position"]
    dvl = R.T @ (truth["velocity"] - np.cross(planet.omega_vec_world(), position))
    gps = planet.R_world_from_planet(t).T @ position
    resolved = world.snapshot()
    pressure = (
        resolved.get_field(FluidField)
        .value_at_sym(Vec3[WorldFrame].constant(position), ca.DM(t))
        .pressure
    )
    # EKF/INS evaluate at the nominal; UKF's curved pressure mean is allowed
    # its small second-order prior correction.
    for name, value in [
        ("dvl.velocity", dvl),
        ("gps.position", gps),
        ("baro.pressure", float(ca.evalf(pressure))),
    ]:
        result = runtime.update(name, value)
        np.testing.assert_allclose(result.innovation, 0.0, atol=2e-6)
        assert result.accepted
    from manta.parts import IMU as BuiltinIMU

    assert not issubclass(UserIMU, BuiltinIMU)


class SmallEarth(Earth):
    R_EQ = 10.0


def circuit(rate):
    # A weak-gravity tiny planet permits a short full circuit without the
    # extreme tidal instability of putting Earth surface gravity on a 10 m
    # body. This is a coordinate/integration oracle, not Earth qualification.
    planet = SmallEarth(
        flattening=0.0, include_j2=False, rotation_rate=0.02, gravity_mu=4.0
    )
    initial, _, _ = trajectory(planet, 0.0)
    world = make_world(planet, initial)
    ins = TargetNumpy(
        INS(
            world,
            imu="imu",
            sensors=[],
            propagation="preintegrated",
            accel_integration="endpoint",
            covariance="conventional_reset",
        )
    )
    prior = ins.module.port("prior_x").spec
    ins.reset(P=np.zeros((prior.tangent_dim, prior.tangent_dim)))
    pre = TargetNumpy(
        IMUPreintegrator(
            gyro_integration="endpoint",
            accel_integration="endpoint",
            sample_rate_hz=rate,
        )
    )
    resolved = world.snapshot()
    gravity = resolved.get_field(GravityField)
    p = ca.MX.sym("p", 3)
    g = ca.Function(
        "gravity", [p], [gravity.value_at_sym(Vec3[WorldFrame].from_mx(p), ca.MX(0)).mx]
    )
    # Exactly one revolution relative to the planet, including a dateline crossing.
    duration = 2 * np.pi * (planet.R_EQ + 10) * np.cos(np.pi / 4) / 2
    steps = round(duration * rate)
    h = duration / steps

    def sample(t):
        truth, R, a = trajectory(planet, t)
        return (
            R.T @ (a - np.asarray(g(truth["position"])).ravel()),
            truth["angular_velocity"],
            truth,
        )

    left = sample(0.0)
    for k in range(steps):
        right = sample((k + 1) * h)
        values = pre.step(
            h,
            accel=left[0],
            gyro=left[1],
            end_accel=right[0],
            end_gyro=right[1],
            accel_bias=[0, 0, 0],
            gyro_bias=[0, 0, 0],
        )
        ins.predict(PreintegratedIMUPacket(values, start_time=k * h))
        pre.reset()
        left = right
    final = ins.state_dict()["vehicle"]
    # Quaternion sign is arbitrary; compare the physical rotation.
    assert abs(np.dot(final["orientation"], right[2]["orientation"])) > 1 - 1e-10
    return np.linalg.norm(final["position"] - right[2]["position"])


def test_complete_circumnavigation_converges_without_reframing():
    coarse, fine = circuit(25), circuit(50)
    print("circumnavigation position errors", coarse, fine)
    assert fine < 0.6 * coarse
    assert fine < 0.01


def test_user_imu_without_bias_states_or_rate_supports_model_force():
    from manta import Craft, World
    from manta.ir import PartFrame, Wrench
    from manta.parts import Part, Output, WhiteNoise, PartUpdate, ModelForce
    from tests._ins_global_models import Body
    from tests.test_ins import _evidence

    class BareIMU(Part):
        gyro = Output()
        accel = Output()
        gyro_noise = WhiteNoise('R3', frame=PartFrame, sigma=.001)
        accel_noise = WhiteNoise('R3', frame=PartFrame, sigma=.01)

        def update(self, ctx):
            from manta.fields import gravity_at
            to_sensor = ctx.orientation.conjugate()
            return PartUpdate(wrench=Wrench.zero(PartFrame), outputs={
                'gyro': to_sensor.apply(ctx.angular_velocity[WorldFrame])+self.gyro_noise,
                'accel': to_sensor.apply(ctx.acceleration[WorldFrame]-gravity_at(ctx,ctx.position[WorldFrame]))+self.accel_noise})

    craft = Craft('craft')
    craft.add(Body('body'))
    imu = BareIMU('imu')
    craft.add(imu)
    craft.add(ModelForce('force',imu=imu,evidence=_evidence()))
    world = World().add_field(GravityField(g=(0,0,-9.81)))
    world.add_craft(craft)
    ins = TargetNumpy(INS(world,imu=imu,sensors=['force.specific_force']))
    assert ins.update('force.specific_force',[0,0,0]).accepted
