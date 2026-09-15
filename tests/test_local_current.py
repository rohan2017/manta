"""Local fluid-state physics, filter state closure, and gyrocompass symmetry."""

import numpy as np
import pytest
from manta.fields import FluidField, GravityField, LocalCurrent
from manta.parts import ConstantBiasIMU, DragSurface, Mass, ModelForce, VelocitySensor

from manta import INS, Craft, NavigationFrame, Sim, TargetNumpy, World
from tests.test_ins import _evidence


def world(sigma=0.0, *, drag_force=(-1., -2., -3.)):
    craft = Craft("craft")
    craft.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    craft.add(DragSurface("drag", force=drag_force))
    imu = ConstantBiasIMU("imu", accel_noise_sigma=1e-4, gyro_noise_sigma=1e-6)
    craft.add(imu)
    craft.add(VelocitySensor("dvl", velocity_noise_sigma=0.001))
    craft.add(ModelForce("model_force", imu=imu, evidence=_evidence(white_sigma=0.1)))
    w = (
        World()
        .add_field(GravityField(g=(0, 0, -9.81)))
        .add_field(FluidField(density=1).add(LocalCurrent(sigma=sigma)))
    )
    w.add_craft(craft)
    return w


def test_zero_diffusion_retains_current_and_changes_drag_without_advecting_position():
    sim = TargetNumpy(Sim(world()))
    sim.state["local_current"]["velocity"] = np.array((0.2, -0.1, 0.05))
    sim.step(0.01)
    np.testing.assert_array_equal(
        sim.state["local_current"]["velocity"], (0.2, -0.1, 0.05)
    )
    np.testing.assert_allclose(
        sim.state["craft"]["velocity"], (0.002, -0.002, -0.0966), atol=1e-10
    )
    # Starting ground velocity is zero: no direct +current position advection.
    np.testing.assert_allclose(
        sim.state["craft"]["position"][:2], (0.00001, -0.00001), atol=1e-10
    )


@pytest.mark.parametrize("covariance", ["linearized", "geometric", "nonlinear"])
def test_current_model_preserves_stationary_heading_bias_family(covariance):
    from examples.qualification.earth_ins_debug import (
        heading_tangent,
        stationary_audit,
        stationary_input,
        stationary_state,
    )

    spin = 7.2921159e-5
    ins = INS(
        world(0.01),
        imu="imu",
        sensors=["dvl.velocity", "model_force.specific_force"],
        covariance=covariance,
        navigation_frame=NavigationFrame(
            frame_id="test",
            epoch="1",
            angular_velocity=(0, spin / 2**0.5, spin / 2**0.5),
            origin_from_rotation_center=(0, 0, 6378137),
            gravity_convention="effective",
        ),
    )
    assert "local_current.velocity" in ins.spec
    audit = stationary_audit(ins)
    for row in audit["family"]:
        assert row["F_direction_max_error"] < 2e-12
        assert row["H_direction_max_error"] < 2e-12
    # The new force aid must not invent information along the stationary
    # heading/gyro-bias ambiguity. This complements the long gyrocompass suite.
    x = stationary_state(ins, 5)
    direction = heading_tangent(ins, x)
    h = np.asarray(
        ins.sys.sensors["craft.model_force.specific_force"].H_fn(
            x, stationary_input(ins), 0, 0
        )
    )
    np.testing.assert_allclose(h @ direction, 0, atol=2e-12)


def test_local_current_does_not_bypass_model_force_noise_ratio_gate():
    with pytest.raises(ValueError, match="rho"):
        w = world(0.01)
        imu = next(p for p in w.crafts[0].parts if p.name == "imu")
        imu.accel_noise_sigma = 1.0
        INS(w, imu="imu", sensors=["model_force.specific_force"])


@pytest.mark.parametrize("covariance", ["linearized", "geometric", "nonlinear"])
def test_current_aiding_retains_earth_rate_gyrocompassing(covariance):
    """Noiseless convergence check with force aiding active, not an ANEES claim.

    The force-error fixture is synthetic, not deployment fit evidence. A
    neutral buoyancy force supplies the accelerometer's stationary gravity
    reading; heading comes only from Earth rate and the tight gyro-bias prior.
    """
    import casadi as ca
    from manta.ir._rotation import quat_to_rotmat, so3_exp
    from manta.parts import PointBuoy

    w = world(0.01)
    w.crafts[0].add(PointBuoy("buoy", volume=1.0))
    spin = 7.2921159e-5
    frame = NavigationFrame(
        frame_id="earth",
        epoch="1",
        angular_velocity=(0, spin / 2**0.5, spin / 2**0.5),
        origin_from_rotation_center=(0, 0, 6378137),
        gravity_convention="effective",
    )
    ins = INS(
        w,
        imu="imu",
        covariance=covariance,
        navigation_frame=frame,
        sensors=["dvl.velocity", "model_force.specific_force"],
        gates=None,
    )
    module = ins.module()
    prior_spec = (
        module.port("prior_x").spec
        if "initialize_prior" in module.functions
        else ins.spec
    )
    p = np.eye(prior_spec.tangent_dim) * 0.04
    for slot in prior_spec.slots:
        sigma = (
            1e-8
            if slot.name.endswith("gyro_bias")
            else 1e-5
            if slot.name.endswith("accel_bias")
            else 0.001
            if slot.name.endswith("velocity") and slot.name.startswith("craft.")
            else None
        )
        if sigma is not None:
            p[
                slot.tangent_offset : slot.tangent_offset + 3,
                slot.tangent_offset : slot.tangent_offset + 3,
            ] = np.eye(3) * sigma**2
    o = prior_spec.slot("craft.orientation").tangent_offset
    p[o : o + 3, o : o + 3] = np.diag(np.radians((0.1, 0.1, 5.0)) ** 2)
    runtime = TargetNumpy(ins)
    runtime.reset(
        state={
            "craft": {
                "orientation": np.asarray(so3_exp(ca.DM([0, 0, np.radians(5)]))).ravel()
            }
        },
        P=p,
    )
    u = ins.sys.u_defaults.copy()
    u[ins.sys._input_slices["craft.imu.accel"]] = (0, 0, 9.81)
    u[ins.sys._input_slices["craft.imu.gyro"]] = frame.angular_velocity
    n, na = ins.spec.tangent_dim, ins.spec.ambient_dim
    xp = ca.MX.sym("xp", na + n * n)
    x, P = xp[:na], ca.reshape(xp[na:], n, n)
    for k in range(4):
        x, P = module.functions["predict"](x, P, u, 0.01, 0)
    x, P = module.functions["update_craft_dvl_velocity"](x, P, np.zeros(3), u, 0)
    x, P = module.functions["update_craft_model_force_specific_force"](
        x, P, (0, 0, 9.81), u, 0
    )
    step = ca.Function("current_gyrocompass", [xp], [ca.vertcat(x, ca.vec(P))])
    result = np.asarray(
        step.fold(7500)(np.r_[runtime.x, runtime.P.ravel(order="F")])
    ).ravel()
    orientation = ins.spec.slot("craft.orientation")
    q = result[orientation.ambient_offset : orientation.ambient_offset + 4]
    r = np.asarray(quat_to_rotmat(q))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    final_p = result[na:].reshape((n, n), order="F")
    assert abs(np.degrees(yaw)) < 0.5
    assert np.linalg.eigvalsh(final_p).min() >= -1e-12
    assert np.sqrt(
        final_p[orientation.tangent_offset + 2, orientation.tangent_offset + 2]
    ) < np.radians(1.0)


def test_current_with_negligible_drag_sensitivity_keeps_its_uncertainty():
    w = world(.1, drag_force=(-1e-10, -1e-10, -1e-10))
    ins = INS(
        w,
        imu="imu",
        sensors=["dvl.velocity", "model_force.specific_force"],
    )
    runtime = TargetNumpy(ins)
    p = np.eye(ins.spec.tangent_dim) * 0.04
    runtime.reset(P=p)
    slot = ins.spec.slot("local_current.velocity")
    o = slot.tangent_offset
    for sensor in ins.sys.sensors.values():
        h = np.asarray(sensor.H_fn(runtime.x, ins.sys.u_defaults, 0, 0))
        np.testing.assert_allclose(h[:, o : o + 3], 0.0, atol=1e-9)
    for i in range(10):
        runtime.predict(
            0.01, u={"craft.imu.accel": (0, 0, 9.81), "craft.imu.gyro": (0, 0, 0)}
        )
        runtime.update("craft.dvl.velocity", (0, 0, 0))
        runtime.update("craft.model_force.specific_force", (0, 0, 0))
    np.testing.assert_allclose(
        runtime.P[o : o + 3, o : o + 3], np.eye(3) * 0.041, atol=1e-12
    )
