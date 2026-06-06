"""F-discretization modes: "exact" vs "euler" (continuous-time F).

"euler" builds F = I + dt·M with M = ∂²δ'/∂dt∂δ at (δ=0, dt=0) — the
continuous-time linearization, Euler-discretized. It must agree with the
exact discrete jacobian to O(dt²) INCLUDING the manifold transport terms
(both come from the same boxminus recipe), while generating ~5× less
deploy code (the dt→0 fold prunes the discrete integrator — both
joint-space factorizations, the momentum recovery, Rodrigues — out of
the differentiated graph).

The test craft is deliberately the nastiest available shape: tumbling
body + spinning reaction wheel + displaced prismatic slider, so F is
dense with gyroscopic, joint-coupling, and configuration-dependent
terms.
"""

import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.estimation import EKF
from manta.fields import GravityField
from manta.linearized_system import LinearizedSystem
from manta.parts import (
    IMU, Mass, PositionSensor, PrismaticJoint, RevoluteJoint, Thruster,
)


def _rover():
    c = Craft("rover")
    c.add(Mass("body", mass=4.0, moi=(0.4, 0.5, 0.6)))
    c.add(Thruster("t", force=(0.0, 0.0, 1.0)))
    c.add(IMU("imu"))
    c.add(PositionSensor("gps"))
    wheel = RevoluteJoint("wheel", mode="saturating", stall_torque=0.5,
                          axis=(0.0, 0.0, 1.0))
    wheel.add(Mass("disk", mass=0.3, moi=(0.004, 0.006, 0.008)))
    c.add(wheel)
    arm = PrismaticJoint("arm", mode="saturating", stall_force=2.0,
                         axis=(1.0, 0.0, 0.0))
    arm.add(Mass("tip", mass=0.2, moi=(1e-4, 1e-4, 1e-4)))
    c.add(arm)
    w = World()
    w.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    w.add_craft(c)
    return w


_STATE = {
    "rover.position":         np.array([0.0, 0.0, 5.0]),
    "rover.orientation":      np.array([1.0, 0.0, 0.0, 0.0]),
    "rover.velocity":         np.array([0.5, 0.0, 0.0]),
    "rover.angular_velocity": np.array([0.3, -0.2, 0.5]),
    "rover.wheel.angle":      0.0,
    "rover.wheel.rate":       40.0,
    "rover.arm.displacement": 0.15,
    "rover.arm.rate":         0.0,
}
_INPUTS = {
    "rover.t.throttle":       4.5 * 9.81,
    "rover.wheel.torque_cmd": 0.2,
    "rover.arm.force_cmd":    0.8,
}


def _xu(sys):
    x = sys.spec.pack(_STATE)
    u = np.array([_INPUTS[n] for n in sys.input_names])
    return x, u


def test_euler_f_converges_quadratically_to_exact():
    sx = LinearizedSystem(_rover())
    se = LinearizedSystem(_rover(), discretization="euler")
    x, u = _xu(sx)

    errs = []
    for dt in (2e-3, 1e-3, 5e-4):
        Fx = np.asarray(sx.F_fn(x, u, dt, 0.0))
        Fe = np.asarray(se.F_fn(x, u, dt, 0.0))
        errs.append(np.abs(Fe - Fx).max())
    # O(dt²): each halving of dt quarters the gap.
    assert errs[0] < 1e-4
    assert 3.0 < errs[0] / errs[1] < 5.0, errs
    assert 3.0 < errs[1] / errs[2] < 5.0, errs
    # And at dt → 0 both converge on the same continuous linearization.
    assert errs[2] < 1e-5


def test_euler_mode_h_and_predict_unchanged():
    """The discretization only touches F: predict and the sensor models
    (the euler mode inlines them, but the math is identical)."""
    sx = LinearizedSystem(_rover())
    se = LinearizedSystem(_rover(), discretization="euler")
    x, u = _xu(sx)
    dt = 1e-3

    np.testing.assert_allclose(
        np.asarray(se.predict_fn(x, u, dt, 0.0)),
        np.asarray(sx.predict_fn(x, u, dt, 0.0)), atol=1e-14)
    assert set(se.sensors) == set(sx.sensors)
    for full in sx.sensors:
        np.testing.assert_allclose(
            np.asarray(se.sensors[full].h_fn(x, u, dt, 0.0)),
            np.asarray(sx.sensors[full].h_fn(x, u, dt, 0.0)),
            atol=1e-12, err_msg=full)
        np.testing.assert_allclose(
            np.asarray(se.sensors[full].H_fn(x, u, dt, 0.0)),
            np.asarray(sx.sensors[full].H_fn(x, u, dt, 0.0)),
            atol=1e-10, err_msg=full)


def test_ekf_euler_covariance_tracks_exact():
    """Two EKFs over the same articulated truth, one per discretization:
    the propagated covariances stay within the O(dt) integration tier of
    each other (the means are identical — predict is shared)."""
    def run(mode):
        # predict-only comparison — no sensors needed (and the default
        # noiseless IMU would be rejected by the zero-R guard anyway).
        ekf = TargetNumpy(EKF(_rover(), sensors=[], discretization=mode))
        nested = {"rover": {k.split(".", 1)[1]: v
                            for k, v in _STATE.items()}}
        for k, v in _INPUTS.items():
            nested["rover"][k.split(".", 1)[1]] = v
        ekf.reset(state=nested, P=np.eye(ekf.spec.tangent_dim) * 0.1)
        Q = np.eye(ekf.spec.tangent_dim) * 1e-9
        for _ in range(100):
            ekf.predict(dt=1e-3, Q=Q)
        return ekf

    e_x = run("exact")
    e_e = run("euler")
    np.testing.assert_allclose(e_e.x, e_x.x, atol=1e-12)   # shared predict
    dP = np.abs(e_e.P - e_x.P).max() / np.abs(e_x.P).max()
    assert dP < 1e-3, f"relative covariance gap {dP:.2e}"


def test_invalid_discretization_raises():
    with pytest.raises(ValueError):
        LinearizedSystem(_rover(), discretization="rk4")
    with pytest.raises(ValueError):
        Sim(_rover(), discretization="midpoint")
