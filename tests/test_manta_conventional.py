"""Independent finite perturbations of the selectable Manta error model."""

import numpy as np
from tests._rotation_oracle import Rotation
from manta import INS, Craft, World, NavigationFrame
from manta.parts import Mass, ConstantBiasIMU, VelocitySensor
from manta.fields import GravityField


def test_conventional_discrete_jacobian_uses_its_velocity_coordinates():
    c = Craft("c")
    c.add(Mass("mass", mass=1, moi=(1, 1, 1)))
    c.add(ConstantBiasIMU("imu", gyro_noise_sigma=0.001, accel_noise_sigma=0.01))
    c.add(VelocitySensor("dvl", velocity_noise_sigma=0.01))
    world = World("conventional_derivative").add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(
        c,
        velocity=(0.7, -0.2, 0.1),
        orientation=Rotation.from_rotvec([0.2, -0.1, 0.6]).as_quat(scalar_first=True),
    )
    f = INS(
        world,
        imu="imu",
        sensors=["dvl.velocity"],
        covariance="conventional",
        navigation_frame=NavigationFrame(
            frame_id="test",
            epoch="1",
            angular_velocity=(0, 5e-5, 5e-5),
            origin_from_rotation_center=(0, 0, 6378137),
            gravity_convention="effective",
        ),
    )
    spec = f.spec
    x = np.asarray(f.module().state.field("x").init).copy()
    n = spec.tangent_dim

    def slots(value, name):
        s = spec.slot("c." + name)
        return value[s.ambient_offset : s.ambient_offset + s.ambient_dim]

    def perturb(value, d):
        out = value.copy()
        si = spec.slot("c.orientation").tangent_offset
        theta = d[si : si + 3]
        D = Rotation.from_rotvec(theta)
        for slot in spec.slots:
            a, t = slot.ambient_offset, slot.tangent_offset
            if slot.name.endswith(".orientation"):
                out[a : a + 4] = (
                    D * Rotation.from_quat(value[a : a + 4], scalar_first=True)
                ).as_quat(scalar_first=True)
            elif slot.name == "c.velocity":
                out[a : a + 3] = D.apply(value[a : a + 3] + d[t : t + 3])
            else:
                out[a : a + slot.ambient_dim] += d[t : t + slot.tangent_dim]
        return out

    def difference(value, base):
        D = (
            Rotation.from_quat(slots(value, "orientation"), scalar_first=True)
            * Rotation.from_quat(slots(base, "orientation"), scalar_first=True).inv()
        )
        out = np.zeros(n)
        for slot in spec.slots:
            a, t = slot.ambient_offset, slot.tangent_offset
            if slot.name == "c.orientation":
                out[t : t + 3] = D.as_rotvec()
            elif slot.name == "c.velocity":
                out[t : t + 3] = D.inv().apply(value[a : a + 3]) - base[a : a + 3]
            else:
                out[t : t + slot.tangent_dim] = (
                    value[a : a + slot.ambient_dim] - base[a : a + slot.ambient_dim]
                )
        return out

    u = f.sys.resolve_u(
        {"imu.accel": [0.2, -0.1, 9.7], "imu.gyro": [0.03, -0.04, 0.07]}
    )
    nominal = np.asarray(f.sys.predict_fn(x, u, 0.01, 0)).ravel()
    eps = 1e-6
    numerical = []
    for d in np.eye(n) * eps:
        plus = np.asarray(f.sys.predict_fn(perturb(x, d), u, 0.01, 0)).ravel()
        minus = np.asarray(f.sys.predict_fn(perturb(x, -d), u, 0.01, 0)).ravel()
        numerical.append(
            (difference(plus, nominal) - difference(minus, nominal)) / (2 * eps)
        )
    np.testing.assert_allclose(
        f.sys.F_fn(x, u, 0.01, 0), np.asarray(numerical).T, atol=2e-9, rtol=1e-6
    )
    d = np.linspace(-0.01, 0.01, n)
    np.testing.assert_allclose(spec.boxplus_num(x, d), perturb(x, d), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(spec.minus(perturb(x, d), x)).ravel(), d, atol=1e-14
    )
