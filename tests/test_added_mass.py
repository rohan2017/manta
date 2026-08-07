"""AddedMass — the entrained fluid's inertia, against hand physics.

Every case is chosen so the expected number is computable on paper:
one-step impulses at rest (no drag, no gravity), still water, COM at
the origin.
"""

import numpy as np

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import FluidField, GravityField
from manta.parts import AddedMass, Mass, Thruster

DT = 1e-3


def _world():
    return (World()
            .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
            .add_field(FluidField().add_uniform(density=1000.0)))


def _craft(name="hull", *, added=None, thruster_axis=None,
           thruster_offset=(0.0, 0.0, 0.0), moi=(2.0, 2.0, 2.0)):
    c = Craft(name)
    c.add(Mass("core", mass=10.0, moi=moi))
    if added is not None:
        c.add(added)
    if thruster_axis is not None:
        c.add(Thruster("t", force=thruster_axis,
                       mount_offset=thruster_offset))
    return c


def _step(world, name, *, throttle=None, **craft_state):
    sim = TargetNumpy(Sim(world))
    for key, value in craft_state.items():
        sim.state[name][key] = np.asarray(value, dtype=float)
    u = {f"{name}.t.throttle": throttle} if throttle is not None else None
    sim.step(DT, u=u)
    return sim


def test_transverse_added_mass_makes_the_same_force_slower():
    """The headline effect: 10 N along y accelerates a hull with 10 kg
    of transverse added mass at F/(m+Ay), while the same 10 N along x
    still gets F/m — a slender hull accelerates sideways as if it
    weighed twice its dry mass."""
    added = AddedMass("am", translational=(0.0, 10.0, 0.0))

    w = _world()
    w.add_craft(_craft(added=added, thruster_axis=(0.0, 10.0, 0.0)))
    vy = _step(w, "hull", throttle=1.0).state["hull"]["velocity"][1]
    np.testing.assert_allclose(vy, 10.0 / (10.0 + 10.0) * DT, rtol=1e-6)

    w = _world()
    w.add_craft(_craft(added=AddedMass("am", translational=(0.0, 10.0, 0.0)),
                       thruster_axis=(10.0, 0.0, 0.0)))
    vx = _step(w, "hull", throttle=1.0).state["hull"]["velocity"][0]
    np.testing.assert_allclose(vx, 10.0 / 10.0 * DT, rtol=1e-6)


def test_the_effective_mass_follows_the_body_not_the_world():
    """Yaw the hull 90° and push along WORLD x: the force now runs along
    body −y (transverse), so the added mass applies — the anisotropy is
    body-fixed, which is the entire reason the linear solve had to leave
    the world frame."""
    # Thruster pushes along body y; hull yawed +90° makes that world −x.
    w = _world()
    w.add_craft(_craft(added=AddedMass("am", translational=(0.0, 10.0, 0.0)),
                       thruster_axis=(0.0, 10.0, 0.0)))
    q_yaw90 = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    sim = _step(w, "hull", throttle=1.0, orientation=q_yaw90)
    v = sim.state["hull"]["velocity"]
    np.testing.assert_allclose(v[0], -10.0 / 20.0 * DT, rtol=1e-5)
    assert abs(v[1]) < 1e-9


def test_isotropic_added_mass_is_just_extra_mass():
    """With A = a·I every velocity-product term cancels identically —
    a coasting, tumbling hull with isotropic added mass feels no
    spurious force. This is the sign-error and double-counting trap."""
    w = _world()
    w.add_craft(_craft(added=AddedMass("am",
                                       translational=(5.0, 5.0, 5.0))))
    sim = _step(w, "hull", velocity=(1.0, 0.5, -0.3),
                angular_velocity=(0.4, -0.2, 0.6))
    v = sim.state["hull"]["velocity"]
    np.testing.assert_allclose(v, [1.0, 0.5, -0.3], atol=1e-12)


def test_the_munk_moment_turns_a_slender_hull_broadside():
    """τ = −ν×(Aν): moving at 45° incidence with Ay > Ax, the couple
    pushes the nose AWAY from the flow — the destabilizing moment that
    is the reason bare torpedo hulls need fins. Magnitude by hand:
    τz = −(Ay−Ax)·vx·vy."""
    ax, ay = 1.0, 11.0
    vx, vy = 2.0, 2.0
    w = _world()
    w.add_craft(_craft(added=AddedMass("am",
                                       translational=(ax, ay, 0.0)),
                       moi=(2.0, 2.0, 2.0)))
    sim = _step(w, "hull", velocity=(vx, vy, 0.0))
    wz = sim.state["hull"]["angular_velocity"][2]
    tau_z = -(ay - ax) * vx * vy
    np.testing.assert_allclose(wz, tau_z / 2.0 * DT, rtol=1e-4)
    assert wz < 0.0        # nose swings starboard: toward broadside


def test_rotational_added_inertia_slows_the_spin_up():
    """A torque from an offset thruster spins the hull at
    τ/(I+B) — the entrained fluid resists angular acceleration too."""
    # 10 N along +y at +1 m x lever: τz = 10 N·m.
    B = 3.0
    w = _world()
    w.add_craft(_craft(added=AddedMass("am", rotational=(0.0, 0.0, B)),
                       thruster_axis=(0.0, 10.0, 0.0),
                       thruster_offset=(1.0, 0.0, 0.0),
                       moi=(2.0, 2.0, 2.0)))
    sim = _step(w, "hull", throttle=1.0)
    wz = sim.state["hull"]["angular_velocity"][2]
    np.testing.assert_allclose(wz, 10.0 / (2.0 + B) * DT, rtol=1e-4)


def test_mount_orientation_rotates_the_tensor():
    """An AddedMass authored slender-along-x but mounted yawed 90° is
    slender along body y: the x push now sees the big entry."""
    q_yaw90 = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    added = AddedMass("am", translational=(0.0, 10.0, 0.0),
                      mount_orientation=q_yaw90)
    w = _world()
    w.add_craft(_craft(added=added, thruster_axis=(10.0, 0.0, 0.0)))
    vx = _step(w, "hull", throttle=1.0).state["hull"]["velocity"][0]
    np.testing.assert_allclose(vx, 10.0 / 20.0 * DT, rtol=1e-5)


def test_zero_added_mass_is_bit_identical_to_none():
    """The augmentation must be invisible when the tensors are zero —
    the compile-time branch matters, not just the values."""
    def run(with_part: bool):
        w = _world()
        added = AddedMass("am") if with_part else None
        w.add_craft(_craft(added=added, thruster_axis=(3.0, 0.0, 0.0)))
        sim = _step(w, "hull", throttle=1.0,
                    velocity=(0.3, -0.2, 0.1),
                    angular_velocity=(0.5, 0.4, -0.3))
        return sim.state["hull"]

    a, b = run(True), run(False)
    for key in ("position", "velocity", "orientation", "angular_velocity"):
        np.testing.assert_allclose(np.asarray(a[key], dtype=float),
                                   np.asarray(b[key], dtype=float),
                                   atol=1e-14)


def test_negative_entries_are_refused():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        AddedMass("am", translational=(-1.0, 0.0, 0.0))


def test_jointed_craft_carries_added_inertia_through_the_block_solve():
    """A live articulated joint routes the dynamics through the
    joint-space block (A(q) from the kinetic-energy Hessian), which gets
    added rotational inertia via ½ωᵀBω — a different code path from the
    flat 3×3 solve. Same physics must come out: an off-axis reaction
    wheel does not couple into z at rest, so the offset thruster still
    spins the hull at τ/(Izz+Bz)."""
    from manta.parts import RevoluteJoint

    B = 3.0
    c = Craft("hull")
    c.add(Mass("core", mass=10.0, moi=(2.0, 2.0, 2.0)))
    c.add(AddedMass("am", rotational=(0.0, 0.0, B)))
    c.add(Thruster("t", force=(0.0, 10.0, 0.0),
                   mount_offset=(1.0, 0.0, 0.0)))     # τz = 10 N·m
    wheel = RevoluteJoint("wheel", mode="passive", axis=(1.0, 0.0, 0.0))
    wheel.add(Mass("rotor", mass=1e-3, moi=(1e-4, 1e-4, 1e-4)))
    c.add(wheel)

    w = _world()
    w.add_craft(c)
    sim = TargetNumpy(Sim(w))
    sim.step(DT, u={"hull.t.throttle": 1.0})
    wz = sim.state["hull"]["angular_velocity"][2]
    izz = 2.0 + 1e-4                     # hull + rotor's own moi
    np.testing.assert_allclose(wz, 10.0 / (izz + B) * DT, rtol=1e-3)


def test_linearization_differentiates_through_added_mass():
    """LQR builds F/B by differentiating the tick — through the new
    (m·I + A) solve and the Munk wrench. Anisotropic added mass must
    show up as anisotropic control effectiveness: the transverse
    channel, twice as heavy, gets different gains from the axial one;
    with no added mass the two channels are identical by symmetry."""
    from manta import LQR

    def gains(a_y: float) -> np.ndarray:
        c = Craft("c")
        c.add(Mass("body", mass=10.0))
        c.add(AddedMass("am", translational=(0.0, a_y, 0.0)))
        c.add(Thruster("tx", force=(1.0, 0.0, 0.0)))
        c.add(Thruster("ty", force=(0.0, 1.0, 0.0)))
        c.add(Thruster("tz", force=(0.0, 0.0, 1.0)))
        w = (World()
             .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
             .add_field(FluidField().add_uniform(density=1000.0)))
        w.add_craft(c)
        lqr = LQR(
            w,
            x_ref={"c": {"position": (0, 0, 0), "velocity": (0, 0, 0)}},
            u_ref={"tx.throttle": 0.0, "ty.throttle": 0.0,
                   "tz.throttle": 0.0},
            regulate=["c.position", "c.velocity"],
            Q=np.diag([10, 10, 10, 1, 1, 1]), R=np.eye(3) * 0.1, dt=0.02)
        return np.asarray(lqr.K)

    K = gains(a_y=10.0)
    assert np.all(np.isfinite(K))
    # x feedback -> tx row, y feedback -> ty row: pull each channel's
    # own-position gain magnitude.
    kx = np.max(np.abs(K[0]))
    ky = np.max(np.abs(K[1]))
    assert not np.isclose(kx, ky, rtol=1e-3)    # anisotropy visible

    K0 = gains(a_y=0.0)
    assert np.isclose(np.max(np.abs(K0[0])), np.max(np.abs(K0[1])),
                      rtol=1e-6)                # symmetric without it
