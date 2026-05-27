"""Frame-indexed TickContext accessors — ctx.X[Frame] = X relative to that
frame, in that frame's coords.

Pins the defining identities: X[PartFrame] = 0, X[CraftFrame] = 0 for a
rigidly-mounted part (nonzero only through a joint), X[WorldFrame] is the
absolute value, and X[ParentFrame] == X[CraftFrame] for a part on the root.
"""

import numpy as np

from manta import World, Craft, TargetNumpy
from manta.fields import GravityField
from manta.parts import Joint, Mass
from manta.parts.base import Output, Part, PartUpdate
from manta.ir.frames import WorldFrame, CraftFrame, ParentFrame, PartFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench


class _FrameProbe(Part):
    """Exposes a few kinematic quantities in several reference frames so a
    test can read them out of the state dict."""
    vel_world   = Output(shape="vec3")
    vel_craft   = Output(shape="vec3")
    vel_parent  = Output(shape="vec3")
    vel_part    = Output(shape="vec3")
    acc_part    = Output(shape="vec3")
    omega_craft = Output(shape="vec3")
    omega_part  = Output(shape="vec3")

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={
                "vel_world":   ctx.velocity[WorldFrame],
                "vel_craft":   ctx.velocity[CraftFrame],
                "vel_parent":  ctx.velocity[ParentFrame],
                "vel_part":    ctx.velocity[PartFrame],
                "acc_part":    ctx.acceleration[PartFrame],
                "omega_craft": ctx.angular_velocity[CraftFrame],
                "omega_part":  ctx.angular_velocity[PartFrame],
            },
        )


def _free(craft, **ov):
    w = World().add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
    w.add_craft(craft, **ov)
    return TargetNumpy(w.compile())


def _o(state, craft, slot):
    return np.asarray(state[craft][slot]).ravel()


def test_partframe_views_are_zero_and_craft_zero_when_rigid():
    """A probe rigidly mounted on the root: it is at rest in both its own
    frame and the craft frame, even while the craft spins + translates."""
    c = Craft("rig")
    c.add(Mass("body", mass=1.0, moi=(1.0, 1.0, 1.0)))
    c.add(_FrameProbe("p", transform=(0.3, 0.0, 0.0)))
    sim = _free(c, velocity=(5.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0))
    st = sim.step(sim.initial_state(), dt=1e-4)

    np.testing.assert_allclose(_o(st, "rig", "p.vel_part"), 0.0, atol=1e-12)
    np.testing.assert_allclose(_o(st, "rig", "p.acc_part"), 0.0, atol=1e-12)
    np.testing.assert_allclose(_o(st, "rig", "p.omega_part"), 0.0, atol=1e-12)
    # Rigid in the craft → zero relative motion w.r.t. the craft.
    np.testing.assert_allclose(_o(st, "rig", "p.vel_craft"), 0.0, atol=1e-12)
    np.testing.assert_allclose(_o(st, "rig", "p.omega_craft"), 0.0, atol=1e-12)


def test_world_craft_part_frames_are_distinct_on_a_rotor():
    """Probe on a spinning rotor (offset d from the z axis), craft
    translating at +x. At angle 0:
      vel[WorldFrame] = craft V + rotor spin = (V, ω₀·d, 0)
      vel[CraftFrame] = rotor spin only      = (0, ω₀·d, 0)
      vel[PartFrame]  = 0
      angular_velocity[CraftFrame] = ω₀·ẑ    (rotor spin rel. to craft)
    """
    V, omega0, d = 5.0, 3.0, 0.4
    c = Craft("rotorcraft")
    c.add(Mass("hub", mass=100.0, moi=(10.0, 10.0, 10.0)))
    wheel = Joint("wheel", mode="passive", axis=(0.0, 0.0, 1.0))
    wheel.add(Mass("rim", mass=0.01, moi=(1e-4, 1e-4, 1e-4)))
    wheel.add(_FrameProbe("p", transform=(d, 0.0, 0.0)))
    c.add(wheel)
    sim = _free(c, velocity=(V, 0.0, 0.0), **{"wheel.rate": omega0})
    st = sim.step(sim.initial_state(), dt=1e-5)

    np.testing.assert_allclose(_o(st, "rotorcraft", "p.vel_part"),
                               0.0, atol=1e-10)
    np.testing.assert_allclose(_o(st, "rotorcraft", "p.vel_craft"),
                               (0.0, omega0 * d, 0.0), atol=1e-3)
    np.testing.assert_allclose(_o(st, "rotorcraft", "p.vel_world"),
                               (V, omega0 * d, 0.0), atol=1e-3)
    np.testing.assert_allclose(_o(st, "rotorcraft", "p.omega_craft"),
                               (0.0, 0.0, omega0), atol=1e-9)
    # Joint sits directly on the root, so "relative to parent" == "relative
    # to craft".
    np.testing.assert_allclose(_o(st, "rotorcraft", "p.vel_parent"),
                               _o(st, "rotorcraft", "p.vel_craft"), atol=1e-12)
