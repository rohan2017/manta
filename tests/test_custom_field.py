"""User-authored Field subclasses — the extension contract.

A field kind defined entirely OUTSIDE manta source (here: a RadarField)
must be a first-class citizen: registrable, queryable from parts via
`ctx.has_field` / `ctx.field`, its state-bearing disturbances plumbed
into the tick and the EKF — with no built-in list of field kinds
anywhere, and no silent empty-field default when it's absent.
"""

import casadi as ca
import numpy as np
import pytest

from manta import Craft, Sim, TargetNumpy, World
from manta.fields import Disturbance, SuperposedField
from manta.ir.frames import PartFrame
from manta.ir.types import Scalar, Vec3
from manta.ir.wrench import Wrench
from manta.parts import Mass, Part, PartUpdate, Output, RandomWalkNoise


# --- the user's field, authored without touching manta -------------------

class RadarField(SuperposedField):
    """Received radar power density (W/m², Scalar) at a point."""
    value_shape = Scalar

    def _zero_value(self):
        return Scalar.constant(0.0)


class RadarEmitter(Disturbance):
    """Isotropic emitter: power / (4π·r²), eps-softened."""
    field_value_shape = Scalar

    def __init__(self, position, power, *, name=None, **overrides):
        super().__init__(name=name, **overrides)
        self.position = tuple(float(x) for x in position)
        self.power = float(power)

    def contribute_at_sym(self, point, t):
        r = point._mx - ca.MX(list(self.position))
        r_sq = ca.dot(r, r) + 1e-9
        return Scalar.from_mx(self.power / (4.0 * np.pi * r_sq))


class DriftingEmitter(RadarEmitter):
    """An emitter whose power drifts as a random walk — state-bearing,
    so the plumbing (and the EKF) must pick it up on a custom field."""
    power_drift = RandomWalkNoise("R1", sigma=0.1)

    def contribute_at_sym(self, point, t):
        base = super().contribute_at_sym(point, t)
        return Scalar.from_mx(base._mx + self.power_drift._mx)


# --- the user's parts ----------------------------------------------------

class RadarDetector(Part):
    """Reads the RadarField at its mount point. Requires the field."""
    requires_fields = [RadarField]
    power = Output()

    def update(self, ctx) -> PartUpdate:
        from manta.ir.frames import WorldFrame
        val = ctx.field(RadarField).value_at_sym(
            ctx.position[WorldFrame], ctx.t)
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          outputs={"power": val})


class OptionalRadarDetector(Part):
    """Branches on presence — reads −1.0 when no RadarField exists."""
    power = Output()

    def update(self, ctx) -> PartUpdate:
        from manta.ir.frames import WorldFrame
        if ctx.has_field(RadarField):
            val = ctx.field(RadarField).value_at_sym(
                ctx.position[WorldFrame], ctx.t)
        else:
            val = Scalar.constant(-1.0)
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          outputs={"power": val})


def _craft(det_cls):
    c = Craft("probe")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(det_cls("radar"))
    return c


# --- tests ---------------------------------------------------------------

def test_custom_field_queried_by_part():
    w = World()
    w.add_field(RadarField().add(RadarEmitter((10.0, 0.0, 0.0), 100.0)))
    w.add_craft(_craft(RadarDetector), position=(0.0, 0.0, 0.0))
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    expected = 100.0 / (4.0 * np.pi * 100.0)
    assert np.isclose(np.asarray(sim.outputs()["probe"]["radar.power"]).ravel()[0],
                      expected, rtol=1e-6)


def test_missing_custom_field_fails_requires_validation():
    """requires_fields works for user-authored field classes too."""
    w = World()
    w.add_craft(_craft(RadarDetector))
    with pytest.raises(ValueError, match="RadarField"):
        Sim(w)


def test_ctx_field_raises_without_registration():
    """A bare ctx.field on an unregistered kind raises at compile — no
    silent empty default."""
    class UndeclaredDetector(RadarDetector):
        requires_fields = []        # skip validation → hit ctx.field raw

    w = World()
    w.add_craft(_craft(UndeclaredDetector))
    with pytest.raises(ValueError, match="no RadarField is registered"):
        Sim(w)


def test_has_field_branch_without_registration():
    """The optional part compiles and reads its fallback with no field."""
    w = World()
    w.add_craft(_craft(OptionalRadarDetector))
    sim = TargetNumpy(Sim(w))
    sim.step(0.001)
    assert np.asarray(sim.outputs()["probe"]["radar.power"]).ravel()[0] == -1.0


def test_state_bearing_disturbance_on_custom_field_is_plumbed():
    """A RW-noise disturbance on a custom field gets its state slot in
    the sim (and evolves by the documented sqrt(dt)·driver law) — the
    plumbing walks the world's registered fields, not a built-in list."""
    w = World()
    w.add_field(RadarField().add(
        DriftingEmitter((10.0, 0.0, 0.0), 100.0, name="emitter")))
    w.add_craft(_craft(RadarDetector))
    sim = TargetNumpy(Sim(w))
    assert "emitter" in sim.initial_state()
    assert "power_drift" in sim.initial_state()["emitter"]
    drv, dt = 2.0, 0.01
    sim.state["emitter"]["power_drift_driver"] = drv
    sim.step(dt)
    assert np.isclose(float(sim.state["emitter"]["power_drift"]),
                      np.sqrt(dt) * drv, atol=1e-12)
