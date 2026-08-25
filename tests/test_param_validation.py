"""Construction-time validation — names and physical parameters.

Names (Craft/Part/Disturbance/World/Planet) must be valid identifiers:
they flow into tick-signature dot-names, Module entry-point names, and
C++ symbols, so anything else would only fail when the *generated* code
compiles. Physical parameters that are nonsense (negative mass, zero
joint axis, non-perpendicular airfoil axes, …) are refused where the
user typed them instead of surfacing as confusing physics.
"""

import pytest

from manta import Craft, World
from manta.fields import PointMassGravity, UniformFluid, UniformGravity
from manta.parts import Aerofoil, Mass, PointBuoy, PrismaticJoint, RevoluteJoint
from manta.planets import Planet

# ---------------------------------------------------------------------------
# Names must be identifiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["imu-9dof", "gps antenna", "9lives", ""])
def test_part_name_rejected(bad):
    with pytest.raises(ValueError, match="not a valid identifier"):
        Mass(bad)


def test_craft_world_planet_names_rejected():
    with pytest.raises(ValueError, match="not a valid identifier"):
        Craft("my-craft")
    with pytest.raises(ValueError, match="not a valid identifier"):
        World("my world")
    with pytest.raises(ValueError, match="not a valid identifier"):
        Planet("planet 9")


def test_disturbance_name_rejected_and_autoname_ok():
    from manta.fields import GravityField
    with pytest.raises(ValueError, match="not a valid identifier"):
        UniformGravity((0.0, 0.0, -9.81), name="g-field")
    # Auto-names are assigned at Field.add — deterministic per field
    # (`<ClassName>_<index>`), never a function of how many disturbances
    # the process happened to construct earlier — and identifier-safe.
    g = GravityField()
    d0, d1 = UniformGravity((0, 0, -9.81)), UniformGravity((0, 0, -1.62))
    assert d0.name is None                     # deferred until registration
    g.add(d0).add(d1)
    assert d0.name == "UniformGravity_0" and d1.name == "UniformGravity_1"
    assert d0.name.isidentifier()


# ---------------------------------------------------------------------------
# Structure parts
# ---------------------------------------------------------------------------

def test_mass_rejects_negative():
    with pytest.raises(ValueError, match="mass must be >= 0"):
        Mass("m", mass=-1.0)
    with pytest.raises(ValueError, match="moi"):
        Mass("m", moi=(1.0, -1.0, 1.0))
    Mass("m", mass=0.0)               # zero allowed; craft-level guard


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_declarations_reject_non_finite_numbers(bad):
    with pytest.raises(ValueError, match="must be finite"):
        Mass("m", mass=bad)
    with pytest.raises(ValueError, match="must be finite"):
        Mass("m", moi=(1.0, bad, 1.0))
    from manta.parts import IMU
    with pytest.raises(ValueError, match="must be finite"):
        IMU("imu", gyro_noise_sigma=bad)


def test_point_buoy_rejects_negative_volume():
    with pytest.raises(ValueError, match="volume must be >= 0"):
        PointBuoy("b", volume=-1e-3)


# ---------------------------------------------------------------------------
# Joint axis: nonzero, normalized at construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RevoluteJoint, PrismaticJoint])
def test_joint_axis_validated_and_normalized(cls):
    with pytest.raises(ValueError, match="axis must be nonzero"):
        cls("j", axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="length-3"):
        cls("j", axis=(1.0, 0.0))
    j = cls("j", axis=(0.0, 0.0, 2.0))
    assert j.axis == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Airfoil: area > 0, orthonormal axes
# ---------------------------------------------------------------------------

def test_aerofoil_validation():
    with pytest.raises(ValueError, match="area must be > 0"):
        Aerofoil("wing", area=0.0)
    with pytest.raises(ValueError, match="must be perpendicular"):
        Aerofoil("wing", chord_axis=(1.0, 0.0, 0.0),
                 normal_axis=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="CD_0 and induced_k"):
        Aerofoil("wing", CD_0=-0.01)
    # Non-unit but orthogonal axes are normalized.
    a = Aerofoil("wing", chord_axis=(2.0, 0.0, 0.0),
                 normal_axis=(0.0, 0.0, 0.5))
    assert a.chord_axis == (1.0, 0.0, 0.0)
    assert a.normal_axis == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Field disturbances
# ---------------------------------------------------------------------------

def test_disturbance_parameter_validation():
    with pytest.raises(ValueError, match="GM must be > 0"):
        PointMassGravity(position=(0.0, 0.0, 0.0), GM=-3.986e14)
    with pytest.raises(ValueError, match="eps must be >= 0"):
        PointMassGravity(position=(0.0, 0.0, 0.0), GM=3.986e14, eps=-1.0)
    PointMassGravity(position=(0.0, 0.0, 0.0), GM=3.986e14, eps=0.0)
    with pytest.raises(ValueError, match="density must be >= 0"):
        UniformFluid(density=-1.225)
