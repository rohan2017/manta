"""Sparse tangent-space RTI MPC contracts."""

import math
from dataclasses import replace

import numpy as np
import pytest

from manta import (
    MPC,
    Craft,
    CraftHorizonReference,
    MpcNumericalError,
    MPCReference,
    Sim,
    TargetNumpy,
    World,
)
from manta.fields import FluidField, GravityField
from manta.parts import DragSurface, Mass, Thruster

RHO = 1025.0


def _world(names=("tug",)):
    world = (World(name="mpc_test")
             .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
             .add_field(FluidField().add_uniform(density=RHO)))
    bounds = {}
    for index, name in enumerate(names):
        craft = Craft(name)
        craft.add(Mass("hull", mass=8.0, moi=(0.2, 0.5, 0.5)))
        craft.add(DragSurface.directional_quadratic(
            "drag", areas=(0.05, 0.12, 0.12), drag_coefficient=1.0))
        craft.add(Thruster("prop", force=(20.0, 0.0, 0.0)))
        world.add_craft(craft, position=(0.0, 2.0*index, 0.0))
        bounds[f"{name}.prop.throttle"] = (-1.0, 1.0)
    return world, bounds


def _reference(horizon, target=(5.0, 0.0, 0.0), bank_deg=20.0):
    return CraftHorizonReference(
        positions=np.tile(target, (horizon, 1)),
        tangents=np.tile([1.0, 0.0, 0.0], (horizon, 1)),
        forward_speeds=np.ones(horizon),
        up=np.tile([0.0, 0.0, 1.0], (horizon, 1)),
        bank_limits=np.full(horizon, math.radians(bank_deg)),
    )


def test_reference_normalizes_and_owns_vectors():
    tangents = np.tile([2.0, 0.0, 0.0], (3, 1))
    ref = CraftHorizonReference(
        positions=np.zeros((3, 3)), tangents=tangents,
        forward_speeds=np.ones(3), up=np.tile([0, 0, 4.0], (3, 1)),
        bank_limits=np.full(3, 0.2))
    tangents[0] = 99.0
    np.testing.assert_allclose(ref.tangents[:, 0], 1.0)
    np.testing.assert_allclose(ref.up[:, 2], 1.0)
    assert not ref.positions.flags.writeable


@pytest.mark.parametrize("field,value,message", [
    ("forward_speeds", [-1, 1], "non-negative"),
    ("bank_limits", [0, .2], "bank limits"),
    ("tangents", [[0, 0, 0], [1, 0, 0]], "tangent"),
])
def test_reference_rejects_invalid_policy(field, value, message):
    kwargs = {
        "positions": np.zeros((2, 3)), "tangents": np.tile([1, 0, 0], (2, 1)),
        "forward_speeds": np.ones(2), "up": np.tile([0, 0, 1], (2, 1)),
        "bank_limits": np.full(2, .2)}
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        CraftHorizonReference(**kwargs)


def test_reference_accepts_maskable_pose_and_twist_objectives():
    ref = CraftHorizonReference(
        positions=np.zeros((2, 3)), tangents=np.tile([1, 0, 0], (2, 1)),
        forward_speeds=np.ones(2), up=np.tile([0, 0, 1], (2, 1)),
        bank_limits=np.full(2, .2),
        orientations=np.tile([2.0, 0, 0, 0], (2, 1)),
        attitude_weights=np.tile([1.0, 0.0, 2.0], (2, 1)),
        body_velocities=np.zeros((2, 3)),
        body_velocity_weights=np.ones((2, 3)),
        angular_rates=np.zeros((2, 3)),
        angular_rate_weights=np.ones((2, 3)))
    np.testing.assert_allclose(ref.orientations[:, 0], 1.0)
    with pytest.raises(ValueError, match="supplied together"):
        CraftHorizonReference(
            positions=np.zeros((2, 3)),
            tangents=np.tile([1, 0, 0], (2, 1)),
            forward_speeds=np.ones(2), up=np.tile([0, 0, 1], (2, 1)),
            bank_limits=np.full(2, .2), body_velocities=np.zeros((2, 3)))


def test_sparse_structure_scales_linearly_and_uses_tangent_state():
    world, bounds = _world()
    short = MPC(world, u_bounds=bounds, horizon=5, dt=.1)
    long = MPC(world, u_bounds=bounds, horizon=10, dt=.1)
    assert short.ndx == short.nx-1  # SO(3): three tangent vs four ambient
    assert short.qp_shape == (5*(short.ndx+short.nu+1),
                              5*(short.ndx+2))
    h_short, a_short = short.qp_nonzeros
    h_long, a_long = long.qp_nonzeros
    assert h_long < 2.1*h_short
    assert a_long < 2.2*a_short


def test_selected_propulsion_preserves_auxiliary_force_in_predicted_dynamics():
    from manta.parts.disturbance.external_wrench import ExternalWrench

    world, bounds = _world()
    world.crafts[0].add(ExternalWrench("auxiliary", fy=4.0))
    mpc = MPC(world, u_bounds=bounds, inputs=("prop.throttle",), horizon=4, dt=.1)
    result = mpc.tick(None, _reference(4))
    assert mpc.input_names == ("tug.prop.throttle",)
    assert set(result.controls) == {"tug.prop.throttle"}
    # An ordinary simulation integrates the same nonzero default load. This
    # checks the retained physical effect, not only the reduced input count.
    sim = TargetNumpy(Sim(world))
    for _ in range(2):
        sim.step(.05, u=result.controls)
    predicted = mpc.spec.unpack(result.nominal_states[1])
    np.testing.assert_allclose(predicted["tug.position"], sim.state["tug"]["position"], atol=1e-8)
    assert predicted["tug.position"][1] > 0


@pytest.mark.parametrize("inputs,controlled,message", [
    (("a.prop.throttle", "a.prop.throttle"), ("a",), "duplicates"),
    (("b.prop.throttle",), ("a",), "no inputs"),
    (("a.prop.throttle", "b.prop.throttle"), ("a",), "uncontrolled crafts"),
])
def test_mpc_rejects_inconsistent_input_selection(inputs, controlled, message):
    world, bounds = _world(("a", "b"))
    with pytest.raises(ValueError, match=message):
        MPC(world, u_bounds=bounds, inputs=inputs, controlled=controlled, horizon=3)


@pytest.mark.parametrize("compiled", [False, True])
def test_mx_articulated_dynamics_keep_joint_state_with_selected_propulsion(compiled):
    from manta.parts import PrismaticJoint

    world, bounds = _world()
    joint = PrismaticJoint("gantry", axis=(0, 1, 0), mode="saturating",
                           stall_force=20, damping=2, force_cmd=5)
    joint.add(Mass("payload", mass=1, moi=(.1, .1, .1)))
    world.crafts[0].add(joint)
    mpc = MPC(world, u_bounds=bounds, inputs=("prop.throttle",),
              expand_dynamics=False, horizon=3, dt=.1, compile=compiled,
              compile_optimization="startup")
    result = mpc.tick(None, _reference(3))
    assert set(result.controls) == {"tug.prop.throttle"}
    assert "tug.gantry.displacement" in {s.name for s in mpc.spec.slots}
    assert np.isfinite(result.nominal_states).all()
    sim = TargetNumpy(Sim(world))
    for _ in range(2):
        sim.step(.05, u=result.controls)
    predicted = mpc.spec.unpack(result.nominal_states[1])
    for name in ("position", "velocity", "gantry.displacement", "gantry.rate"):
        np.testing.assert_allclose(predicted[f"tug.{name}"],
                                   sim.state["tug"][name], atol=1e-8)
    assert predicted["tug.gantry.displacement"] > 0


def test_one_rti_tick_obeys_bounds_and_shifts_a_finite_plan():
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=8, dt=.1)
    result = mpc.tick(None, _reference(8))
    assert result.qp_status.lower().startswith("solved")
    assert result.control_vector.shape == (1,)
    assert -1.0 <= result.control_vector[0] <= 1.0
    assert result.nominal_controls.shape == (8, 1)
    assert result.nominal_states.shape == (9, mpc.nx)
    assert np.all(np.isfinite(result.nominal_states))
    assert result.timings.total_ms > 0.0
    # Solver-side adjustments are declared, not hidden: the Tikhonov term
    # the QP was solved under and how many bound projections moved values.
    from manta.control.rti import HESSIAN_REGULARIZATION
    assert result.hessian_regularization == HESSIAN_REGULARIZATION > 0.0
    assert isinstance(result.clipped_controls, int)
    assert result.clipped_controls == 0
    mpc.reset()
    assert mpc.last_result is None
    np.testing.assert_array_equal(mpc._U, 0.0)
    np.testing.assert_array_equal(mpc._qp_x, 0.0)
    np.testing.assert_array_equal(mpc._qp_lam_x, 0.0)
    np.testing.assert_array_equal(mpc._qp_lam_a, 0.0)


def test_reset_refuses_out_of_bound_warm_controls():
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=4, dt=.1)
    with pytest.raises(ValueError, match="stage 2.*prop.throttle.*outside"):
        mpc.reset(np.array([[0.0], [0.5], [1.01], [-0.5]]))
    np.testing.assert_array_equal(mpc._U, 0.0)


def test_failed_qp_raises_typed_retryable_numerical_error():
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=4, dt=.1)

    class FailedSolver:
        def __call__(self, **_kwargs):
            return {}

        def stats(self):
            return {"success": False, "return_status": "maximum iterations"}

    mpc._qp = FailedSolver()
    with pytest.raises(MpcNumericalError, match="maximum iterations"):
        mpc.tick(None, _reference(4))


def test_warm_start_advances_by_controller_time_not_a_whole_node():
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=8, dt=.1)
    result = mpc.tick(None, _reference(8), advance_s=.025)
    expected = (.75*result.nominal_controls[:-1]
                + .25*result.nominal_controls[1:])
    np.testing.assert_allclose(mpc._U[:-1], expected)
    np.testing.assert_allclose(mpc._U[-1], result.nominal_controls[-1])
    with pytest.raises(ValueError, match="advance_s"):
        mpc.tick(None, _reference(8), advance_s=-.1)


def test_reference_must_exactly_cover_controlled_crafts():
    world, bounds = _world(("a", "b"))
    mpc = MPC(world, u_bounds=bounds, horizon=3, controlled=("a", "b"))
    with pytest.raises(ValueError, match="exactly cover"):
        mpc.tick(None, MPCReference({"a": _reference(3)}))


def test_multi_craft_world_expands_state_controls_and_references():
    world, bounds = _world(("a", "b"))
    mpc = MPC(world, u_bounds=bounds, horizon=4, controlled=("a", "b"))
    result = mpc.tick(None, MPCReference({
        "a": _reference(4, (4, 0, 0)),
        "b": _reference(4, (4, 2, 0)),
    }))
    assert set(result.controls) == {"a.prop.throttle", "b.prop.throttle"}
    assert result.nominal_states.shape == (5, mpc.nx)
    assert result.control_vector.shape == (2,)
    assert all(value > 0.0 for value in result.controls.values())


def test_scheduled_attitude_envelope_is_a_hard_maneuver_constraint():
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=4, dt=.1)
    constrained = replace(
        _reference(4),
        constraint_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        attitude_error_lower=np.tile([-.05, -np.inf, -np.inf], (4, 1)),
        attitude_error_upper=np.tile([+.05, +np.inf, +np.inf], (4, 1)),
    )
    result = mpc.tick(None, constrained)
    assert mpc.qp_shape[1] == 4*(mpc.ndx+5)
    assert result.predicted_attitude_constraint_violation < 1e-8
    with pytest.raises(ValueError, match="supplied together"):
        replace(_reference(4), constraint_orientations=np.tile(
            [1.0, 0.0, 0.0, 0.0], (4, 1)))


def test_native_qp_selects_constraint_structure_only_when_needed(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    world, bounds = _world()
    mpc = MPC(world, u_bounds=bounds, horizon=4, dt=.1, compile=True)
    base = _reference(4)
    mpc.tick(None, base)
    assert mpc.qp_shape[1] == 4*(mpc.ndx+2)
    constrained = replace(
        base,
        constraint_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        attitude_error_lower=np.tile([-.05, -np.inf, -np.inf], (4, 1)),
        attitude_error_upper=np.tile([+.05, +np.inf, +np.inf], (4, 1)),
    )
    assert mpc.tick(None, constrained).qp_status == "solved"
    assert mpc.qp_shape[1] == 4*(mpc.ndx+5)
    assert mpc.tick(None, base).qp_status == "solved"
    assert mpc.qp_shape[1] == 4*(mpc.ndx+2)


def test_uncontrolled_craft_remains_in_world_dynamics_at_default_input():
    world, bounds = _world(("a", "b"))
    mpc = MPC(world, u_bounds=bounds, horizon=3, controlled=("a",))
    result = mpc.tick(None, _reference(3))
    assert set(result.controls) == {"a.prop.throttle"}
    assert mpc.spec.slot("b.position").ambient_offset < mpc.nx


def test_closed_loop_moves_toward_the_reference():
    world, bounds = _world()
    plant = TargetNumpy(Sim(world))
    mpc = MPC(world, u_bounds=bounds, horizon=12, dt=.1)
    ref = _reference(12, (5, 0, 0))
    start = plant.state["tug"]["position"].copy()
    for _ in range(12):
        result = mpc.tick(plant.model_state(), ref)
        plant.step(.1, u={"prop.throttle": result.control_vector[0]})
    assert plant.state["tug"]["position"][0] > start[0]+0.1


def test_native_kernel_compilation_matches_uncompiled(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    world, bounds = _world()
    interpreted = MPC(world, u_bounds=bounds, horizon=3, dt=.1)
    compiled = MPC(world, u_bounds=bounds, horizon=3, dt=.1, compile=True)
    a = interpreted.tick(None, _reference(3)).control_vector
    compiled_result = compiled.tick(None, _reference(3))
    b = compiled_result.control_vector
    # The native path calls OSQP directly while the interpreted path uses
    # CasADi's conic adapter. Both use 2e-3 QP tolerances, so their primal
    # answers need only agree within that solver accuracy.
    np.testing.assert_allclose(a, b, rtol=2e-3, atol=1e-3)
    assert compiled_result.qp_iterations > 0


@pytest.mark.parametrize("condense_to", [0, 4])
def test_structured_hpipm_matches_sparse_osqp(
    monkeypatch, tmp_path, condense_to,
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    world, bounds = _world()
    reference = _reference(8)
    osqp = MPC(
        world, u_bounds=bounds, horizon=8, dt=.1,
        compile=True, qp_backend="osqp")
    hpipm = MPC(
        world, u_bounds=bounds, horizon=8, dt=.1,
        compile=True, qp_backend="hpipm",
        qp_options={"condense_to": condense_to})
    expected = osqp.tick(None, reference)
    actual = hpipm.tick(None, reference)
    # OSQP permits a 2e-3 residual and can therefore step slightly beyond an
    # active trust bound. HPIPM lands on that bound to much tighter accuracy.
    np.testing.assert_allclose(
        actual.control_vector, expected.control_vector,
        rtol=5e-3, atol=2e-3)
    assert actual.qp_status == "solved"
    assert actual.qp_primal_residual < 2e-3
    assert actual.qp_dual_residual < 2e-3
    assert actual.predicted_bank_violation < 1e-8
    assert actual.qp_used_uncondensed_fallback is (condense_to != 0)
    assert actual.qp_condensed_candidate_valid is (
        None if condense_to == 0 else True)


def test_structured_hpipm_rebuilds_for_hard_attitude_constraints(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    world, bounds = _world()
    mpc = MPC(
        world, u_bounds=bounds, horizon=4, dt=.1,
        compile=True, qp_backend="hpipm",
        qp_options={"condense_to": 2})
    base = _reference(4)
    assert mpc.tick(None, base).qp_status == "solved"
    constrained = replace(
        base,
        constraint_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        attitude_error_lower=np.tile([-.05, -np.inf, -np.inf], (4, 1)),
        attitude_error_upper=np.tile([+.05, +np.inf, +np.inf], (4, 1)),
    )
    result = mpc.tick(None, constrained)
    assert result.qp_status == "solved"
    assert result.predicted_attitude_constraint_violation < 1e-8
    assert mpc.tick(None, base).qp_status == "solved"
