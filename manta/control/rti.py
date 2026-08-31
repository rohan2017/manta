"""Sparse real-time-iteration model predictive control.

``MPC`` is deliberately a solver-backed Manta runtime rather than a typed
``Module`` lowered by every backend.  Manta owns the model mathematics and the
fixed sparse transcription; the runtime keeps the nonlinear warm trajectory
and performs one constrained SQP/QP update per call.

The transcription is direct multiple shooting in the world's *tangent* state
space.  Predicted states remain on their declared manifolds, actuator limits
are hard, and bank envelopes are elastic and diagnosed.  A world may contain
several controlled crafts: their residuals and constraints are concatenated
while the complete coupled world dynamics are advanced together.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import casadi as ca
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _owned(value: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CraftHorizonReference:
    """One controlled craft's sampled intent over ``N`` future stages.

    Positions, tangents, and up vectors are expressed in the Manta world
    frame.  Forward speed is expressed in the craft's body-forward direction.
    """

    positions: FloatArray
    tangents: FloatArray
    forward_speeds: FloatArray
    up: FloatArray
    bank_limits: FloatArray
    orientations: FloatArray | None = None
    attitude_weights: FloatArray | None = None
    body_velocities: FloatArray | None = None
    body_velocity_weights: FloatArray | None = None
    world_velocities: FloatArray | None = None
    world_velocity_weights: FloatArray | None = None
    angular_rates: FloatArray | None = None
    angular_rate_weights: FloatArray | None = None
    constraint_orientations: FloatArray | None = None
    attitude_error_lower: FloatArray | None = None
    attitude_error_upper: FloatArray | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("reference positions must have shape (N, 3)")
        n = positions.shape[0]
        positions = _owned(positions, (n, 3), "reference positions")
        tangents = _owned(self.tangents, (n, 3), "reference tangents")
        up = _owned(self.up, (n, 3), "reference up vectors")
        speeds = _owned(
            np.asarray(self.forward_speeds).reshape(-1), (n,),
            "reference forward speeds")
        banks = _owned(
            np.asarray(self.bank_limits).reshape(-1), (n,),
            "reference bank limits")
        tangent_norms = np.linalg.norm(tangents, axis=1)
        up_norms = np.linalg.norm(up, axis=1)
        if np.any(tangent_norms < 1e-9):
            raise ValueError("every reference tangent must be non-zero")
        if np.any(up_norms < 1e-9):
            raise ValueError("every reference up vector must be non-zero")
        if np.any(speeds < 0.0):
            raise ValueError("reference forward speeds must be non-negative")
        if np.any(banks <= 0.0) or np.any(banks >= math.pi / 2.0):
            raise ValueError("reference bank limits must be in (0, pi/2)")
        normalized_tangents = tangents / tangent_norms[:, None]
        normalized_up = up / up_norms[:, None]
        normalized_tangents.setflags(write=False)
        normalized_up.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "tangents", normalized_tangents)
        object.__setattr__(self, "forward_speeds", speeds)
        object.__setattr__(self, "up", normalized_up)
        object.__setattr__(self, "bank_limits", banks)
        optional_pairs = (
            ("orientations", "attitude_weights", 4),
            ("body_velocities", "body_velocity_weights", 3),
            ("world_velocities", "world_velocity_weights", 3),
            ("angular_rates", "angular_rate_weights", 3),
        )
        for value_name, weight_name, width in optional_pairs:
            value = getattr(self, value_name)
            weight = getattr(self, weight_name)
            if (value is None) != (weight is None):
                raise ValueError(
                    f"{value_name} and {weight_name} must be supplied together")
            if value is None:
                continue
            owned_value = _owned(value, (n, width), value_name)
            owned_weight = _owned(weight, (n, 3), weight_name)
            if np.any(owned_weight < 0.0):
                raise ValueError(f"{weight_name} must be non-negative")
            if value_name == "orientations":
                norms = np.linalg.norm(owned_value, axis=1)
                if np.any(norms < 1e-9):
                    raise ValueError("reference orientations must be non-zero")
                owned_value = owned_value / norms[:, None]
                owned_value.setflags(write=False)
            object.__setattr__(self, value_name, owned_value)
            object.__setattr__(self, weight_name, owned_weight)
        constraint_values = (
            self.constraint_orientations,
            self.attitude_error_lower,
            self.attitude_error_upper,
        )
        if any(value is not None for value in constraint_values):
            if not all(value is not None for value in constraint_values):
                raise ValueError(
                    "constraint_orientations, attitude_error_lower, and "
                    "attitude_error_upper must be supplied together")
            assert self.constraint_orientations is not None
            assert self.attitude_error_lower is not None
            assert self.attitude_error_upper is not None
            orientations = _owned(
                self.constraint_orientations, (n, 4),
                "constraint_orientations")
            norms = np.linalg.norm(orientations, axis=1)
            if np.any(norms < 1e-9):
                raise ValueError("constraint orientations must be non-zero")
            orientations = orientations / norms[:, None]
            orientations.setflags(write=False)
            lower = np.asarray(self.attitude_error_lower, dtype=np.float64)
            upper = np.asarray(self.attitude_error_upper, dtype=np.float64)
            if lower.shape != (n, 3) or upper.shape != (n, 3):
                raise ValueError(
                    "attitude constraint bounds must have shape (N, 3)")
            if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
                raise ValueError("attitude constraint bounds must not contain NaN")
            if np.any(lower > upper):
                raise ValueError(
                    "attitude constraint lower bounds must not exceed upper")
            object.__setattr__(self, "constraint_orientations", orientations)
            lower, upper = lower.copy(), upper.copy()
            lower.setflags(write=False)
            upper.setflags(write=False)
            object.__setattr__(self, "attitude_error_lower", lower)
            object.__setattr__(self, "attitude_error_upper", upper)

    @property
    def horizon(self) -> int:
        return self.positions.shape[0]


@dataclass(frozen=True)
class MPCReference:
    """Horizon references keyed by controlled craft name."""

    crafts: Mapping[str, CraftHorizonReference]

    def __post_init__(self) -> None:
        copied = dict(self.crafts)
        if not copied:
            raise ValueError("MPCReference requires at least one craft")
        if not all(isinstance(v, CraftHorizonReference) for v in copied.values()):
            raise TypeError("every MPC reference value must be CraftHorizonReference")
        object.__setattr__(self, "crafts", MappingProxyType(copied))


# Tikhonov term added to the QP Hessian diagonal every tick. The reduced
# Hessian of a direct-multiple-shooting RTI step can be exactly singular in
# directions the cost does not see (an unweighted stage, a slack at its
# bound), and both OSQP and HPIPM need a strictly convex objective to
# factor. The value is far below any weight the cost contributes and is
# reported on every `MPCResult`, so it is a declared solver setting rather
# than an invisible nudge.
HESSIAN_REGULARIZATION = 1e-9
# Native QP bridges take finite bounds only; an unbounded row is passed at
# this magnitude (the solvers treat it as "inactive"). It is a representation
# detail of the bound, not a change to the problem.
QP_BOUND_INFINITY = 1e30


@dataclass(frozen=True)
class MPCTimings:
    rollout_linearize_ms: float
    assemble_ms: float
    solve_ms: float
    total_ms: float
    qp_update_ms: float = 0.0
    qp_iterations_ms: float = 0.0


@dataclass(frozen=True)
class MPCResult:
    """One RTI update and its inspectable warm trajectory."""

    controls: Mapping[str, float]
    control_vector: FloatArray
    nominal_controls: FloatArray
    nominal_states: FloatArray
    qp_status: str
    qp_iterations: int
    qp_cost: float
    predicted_peak_bank: float
    predicted_bank_violation: float
    predicted_attitude_constraint_violation: float
    bank_slack: float
    saturated: tuple[str, ...]
    timings: MPCTimings
    qp_primal_residual: float = math.nan
    qp_dual_residual: float = math.nan
    qp_rho_updates: int = 0
    qp_rho_estimate: float = math.nan
    # Tikhonov term added to the QP Hessian diagonal for this solve
    # (`HESSIAN_REGULARIZATION`): declared here so the regularization the
    # solution was computed under is visible alongside its residuals.
    hessian_regularization: float = HESSIAN_REGULARIZATION
    # Number of accepted control entries (over the whole horizon) that the
    # post-QP actuator-bound projection moved. The QP already bounds the
    # normalized step, so a nonzero count means the solver returned a step
    # outside its own bounds by more than roundoff and the projection, not
    # the QP, decided those values.
    clipped_controls: int = 0
    # Partial HPIPM condensing is currently safety-shadowed by an uncondensed
    # solve. These fields keep the reduction attempt visible to benchmarks.
    qp_used_uncondensed_fallback: bool = False
    qp_condensed_candidate_valid: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))
        for name in ("control_vector", "nominal_controls", "nominal_states"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class MPC:
    """Sparse direct-multiple-shooting RTI over a Manta world.

    ``controlled`` defaults to every craft with at least one input.  Inputs on
    other crafts remain at their declared defaults while their states and all
    inter-craft couplings remain part of the predicted world dynamics.
    """

    formulation = "sparse_direct_multiple_shooting_rti"

    def __init__(
        self,
        world: Any,
        *,
        u_bounds: Mapping[str, tuple[float, float]],
        controlled: Sequence[str] | None = None,
        horizon: int = 100,
        dt: float = 0.1,
        substeps: int = 2,
        control_rate_weight: float = 0.2,
        effort_weight: float = 2e-3,
        position_weight: float = 2.0,
        velocity_weight: float = 1.0 / 0.30**2,
        terminal_multiplier: float = 4.0,
        bank_slack_weight: float = 1e4,
        trust_region: float = 0.5,
        compile: bool = False,
        qp_backend: str = "osqp",
        qp_options: Mapping[str, Any] | None = None,
    ) -> None:
        if horizon < 2 or dt <= 0.0 or substeps < 1:
            raise ValueError("MPC requires horizon >= 2, dt > 0, substeps >= 1")
        nonnegative = {
            "control_rate_weight": control_rate_weight,
            "effort_weight": effort_weight,
            "position_weight": position_weight,
            "velocity_weight": velocity_weight,
        }
        if any(not math.isfinite(float(v)) or float(v) < 0.0
               for v in nonnegative.values()):
            raise ValueError("MPC cost weights must be finite and non-negative")
        if (not math.isfinite(float(terminal_multiplier))
                or terminal_multiplier < 1.0
                or not math.isfinite(float(bank_slack_weight))
                or bank_slack_weight <= 0.0
                or not 0.0 < trust_region <= 1.0):
            raise ValueError("invalid terminal, bank-slack, or trust-region setting")

        from ..sim import Sim

        sim_transform = Sim(world)
        self.world = sim_transform.world
        self.model = sim_transform.model
        sim_module = sim_transform.module()
        step = sim_module.functions["step"].expand()
        self.spec = sim_module.state.fields[0].spec
        if self.spec is None:
            raise RuntimeError("MPC requires a manifold state specification")
        self._x_init = np.asarray(
            sim_module.state.fields[0].init, dtype=float).reshape(-1)
        u_port = sim_module.port("u")
        all_input_names = tuple(field.name for field in u_port.fields)
        all_defaults = np.asarray(
            [float(field.default) for field in u_port.fields], dtype=float)
        craft_names = tuple(craft.name for craft in self.world.crafts)
        if not craft_names:
            raise ValueError("MPC world has no crafts")

        inputs_by_craft = {
            craft: tuple(i for i, name in enumerate(all_input_names)
                         if name.startswith(craft + "."))
            for craft in craft_names
        }
        selected = tuple(controlled) if controlled is not None else tuple(
            craft for craft in craft_names if inputs_by_craft[craft])
        if not selected:
            raise ValueError("MPC has no controlled crafts with actuator inputs")
        if len(set(selected)) != len(selected):
            raise ValueError("controlled craft names must be unique")
        unknown = sorted(set(selected) - set(craft_names))
        if unknown:
            raise KeyError(f"unknown controlled craft(s) {unknown}")
        without_inputs = [craft for craft in selected if not inputs_by_craft[craft]]
        if without_inputs:
            raise ValueError(f"controlled craft(s) have no inputs: {without_inputs}")

        controlled_indices = tuple(
            index for craft in selected for index in inputs_by_craft[craft])
        self.input_names = tuple(all_input_names[i] for i in controlled_indices)
        self.controlled_crafts = selected
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.nx = self.spec.ambient_dim
        self.ndx = self.spec.tangent_dim
        self.nu = len(self.input_names)
        self.nc = len(self.controlled_crafts)
        self._all_defaults = all_defaults
        self._controlled_indices = controlled_indices

        bounds = [self._resolve_bound(name, u_bounds) for name in self.input_names]
        self.u_lo = np.asarray([b[0] for b in bounds], dtype=float)
        self.u_hi = np.asarray([b[1] for b in bounds], dtype=float)
        if (not np.all(np.isfinite(self.u_lo))
                or not np.all(np.isfinite(self.u_hi))
                or np.any(self.u_lo >= self.u_hi)):
            raise ValueError("every MPC input bound must be finite with lo < hi")
        self._authority = np.maximum(np.abs(self.u_lo), np.abs(self.u_hi))
        if np.any(self._authority <= 0.0):
            raise ValueError("every controlled input must have non-zero authority")

        self.control_rate_weight = float(control_rate_weight)
        self.effort_weight = float(effort_weight)
        self.position_weight = float(position_weight)
        self.velocity_weight = float(velocity_weight)
        self.terminal_multiplier = float(terminal_multiplier)
        self.bank_slack_weight = float(bank_slack_weight)
        self.trust_region = float(trust_region)
        self.compiled = bool(compile)
        self.qp_backend = str(qp_backend).lower()
        if self.qp_backend not in ("osqp", "hpipm"):
            raise ValueError("qp_backend must be 'osqp' or 'hpipm'")
        if self.qp_backend == "hpipm" and not self.compiled:
            raise ValueError("the HPIPM backend requires compile=True")
        self.qp_options = dict(qp_options or {})

        self._f, self._fj = self._build_dynamics(
            step, all_defaults, controlled_indices, int(substeps))
        self._rollout_kernel = self._f.mapaccum(
            "mpc_rollout", self.horizon, {"base": 10})
        self._rollout_linearize_kernel = self._fj.mapaccum(
            "mpc_rollout_linearize", self.horizon, {"base": 10})
        self._feature = self._build_feature_function()
        self._attitude = self._build_attitude_function()
        self._objective = self._build_objective_function()
        self._attitude_map = self._attitude.map(self.horizon)
        self._objective_map = self._objective.map(self.horizon)
        self._accepted_rollout_kernel = self._build_accepted_rollout_function()
        if compile:
            from ..codegen.numpy import (
                DEFAULT_COMPILATION_TIMEOUT_S,
                compile_functions,
            )
            compiled = compile_functions(
                {"attitude_horizon": self._attitude_map,
                 "objective_horizon": self._objective_map,
                 "rollout": self._rollout_kernel,
                 "rollout_linearize": self._rollout_linearize_kernel,
                 "accepted_rollout": self._accepted_rollout_kernel},
                max_instructions=None,
                optimization="runtime",
                timeout_s=DEFAULT_COMPILATION_TIMEOUT_S,
            )
            self._attitude_map = compiled["attitude_horizon"]
            self._objective_map = compiled["objective_horizon"]
            self._rollout_kernel = compiled["rollout"]
            self._rollout_linearize_kernel = compiled[
                "rollout_linearize"]
            self._accepted_rollout_kernel = compiled["accepted_rollout"]
        n, nc = self.horizon, self.nc
        self._positions = np.empty((n, nc, 3))
        self._world_targets = np.empty_like(self._positions)
        self._body_targets = np.zeros_like(self._positions)
        self._body_weights = np.zeros_like(self._positions)
        self._optional_world_targets = np.zeros_like(self._positions)
        self._optional_world_weights = np.zeros_like(self._positions)
        self._rate_targets = np.zeros_like(self._positions)
        self._rate_weights = np.zeros_like(self._positions)
        self._attitude_references = np.zeros((n, nc, 4))
        self._attitude_weights = np.zeros((n, nc, 3))
        self._reference_up = np.empty_like(self._positions)
        self._bank_limits = np.empty((n, nc))
        self._slack_scales = np.empty((n, nc))
        self._multipliers = np.ones((1, n))
        self._multipliers[0, -1] = self.terminal_multiplier
        self._identity_u = np.eye(self.nu)
        self._control_hessians = np.empty((n, self.nu, self.nu))
        self._control_gradients = np.empty((n, self.nu))
        self._rate_errors = np.empty((n, self.nu))
        self._accepted_states = np.empty((n + 1, self.nx))
        self._accepted_controls = np.empty((n, self.nu))

        self._U = np.zeros((self.horizon, self.nu))
        self._last_u = np.zeros(self.nu)
        self._prepare_structure(include_attitude_constraints=False)
        self.last_result: MPCResult | None = None

    @staticmethod
    def _resolve_bound(
        full: str, bounds: Mapping[str, tuple[float, float]],
    ) -> tuple[float, float]:
        if full in bounds:
            return tuple(float(v) for v in bounds[full])  # type: ignore[return-value]
        suffix = full.split(".", 1)[1]
        if suffix in bounds:
            return tuple(float(v) for v in bounds[suffix])  # type: ignore[return-value]
        raise KeyError(f"MPC has no bounds for controlled input {full!r}")

    def _build_dynamics(
        self, step: ca.Function, defaults: FloatArray,
        controlled_indices: tuple[int, ...], substeps: int,
    ) -> tuple[ca.Function, ca.Function]:
        x = ca.MX.sym("mpc_x", self.nx)
        u = ca.MX.sym("mpc_u", self.nu)
        full_u = ca.MX(ca.DM(defaults))
        for local, absolute in enumerate(controlled_indices):
            full_u[absolute] = u[local]
        n_noise = step.size_in(2)[0]
        zero_noise = ca.MX.zeros(n_noise)
        dt_sub = self.dt / substeps

        def advance(state: Any, controls: Any) -> Any:
            current = state
            for index in range(substeps):
                result = step(current, controls, zero_noise, dt_sub,
                              index * dt_sub)
                current = result[0] if isinstance(result, tuple) else result
            return current

        x_next = advance(x, full_u)
        f = ca.Function("mpc_dynamics", [x, u], [x_next])
        dx = ca.MX.sym("mpc_dx", self.ndx)
        du = ca.MX.sym("mpc_du", self.nu)
        perturbed_full_u = ca.MX(full_u)
        for local, absolute in enumerate(controlled_indices):
            perturbed_full_u[absolute] = u[local] + du[local]
        perturbed_next = advance(self.spec.boxplus_sym(x, dx), perturbed_full_u)
        error_next = self.spec.boxminus_sym(perturbed_next, x_next)
        zeros_x = ca.MX.zeros(self.ndx)
        zeros_u = ca.MX.zeros(self.nu)
        A = ca.substitute(ca.jacobian(error_next, dx), dx, zeros_x)
        A = ca.substitute(A, du, zeros_u)
        B = ca.substitute(ca.jacobian(error_next, du), dx, zeros_x)
        B = ca.substitute(B, du, zeros_u)
        fj = ca.Function("mpc_jacobian", [x, u], [x_next, A, B])
        return f, fj

    def _build_feature_function(self) -> ca.Function:
        x = ca.MX.sym("mpc_feature_x", self.nx)
        dx = ca.MX.sym("mpc_feature_dx", self.ndx)

        def features(value: Any) -> Any:
            chunks = []
            for craft in self.controlled_crafts:
                position = self.spec.slot(f"{craft}.position")
                orientation = self.spec.slot(f"{craft}.orientation")
                velocity = self.spec.slot(f"{craft}.velocity")
                rates = self.spec.slot(f"{craft}.angular_velocity")
                p = value[position.ambient_offset:position.ambient_offset + 3]
                q = value[
                    orientation.ambient_offset:orientation.ambient_offset + 4]
                v = value[velocity.ambient_offset:velocity.ambient_offset + 3]
                omega = value[rates.ambient_offset:rates.ambient_offset + 3]
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                rotation = ca.vertcat(
                    ca.horzcat(1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),
                               2*(qx*qz+qw*qy)),
                    ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),
                               2*(qy*qz-qw*qx)),
                    ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx),
                               1-2*(qx*qx+qy*qy)),
                )
                body_velocity = rotation.T @ v
                chunks.append(ca.vertcat(
                    p, rotation[:, 0], body_velocity, v,
                    rotation[:, 1], omega))
            return ca.vertcat(*chunks)

        nominal = features(x)
        perturbed = features(self.spec.boxplus_sym(x, dx))
        jacobian = ca.substitute(
            ca.jacobian(perturbed, dx), dx, ca.MX.zeros(self.ndx))
        return ca.Function("mpc_features", [x], [nominal, jacobian])

    def _build_attitude_function(self) -> ca.Function:
        """Masked SO(3) residuals in each craft's body-axis coordinates."""
        x = ca.MX.sym("mpc_attitude_x", self.nx)
        references = ca.MX.sym("mpc_attitude_reference", 4*self.nc)
        dx = ca.MX.sym("mpc_attitude_dx", self.ndx)

        def errors(value: Any) -> Any:
            chunks = []
            for craft_index, craft in enumerate(self.controlled_crafts):
                orientation = self.spec.slot(f"{craft}.orientation")
                q = value[
                    orientation.ambient_offset:orientation.ambient_offset+4]
                q_ref = references[4*craft_index:4*(craft_index+1)]
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                rotation = ca.vertcat(
                    ca.horzcat(1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),
                               2*(qx*qz+qw*qy)),
                    ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),
                               2*(qy*qz-qw*qx)),
                    ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx),
                               1-2*(qx*qx+qy*qy)),
                )
                error_world = orientation.manifold.boxminus_sym(q, q_ref)
                chunks.append(rotation.T @ error_world)
            return ca.vertcat(*chunks)

        nominal = errors(x)
        perturbed = errors(self.spec.boxplus_sym(x, dx))
        jacobian = ca.substitute(
            ca.jacobian(perturbed, dx), dx, ca.MX.zeros(self.ndx))
        return ca.Function(
            "mpc_attitude_error", [x, references], [nominal, jacobian])

    def _build_accepted_rollout_function(self) -> ca.Function:
        """Roll out an accepted plan and return only its bank-relevant axes."""
        x = ca.MX.sym("mpc_accepted_x", self.nx)
        controls = ca.MX.sym(
            "mpc_accepted_controls", self.nu, self.horizon)
        raw_states = self._rollout_kernel(x, controls)
        if isinstance(raw_states, tuple):
            raw_states = raw_states[0]
        feature_result = self._feature.map(self.horizon)(raw_states)
        raw_features = (
            feature_result[0]
            if isinstance(feature_result, tuple) else feature_result)
        body_left_rows = [
            row
            for craft_index in range(self.nc)
            for row in range(18*craft_index+12, 18*craft_index+15)
        ]
        body_left = raw_features[body_left_rows, :]
        return ca.Function(
            "mpc_accepted_rollout", [x, controls],
            [raw_states, body_left])

    def _build_objective_function(self) -> ca.Function:
        """One stage's state cost and bank linearization as a native kernel."""
        x = ca.MX.sym("mpc_objective_x", self.nx)
        positions = ca.MX.sym("mpc_positions", 3*self.nc)
        world_targets = ca.MX.sym("mpc_world_targets", 3*self.nc)
        body_targets = ca.MX.sym("mpc_body_targets", 3*self.nc)
        body_weights = ca.MX.sym("mpc_body_weights", 3*self.nc)
        optional_world_targets = ca.MX.sym(
            "mpc_optional_world_targets", 3*self.nc)
        optional_world_weights = ca.MX.sym(
            "mpc_optional_world_weights", 3*self.nc)
        rate_targets = ca.MX.sym("mpc_rate_targets", 3*self.nc)
        rate_weights = ca.MX.sym("mpc_rate_weights", 3*self.nc)
        orientation_targets = ca.MX.sym(
            "mpc_orientation_targets", 4*self.nc)
        attitude_weights = ca.MX.sym("mpc_attitude_weights", 3*self.nc)
        up = ca.MX.sym("mpc_up", 3*self.nc)
        multiplier = ca.MX.sym("mpc_multiplier")

        features, feature_jacobian = self._feature(x)
        attitudes, attitude_jacobian = self._attitude(
            x, orientation_targets)
        hessian = ca.MX.zeros(self.ndx, self.ndx)
        gradient = ca.MX.zeros(self.ndx)
        bank_values = []
        bank_jacobians = []

        def add_cost(
            feature: Any, jacobian: Any, target: Any, weights: Any,
        ) -> None:
            nonlocal hessian, gradient
            weighted = ca.diag(weights)
            error = feature-target
            hessian += 2.0*multiplier*(jacobian.T@weighted@jacobian)
            gradient += 2.0*multiplier*(jacobian.T@weighted@error)

        for craft_index in range(self.nc):
            feature_start = 18*craft_index
            feature = features[feature_start:feature_start+18]
            J = feature_jacobian[feature_start:feature_start+18, :]
            vector = slice(3*craft_index, 3*(craft_index+1))
            add_cost(
                feature[:3], J[:3, :], positions[vector],
                self.position_weight*ca.MX.ones(3))
            add_cost(
                feature[9:12], J[9:12, :], world_targets[vector],
                self.velocity_weight*ca.MX.ones(3))
            add_cost(
                feature[6:9], J[6:9, :], body_targets[vector],
                body_weights[vector])
            add_cost(
                feature[9:12], J[9:12, :],
                optional_world_targets[vector],
                optional_world_weights[vector])
            add_cost(
                feature[15:18], J[15:18, :], rate_targets[vector],
                rate_weights[vector])
            attitude_start = 3*craft_index
            add_cost(
                attitudes[attitude_start:attitude_start+3],
                attitude_jacobian[attitude_start:attitude_start+3, :],
                ca.MX.zeros(3), attitude_weights[vector])
            body_left = feature[12:15]
            body_left_jacobian = J[12:15, :]
            bank_values.append(ca.dot(body_left, up[vector]))
            bank_jacobians.append(up[vector].T@body_left_jacobian)
        hessian = 0.5*(hessian+hessian.T)
        return ca.Function(
            "mpc_objective", [
                x, positions, world_targets,
                body_targets, body_weights,
                optional_world_targets, optional_world_weights,
                rate_targets, rate_weights,
                orientation_targets, attitude_weights, up, multiplier,
            ], [
                hessian, gradient, ca.vertcat(*bank_values),
                ca.vertcat(*bank_jacobians),
            ])

    def _prepare_structure(self, *, include_attitude_constraints: bool) -> None:
        previous_qp = getattr(self, "_qp", None)
        close = getattr(previous_qp, "close", None)
        if callable(close):
            close()
        self._with_attitude_constraints = bool(include_attitude_constraints)
        n, ndx, nu, nc = self.horizon, self.ndx, self.nu, self.nc
        self._state_nvar = n * ndx
        self._control_nvar = n * nu
        self._control_offset = self._state_nvar
        self._slack_offset = self._control_offset + self._control_nvar
        self._nslack = n * nc
        self._nvar = self._slack_offset + self._nslack
        self._dynamics_ncon = n * ndx
        self._upper_bank_offset = self._dynamics_ncon
        self._lower_bank_offset = self._upper_bank_offset + n * nc
        self._attitude_constraint_offset = self._lower_bank_offset + n * nc
        self._ncon = self._attitude_constraint_offset
        if self._with_attitude_constraints:
            self._ncon += 3 * n * nc

        h_pattern = np.zeros((self._nvar, self._nvar), dtype=bool)
        a_pattern = np.zeros((self._ncon, self._nvar), dtype=bool)
        for k in range(n):
            state = self._state_slice(k)
            control = self._control_slice(k)
            h_pattern[state, state] = True
            h_pattern[control, control] = True
            if k > 0:
                previous = self._control_slice(k - 1)
                h_pattern[control, previous] = True
                h_pattern[previous, control] = True
            dynamics = slice(k * ndx, (k + 1) * ndx)
            a_pattern[dynamics, state] = np.eye(ndx, dtype=bool)
            if k > 0:
                a_pattern[dynamics, self._state_slice(k - 1)] = True
            a_pattern[dynamics, control] = True
            for craft_index in range(nc):
                slack = self._slack_index(k, craft_index)
                h_pattern[slack, slack] = True
                upper = self._bank_row(True, k, craft_index)
                lower = self._bank_row(False, k, craft_index)
                a_pattern[upper, state] = True
                a_pattern[lower, state] = True
                a_pattern[upper, slack] = True
                a_pattern[lower, slack] = True
                if self._with_attitude_constraints:
                    attitude_rows = self._attitude_constraint_rows(
                        k, craft_index)
                    a_pattern[attitude_rows, state] = True

        def sparsity(pattern: FloatArray) -> ca.Sparsity:
            rows, columns = np.nonzero(pattern)
            return ca.Sparsity.triplet(
                pattern.shape[0], pattern.shape[1],
                rows.tolist(), columns.tolist())

        self._h_sparsity = sparsity(h_pattern)
        self._a_sparsity = sparsity(a_pattern)
        self._index_sparse_blocks()
        if self.compiled:
            if self.qp_backend == "osqp":
                self._prepare_native_qp()
            else:
                self._prepare_hpipm()
        else:
            self._qp = ca.conic(
                "manta_mpc_qp", "osqp",
                {"h": self._h_sparsity, "a": self._a_sparsity},
                {
                    "verbose": False,
                    "error_on_fail": False,
                    "warm_start_primal": True,
                    "warm_start_dual": True,
                    "osqp": {
                        "verbose": False, "eps_abs": 2e-3,
                        "eps_rel": 2e-3, "max_iter": 800,
                        "polish": False, "check_termination": 5,
                    },
                },
            )
        self._qp_x = np.zeros(self._nvar)
        self._qp_lam_x = np.zeros(self._nvar)
        self._qp_lam_a = np.zeros(self._ncon)
        self._H = np.zeros(self._h_sparsity.nnz())
        self._g = np.zeros(self._nvar)
        self._A_values = np.zeros(self._a_sparsity.nnz())
        self._constraint_lo = np.empty(self._ncon)
        self._constraint_hi = np.empty(self._ncon)
        self._lower = np.empty(self._nvar)
        self._upper = np.empty(self._nvar)
        self._constraint_references = np.zeros(
            (self.horizon, self.nc, 4))
        self._constraint_attitudes = np.zeros(
            (self.horizon, self.nc, 3))
        self._constraint_attitude_jacobians = np.zeros(
            (self.horizon, self.nc, 3, self.ndx))
        general_count = 2*self.nc + (
            3*self.nc if self._with_attitude_constraints else 0)
        self._hp_general_jacobians = np.zeros(
            (self.horizon, general_count, self.ndx))
        self._hp_general_lower = np.empty((self.horizon, general_count))
        self._hp_general_upper = np.empty((self.horizon, general_count))
        self._hp_general_lower_clipped = np.empty_like(
            self._hp_general_lower)
        self._hp_general_upper_clipped = np.empty_like(
            self._hp_general_upper)
        self._hp_dynamics_B = np.empty(
            (self.horizon, self.ndx, self.nu))
        self._hp_nominal_controls = np.empty(
            (self.horizon, self.nu))
        self._hp_previous_control = np.empty(self.nu)
        if self.compiled and self.qp_backend == "osqp":
            self._native_lower = np.empty(self._ncon + self._nvar)
            self._native_upper = np.empty(self._ncon + self._nvar)
            self._native_dual = np.empty(self._ncon + self._nvar)

    @staticmethod
    def _column_pointers(
        columns: Sequence[int], count: int,
    ) -> npt.NDArray[np.int64]:
        occurrences = np.bincount(
            np.asarray(columns, dtype=np.int64), minlength=count)
        return np.r_[0, np.cumsum(occurrences)].astype(np.int64)

    def _prepare_native_qp(self) -> None:
        from ._osqp import NativeOSQP

        h_rows, h_columns = self._h_sparsity.get_triplet()
        upper = np.asarray([
            index for index, (row, column) in enumerate(
                zip(h_rows, h_columns, strict=True))
            if row <= column
        ], dtype=np.int64)
        p_rows = np.asarray(h_rows, dtype=np.int64)[upper]
        p_columns = np.asarray(h_columns, dtype=np.int64)[upper]
        self._native_p_from_h = upper
        Pp = self._column_pointers(p_columns, self._nvar)

        a_rows, a_columns = self._a_sparsity.get_triplet()
        a_rows_array = np.asarray(a_rows, dtype=np.int64)
        a_columns_array = np.asarray(a_columns, dtype=np.int64)
        base_pointers = self._column_pointers(a_columns_array, self._nvar)
        combined_rows: list[int] = []
        combined_base_positions = np.empty(len(a_rows), dtype=np.int64)
        identity_positions = np.empty(self._nvar, dtype=np.int64)
        combined_pointers = [0]
        for column in range(self._nvar):
            for base_index in range(
                    int(base_pointers[column]),
                    int(base_pointers[column+1])):
                combined_base_positions[base_index] = len(combined_rows)
                combined_rows.append(int(a_rows_array[base_index]))
            identity_positions[column] = len(combined_rows)
            combined_rows.append(self._ncon + column)
            combined_pointers.append(len(combined_rows))
        self._native_a_from_base = combined_base_positions
        self._native_a_identity = identity_positions
        self._native_a_values = np.zeros(len(combined_rows))
        self._native_a_values[identity_positions] = 1.0
        self._qp = NativeOSQP(
            self._nvar, self._ncon+self._nvar,
            Pp, p_rows,
            np.asarray(combined_pointers, dtype=np.int64),
            np.asarray(combined_rows, dtype=np.int64),
            **self.qp_options,
        )

    def _prepare_hpipm(self) -> None:
        from ._hpipm import NativeHPIPM

        options = dict(self.qp_options)
        condense_to = int(options.pop("condense_to", 0))
        tolerance = float(options.pop("tolerance", 2e-3))
        max_iter = int(options.pop("max_iter", 30))
        if options:
            raise TypeError(
                f"unknown HPIPM option(s): {', '.join(sorted(options))}")
        self._qp = NativeHPIPM(
            self.horizon, self.ndx, self.nu, self.nc,
            attitude_rows=3*self.nc if self._with_attitude_constraints else 0,
            condense_to=condense_to,
            effort_weight=self.effort_weight,
            control_rate_weight=self.control_rate_weight,
            bank_slack_weight=self.bank_slack_weight,
            tolerance=tolerance,
            max_iter=max_iter,
        )

    def _ensure_structure(self, reference: MPCReference) -> None:
        required = any(
            value.constraint_orientations is not None
            for value in reference.crafts.values())
        if required != self._with_attitude_constraints:
            self._prepare_structure(include_attitude_constraints=required)

    def _index_sparse_blocks(self) -> None:
        h_rows, h_cols = self._h_sparsity.get_triplet()
        a_rows, a_cols = self._a_sparsity.get_triplet()
        h_lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(
            zip(h_rows, h_cols, strict=True))}
        a_lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(
            zip(a_rows, a_cols, strict=True))}

        def block(lookup: Mapping[tuple[int, int], int], rows: range,
                  columns: range) -> npt.NDArray[np.int64]:
            return np.asarray([[lookup[(r, c)] for c in columns]
                               for r in rows], dtype=np.int64)

        self._h_state_blocks = []
        self._h_control_blocks = []
        self._h_control_previous = []
        self._h_previous_control = []
        self._a_dynamics_state_diagonal = []
        self._a_dynamics_previous = []
        self._a_dynamics_control = []
        self._a_bank_state: dict[tuple[bool, int, int], npt.NDArray[np.int64]] = {}
        self._a_bank_slack: dict[tuple[bool, int, int], int] = {}
        self._a_attitude_constraint: dict[
            tuple[int, int], npt.NDArray[np.int64]] = {}
        for k in range(self.horizon):
            state = self._state_slice(k)
            control = self._control_slice(k)
            sr = range(state.start, state.stop)
            cr = range(control.start, control.stop)
            dr = range(k * self.ndx, (k + 1) * self.ndx)
            self._h_state_blocks.append(block(h_lookup, sr, sr))
            self._h_control_blocks.append(block(h_lookup, cr, cr))
            self._a_dynamics_state_diagonal.append(np.asarray(
                [a_lookup[(r, c)] for r, c in zip(dr, sr, strict=True)],
                dtype=np.int64))
            self._a_dynamics_control.append(block(a_lookup, dr, cr))
            if k == 0:
                self._h_control_previous.append(None)
                self._h_previous_control.append(None)
                self._a_dynamics_previous.append(None)
            else:
                previous_control = self._control_slice(k - 1)
                previous_state = self._state_slice(k - 1)
                pcr = range(previous_control.start, previous_control.stop)
                self._h_control_previous.append(block(h_lookup, cr, pcr))
                self._h_previous_control.append(block(h_lookup, pcr, cr))
                self._a_dynamics_previous.append(block(
                    a_lookup, dr, range(previous_state.start,
                                        previous_state.stop)))
            for craft_index in range(self.nc):
                slack = self._slack_index(k, craft_index)
                for upper in (True, False):
                    row = self._bank_row(upper, k, craft_index)
                    self._a_bank_state[(upper, k, craft_index)] = np.asarray(
                        [a_lookup[(row, column)] for column in sr],
                        dtype=np.int64)
                    self._a_bank_slack[(upper, k, craft_index)] = a_lookup[
                        (row, slack)]
                if self._with_attitude_constraints:
                    rows = self._attitude_constraint_rows(k, craft_index)
                    self._a_attitude_constraint[(k, craft_index)] = block(
                        a_lookup, range(rows.start, rows.stop), sr)
        self._h_diagonal = np.asarray(
            [h_lookup[(i, i)] for i in range(self._nvar)], dtype=np.int64)
        self._h_slack = np.asarray(
            [h_lookup[(i, i)] for i in range(
                self._slack_offset, self._nvar)], dtype=np.int64)
        self._h_state_block_array = np.stack(self._h_state_blocks)
        self._h_control_block_array = np.stack(self._h_control_blocks)
        self._a_dynamics_diagonal_array = np.stack(
            self._a_dynamics_state_diagonal)
        self._a_dynamics_control_array = np.stack(
            self._a_dynamics_control)
        self._a_dynamics_previous_array = np.stack([
            value for value in self._a_dynamics_previous[1:]
            if value is not None])
        if self.horizon > 1:
            self._h_control_previous_array = np.stack([
                value for value in self._h_control_previous[1:]
                if value is not None])
            self._h_previous_control_array = np.stack([
                value for value in self._h_previous_control[1:]
                if value is not None])
        self._a_bank_state_array = {
            upper: np.stack([
                self._a_bank_state[(upper, stage, craft)]
                for stage in range(self.horizon)
                for craft in range(self.nc)
            ]).reshape(self.horizon, self.nc, self.ndx)
            for upper in (True, False)
        }
        self._a_bank_slack_array = {
            upper: np.asarray([
                self._a_bank_slack[(upper, stage, craft)]
                for stage in range(self.horizon)
                for craft in range(self.nc)
            ], dtype=np.int64).reshape(self.horizon, self.nc)
            for upper in (True, False)
        }

    def _state_slice(self, stage: int) -> slice:
        start = stage * self.ndx
        return slice(start, start + self.ndx)

    def _control_slice(self, stage: int) -> slice:
        start = self._control_offset + stage * self.nu
        return slice(start, start + self.nu)

    def _slack_index(self, stage: int, craft_index: int) -> int:
        return self._slack_offset + stage * self.nc + craft_index

    def _bank_row(self, upper: bool, stage: int, craft_index: int) -> int:
        base = self._upper_bank_offset if upper else self._lower_bank_offset
        return base + stage * self.nc + craft_index

    def _attitude_constraint_rows(
        self, stage: int, craft_index: int,
    ) -> slice:
        start = self._attitude_constraint_offset + 3 * (
            stage * self.nc + craft_index)
        return slice(start, start + 3)

    @property
    def qp_shape(self) -> tuple[int, int]:
        return self._nvar, self._ncon

    @property
    def qp_nonzeros(self) -> tuple[int, int]:
        return self._h_sparsity.nnz(), self._a_sparsity.nnz()

    def reset(self, controls: Any | None = None) -> None:
        if controls is None:
            self._U.fill(0.0)
        else:
            candidate = np.asarray(controls, dtype=float)
            if candidate.shape != self._U.shape or not np.all(np.isfinite(candidate)):
                raise ValueError(
                    f"reset controls must be finite with shape {self._U.shape}")
            self._U[:] = np.clip(candidate, self.u_lo, self.u_hi)
        self._last_u.fill(0.0)
        self._qp_x.fill(0.0)
        self._qp_lam_x.fill(0.0)
        self._qp_lam_a.fill(0.0)
        self.last_result = None

    def _reference(self, reference: Any) -> MPCReference:
        if isinstance(reference, CraftHorizonReference):
            if self.nc != 1:
                raise TypeError("a bare craft reference is valid only for one craft")
            reference = MPCReference({self.controlled_crafts[0]: reference})
        if not isinstance(reference, MPCReference):
            raise TypeError("reference must be MPCReference")
        expected, actual = set(self.controlled_crafts), set(reference.crafts)
        if actual != expected:
            raise ValueError(
                f"MPC references must exactly cover controlled crafts; "
                f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        for craft, value in reference.crafts.items():
            if value.horizon != self.horizon:
                raise ValueError(
                    f"reference for {craft!r} has horizon {value.horizon}; "
                    f"expected {self.horizon}")
        return reference

    def _rollout(self, x0: FloatArray, controls: FloatArray) -> FloatArray:
        raw = self._rollout_kernel(x0, controls.T)
        if isinstance(raw, tuple):
            raw = raw[0]
        states = np.empty((self.horizon + 1, self.nx))
        states[0] = x0
        states[1:] = np.asarray(raw, dtype=float).T
        return states

    def _rollout_linearize(
        self, x0: FloatArray, controls: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        raw_states, a_flat, b_flat = self._rollout_linearize_kernel(
            x0, controls.T)
        states = np.empty((self.horizon + 1, self.nx))
        states[0] = x0
        states[1:] = np.asarray(raw_states, dtype=float).T
        raw_a, raw_b = np.asarray(a_flat), np.asarray(b_flat)
        A = raw_a.reshape(
            self.ndx, self.horizon, self.ndx).transpose(1, 0, 2)
        B = raw_b.reshape(
            self.ndx, self.horizon, self.nu).transpose(1, 0, 2)
        return states, A, B

    def _rollout_accepted(
        self, x0: FloatArray, controls: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        raw_states, raw_body_left = self._accepted_rollout_kernel(
            x0, controls.T)
        states = self._accepted_states
        states[0] = x0
        states[1:] = np.asarray(raw_states, dtype=float).T
        body_left = np.asarray(raw_body_left, dtype=float).T.reshape(
            self.horizon, self.nc, 3)
        return states, body_left

    @staticmethod
    def _sparse(sparsity: ca.Sparsity, values: FloatArray) -> ca.DM:
        return ca.DM(sparsity, values)

    def _assemble(
        self, states: FloatArray, controls: FloatArray,
        dynamics_A: FloatArray, dynamics_B: FloatArray,
        reference: MPCReference,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray,
               FloatArray, FloatArray, FloatArray, FloatArray]:
        positions = self._positions
        world_targets = self._world_targets
        body_targets = self._body_targets
        body_weights = self._body_weights
        optional_world_targets = self._optional_world_targets
        optional_world_weights = self._optional_world_weights
        rate_targets = self._rate_targets
        rate_weights = self._rate_weights
        attitude_references = self._attitude_references
        attitude_weights = self._attitude_weights
        up = self._reference_up
        bank_limits = self._bank_limits
        for values in (
            body_targets, body_weights,
            optional_world_targets, optional_world_weights,
            rate_targets, rate_weights, attitude_weights,
        ):
            values.fill(0.0)
        for craft_index, craft in enumerate(self.controlled_crafts):
            craft_reference = reference.crafts[craft]
            positions[:, craft_index] = craft_reference.positions
            np.multiply(
                craft_reference.tangents,
                craft_reference.forward_speeds[:, None],
                out=world_targets[:, craft_index])
            up[:, craft_index] = craft_reference.up
            bank_limits[:, craft_index] = craft_reference.bank_limits
            optional_values = (
                (craft_reference.body_velocities,
                 craft_reference.body_velocity_weights,
                 body_targets, body_weights),
                (craft_reference.world_velocities,
                 craft_reference.world_velocity_weights,
                 optional_world_targets, optional_world_weights),
                (craft_reference.angular_rates,
                 craft_reference.angular_rate_weights,
                 rate_targets, rate_weights),
            )
            for values, weights, targets, target_weights in optional_values:
                if values is not None:
                    assert weights is not None
                    targets[:, craft_index] = values
                    target_weights[:, craft_index] = weights
            if craft_reference.orientations is None:
                slot = self.spec.slot(f"{craft}.orientation")
                attitude_references[:, craft_index] = states[
                    1:, slot.ambient_offset:slot.ambient_offset+4]
            else:
                attitude_references[:, craft_index] = (
                    craft_reference.orientations)
                assert craft_reference.attitude_weights is not None
                attitude_weights[:, craft_index] = (
                    craft_reference.attitude_weights)
        def mapped(values: FloatArray) -> FloatArray:
            return values.reshape(self.horizon, -1).T

        raw_hessians, raw_gradients, raw_banks, raw_bank_jacobians = (
            self._objective_map(
                states[1:].T,
                mapped(positions), mapped(world_targets),
                mapped(body_targets), mapped(body_weights),
                mapped(optional_world_targets),
                mapped(optional_world_weights), mapped(rate_targets),
                mapped(rate_weights), mapped(attitude_references),
                mapped(attitude_weights), mapped(up), self._multipliers))
        raw_hessians = np.asarray(raw_hessians, dtype=float)
        state_hessians = raw_hessians.reshape(
            self.ndx, self.horizon, self.ndx).transpose(1, 0, 2)
        state_gradients = np.asarray(raw_gradients, dtype=float).T
        bank_values = np.asarray(raw_banks, dtype=float).T
        raw_bank_jacobians = np.asarray(raw_bank_jacobians, dtype=float)
        bank_jacobians = raw_bank_jacobians.reshape(
            self.nc, self.horizon, self.ndx).transpose(1, 0, 2)
        constraint_attitudes = self._constraint_attitudes
        constraint_attitude_jacobians = (
            self._constraint_attitude_jacobians)
        constraint_attitudes.fill(0.0)
        constraint_attitude_jacobians.fill(0.0)
        has_attitude_constraints = any(
            reference.crafts[craft].constraint_orientations is not None
            for craft in self.controlled_crafts)
        if has_attitude_constraints:
            constraint_references = self._constraint_references
            constraint_references.fill(0.0)
            for craft_index, craft in enumerate(self.controlled_crafts):
                value = reference.crafts[craft].constraint_orientations
                if value is None:
                    constraint_references[:, craft_index, 0] = 1.0
                else:
                    constraint_references[:, craft_index] = value
            raw_constraints, raw_constraint_jacobians = self._attitude_map(
                states[1:].T,
                constraint_references.reshape(self.horizon, 4*self.nc).T)
            constraint_attitudes = np.asarray(
                raw_constraints, dtype=float).T.reshape(
                    self.horizon, self.nc, 3)
            raw_constraint_jacobians = np.asarray(
                raw_constraint_jacobians, dtype=float)
            constraint_attitude_jacobians[:] = (
                raw_constraint_jacobians.reshape(
                    3*self.nc, self.horizon, self.ndx)
                .transpose(1, 0, 2)
                .reshape(self.horizon, self.nc, 3, self.ndx))
        H = self._H
        g = self._g
        A_values = self._A_values
        constraint_lo = self._constraint_lo
        constraint_hi = self._constraint_hi
        lower = self._lower
        upper = self._upper
        H.fill(0.0)
        g.fill(0.0)
        A_values.fill(0.0)
        constraint_lo.fill(0.0)
        constraint_hi.fill(0.0)
        constraint_lo[self._attitude_constraint_offset:] = -np.inf
        constraint_hi[self._attitude_constraint_offset:] = np.inf
        lower.fill(-np.inf)
        upper.fill(np.inf)
        slack_scales = self._slack_scales

        if has_attitude_constraints:
            for stage in range(self.horizon):
                for craft_index, craft in enumerate(self.controlled_crafts):
                    ref = reference.crafts[craft]
                    if ref.constraint_orientations is None:
                        continue
                    assert ref.attitude_error_lower is not None
                    assert ref.attitude_error_upper is not None
                    rows = self._attitude_constraint_rows(
                        stage, craft_index)
                    residual = constraint_attitudes[stage, craft_index]
                    A_values[self._a_attitude_constraint[
                        (stage, craft_index)]] = (
                            constraint_attitude_jacobians[
                                stage, craft_index])
                    constraint_lo[rows] = (
                        ref.attitude_error_lower[stage] - residual)
                    constraint_hi[rows] = (
                        ref.attitude_error_upper[stage] - residual)

        H[self._h_state_block_array] += state_hessians
        g[:self._state_nvar].reshape(
            self.horizon, self.ndx)[:] += state_gradients
        A_values[self._a_dynamics_diagonal_array] = 1.0
        A_values[self._a_dynamics_previous_array] = -dynamics_A[1:]
        A_values[self._a_dynamics_control_array] = (
            -dynamics_B*self._authority[None, None, :])

        slack_scales[:] = np.sin(0.9*bank_limits)
        upper_rows = slice(
            self._upper_bank_offset, self._lower_bank_offset)
        lower_rows = slice(
            self._lower_bank_offset, self._attitude_constraint_offset)
        constraint_lo[upper_rows] = -np.inf
        constraint_hi[upper_rows] = (
            slack_scales-bank_values).reshape(-1)
        constraint_lo[lower_rows] = (
            -slack_scales-bank_values).reshape(-1)
        constraint_hi[lower_rows] = np.inf
        for is_upper, sign in ((True, -1.0), (False, 1.0)):
            A_values[self._a_bank_state_array[is_upper]] = bank_jacobians
            A_values[self._a_bank_slack_array[is_upper]] = sign*slack_scales

        identity_u = self._identity_u
        control_hessians = self._control_hessians
        control_hessians[:] = 2.0*self.effort_weight*identity_u
        control_gradients = self._control_gradients
        np.multiply(
            controls, 2.0*self.effort_weight,
            out=control_gradients)
        control_gradients /= self._authority[None, :]
        if self.control_rate_weight > 0.0:
            rate_hessian = 2.0*self.control_rate_weight*identity_u
            control_hessians += rate_hessian
            control_hessians[:-1] += rate_hessian
            rate_errors = self._rate_errors
            np.subtract(controls[0], self._last_u, out=rate_errors[0])
            np.subtract(controls[1:], controls[:-1], out=rate_errors[1:])
            rate_errors /= self._authority[None, :]
            control_gradients += (
                2.0*self.control_rate_weight*rate_errors)
            control_gradients[:-1] -= (
                2.0*self.control_rate_weight*rate_errors[1:])
            H[self._h_control_previous_array] -= rate_hessian
            H[self._h_previous_control_array] -= rate_hessian
        H[self._h_control_block_array] += control_hessians
        g[self._control_offset:self._slack_offset].reshape(
            self.horizon, self.nu)[:] += control_gradients
        lower_controls = lower[
            self._control_offset:self._slack_offset].reshape(
                self.horizon, self.nu)
        upper_controls = upper[
            self._control_offset:self._slack_offset].reshape(
                self.horizon, self.nu)
        lower_controls[:] = np.maximum(
            (self.u_lo[None, :]-controls)/self._authority[None, :],
            -self.trust_region)
        upper_controls[:] = np.minimum(
            (self.u_hi[None, :]-controls)/self._authority[None, :],
            self.trust_region)
        H[self._h_slack] = 2.0 * self.bank_slack_weight
        lower[self._slack_offset:] = 0.0
        H[self._h_diagonal] += HESSIAN_REGULARIZATION
        return (H, g, A_values, constraint_lo, constraint_hi,
                lower, upper, slack_scales)

    def _bank_metrics(
        self, body_left: FloatArray,
    ) -> tuple[float, float]:
        vertical = np.einsum(
            "nci,nci->nc", body_left, self._reference_up, optimize=False)
        bank = np.abs(np.arcsin(np.clip(vertical, -1.0, 1.0)))
        return (float(np.max(bank)),
                max(0.0, float(np.max(bank-self._bank_limits))))

    def _attitude_constraint_violation(
        self, states: FloatArray, reference: MPCReference,
    ) -> float:
        references = np.zeros((self.horizon, self.nc, 4))
        active: list[tuple[int, int]] = []
        for craft_index, craft in enumerate(self.controlled_crafts):
            value = reference.crafts[craft].constraint_orientations
            if value is None:
                references[:, craft_index, 0] = 1.0
            else:
                references[:, craft_index] = value
                active.extend((stage, craft_index)
                              for stage in range(self.horizon))
        if not active:
            return 0.0
        raw_errors, _ = self._attitude_map(
            states[1:].T,
            references.reshape(self.horizon, 4*self.nc).T)
        errors = np.asarray(raw_errors, dtype=float).T.reshape(
            self.horizon, self.nc, 3)
        violation = 0.0
        for stage, craft_index in active:
            ref = reference.crafts[self.controlled_crafts[craft_index]]
            assert ref.attitude_error_lower is not None
            assert ref.attitude_error_upper is not None
            violation = max(
                violation,
                float(np.max(ref.attitude_error_lower[stage]
                             - errors[stage, craft_index])),
                float(np.max(errors[stage, craft_index]
                             - ref.attitude_error_upper[stage])),
            )
        return max(0.0, violation)

    @staticmethod
    def _advance_blocks(
        values: FloatArray, offset: int, count: int, width: int,
        stages: float,
    ) -> None:
        """Advance a stage-major warm start, interpolating between nodes."""
        shaped = values[offset:offset+count*width].reshape(count, width)
        previous = shaped.copy()
        sample = np.minimum(np.arange(count, dtype=float) + stages, count-1)
        lower = np.floor(sample).astype(int)
        upper = np.minimum(lower+1, count-1)
        fraction = (sample-lower)[:, None]
        shaped[:] = ((1.0-fraction)*previous[lower]
                     + fraction*previous[upper])

    def _advance_qp_warm_start(self, stages: float) -> None:
        """Align the previous QP solution with the advanced nonlinear plan."""
        for values in (self._qp_x, self._qp_lam_x):
            self._advance_blocks(
                values, 0, self.horizon, self.ndx, stages)
            self._advance_blocks(
                values, self._control_offset, self.horizon, self.nu, stages)
            self._advance_blocks(
                values, self._slack_offset, self.horizon, self.nc, stages)
        self._advance_blocks(
            self._qp_lam_a, 0, self.horizon, self.ndx, stages)
        self._advance_blocks(
            self._qp_lam_a, self._upper_bank_offset,
            self.horizon, self.nc, stages)
        self._advance_blocks(
            self._qp_lam_a, self._lower_bank_offset,
            self.horizon, self.nc, stages)
        if self._with_attitude_constraints:
            self._advance_blocks(
                self._qp_lam_a, self._attitude_constraint_offset,
                self.horizon, 3*self.nc, stages)

    def _solve_hpipm(
        self, H: FloatArray, g: FloatArray, A_values: FloatArray,
        constraint_lo: FloatArray, constraint_hi: FloatArray,
        lower: FloatArray, upper: FloatArray, slack_scales: FloatArray,
        controls: FloatArray, dynamics_A: FloatArray,
        dynamics_B: FloatArray,
    ) -> Any:
        """Translate the sparse RTI QP into HPIPM's stage representation."""
        attitude_rows = 3*self.nc if self._with_attitude_constraints else 0
        general_jacobians = self._hp_general_jacobians
        general_jacobians.fill(0.0)
        bank_jacobians = A_values[self._a_bank_state_array[True]]
        general_jacobians[:, :self.nc] = bank_jacobians
        general_jacobians[:, self.nc:2*self.nc] = bank_jacobians
        if self._with_attitude_constraints:
            for stage in range(self.horizon):
                for craft in range(self.nc):
                    rows = slice(2*self.nc+3*craft, 2*self.nc+3*(craft+1))
                    general_jacobians[stage, rows] = A_values[
                        self._a_attitude_constraint[(stage, craft)]]
        general_lower = self._hp_general_lower
        general_upper = self._hp_general_upper
        general_lower[:, :self.nc] = constraint_lo[
            self._upper_bank_offset:self._lower_bank_offset].reshape(
                self.horizon, self.nc)
        general_upper[:, :self.nc] = constraint_hi[
            self._upper_bank_offset:self._lower_bank_offset].reshape(
                self.horizon, self.nc)
        general_lower[:, self.nc:2*self.nc] = constraint_lo[
            self._lower_bank_offset:self._attitude_constraint_offset].reshape(
                self.horizon, self.nc)
        general_upper[:, self.nc:2*self.nc] = constraint_hi[
            self._lower_bank_offset:self._attitude_constraint_offset].reshape(
                self.horizon, self.nc)
        if self._with_attitude_constraints:
            general_lower[:, 2*self.nc:] = constraint_lo[
                self._attitude_constraint_offset:].reshape(
                    self.horizon, attitude_rows)
            general_upper[:, 2*self.nc:] = constraint_hi[
                self._attitude_constraint_offset:].reshape(
                    self.horizon, attitude_rows)
        state_hessians = H[self._h_state_block_array]
        state_gradients = g[:self._state_nvar].reshape(
            self.horizon, self.ndx)
        control_lower = lower[
            self._control_offset:self._slack_offset].reshape(
                self.horizon, self.nu)
        control_upper = upper[
            self._control_offset:self._slack_offset].reshape(
                self.horizon, self.nu)
        np.multiply(
            dynamics_B, self._authority[None, None, :],
            out=self._hp_dynamics_B)
        np.divide(
            controls, self._authority[None, :],
            out=self._hp_nominal_controls)
        np.divide(
            self._last_u, self._authority,
            out=self._hp_previous_control)
        np.clip(
            general_lower, -QP_BOUND_INFINITY, QP_BOUND_INFINITY,
            out=self._hp_general_lower_clipped)
        np.clip(
            general_upper, -QP_BOUND_INFINITY, QP_BOUND_INFINITY,
            out=self._hp_general_upper_clipped)
        return self._qp.solve(
            dynamics_A,
            self._hp_dynamics_B,
            state_hessians, state_gradients,
            self._hp_nominal_controls,
            self._hp_previous_control,
            control_lower, control_upper,
            general_jacobians,
            self._hp_general_lower_clipped,
            self._hp_general_upper_clipped,
            slack_scales, self._qp_x,
        )

    def tick(
        self, state: Any, reference: Any, *, advance_s: float | None = None,
    ) -> MPCResult:
        """Apply RTI and advance its warm start by elapsed controller time.

        ``advance_s`` is independent of the shooting-node spacing. Omitting it
        preserves the conventional one-solve-per-node behavior.
        """
        elapsed = self.dt if advance_s is None else float(advance_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("MPC advance_s must be finite and non-negative")
        advance_stages = elapsed/self.dt
        ref = self._reference(reference)
        self._ensure_structure(ref)
        x0 = self.spec.pack_any(state, base=self._x_init)
        if not np.all(np.isfinite(x0)):
            raise ValueError("MPC state must contain only finite values")
        total_start = time.perf_counter_ns()
        nominal, dynamics_A, dynamics_B = self._rollout_linearize(
            x0, self._U)
        linearized_at = time.perf_counter_ns()
        matrices = self._assemble(
            nominal, self._U, dynamics_A, dynamics_B, ref)
        assembled_at = time.perf_counter_ns()
        H, g, A_values, clo, chi, lower, upper, slack_scales = matrices
        next_lam_x = self._qp_lam_x.copy()
        next_lam_a = self._qp_lam_a.copy()
        qp_used_uncondensed_fallback = False
        qp_condensed_candidate_valid: bool | None = None
        if self.compiled:
            if self.qp_backend == "osqp":
                self._native_a_values[self._native_a_from_base] = A_values
                self._native_lower[:self._ncon] = clo
                self._native_lower[self._ncon:] = lower
                self._native_upper[:self._ncon] = chi
                self._native_upper[self._ncon:] = upper
                np.clip(
                    self._native_lower, -QP_BOUND_INFINITY, QP_BOUND_INFINITY,
                    out=self._native_lower)
                np.clip(
                    self._native_upper, -QP_BOUND_INFINITY, QP_BOUND_INFINITY,
                    out=self._native_upper)
                self._native_dual[:self._ncon] = self._qp_lam_a
                self._native_dual[self._ncon:] = self._qp_lam_x
                native = self._qp.solve(
                    H[self._native_p_from_h], self._native_a_values, g,
                    self._native_lower, self._native_upper, self._qp_x,
                    self._native_dual)
            else:
                native = self._solve_hpipm(
                    H, g, A_values, clo, chi, lower, upper, slack_scales,
                    self._U, dynamics_A, dynamics_B)
            status = native.status
            if not native.success:
                residual_detail = ""
                if self.qp_backend == "hpipm":
                    residual_detail = (
                        f" (stationarity={native.stationarity_residual:.3g}, "
                        f"equality={native.equality_residual:.3g}, "
                        f"inequality={native.inequality_residual:.3g}, "
                        f"complementarity={native.complementarity_residual:.3g})")
                raise RuntimeError(
                    f"MPC RTI QP failed: {status}{residual_detail}")
            qp_step = native.x
            if self.qp_backend == "osqp":
                next_lam_a = np.asarray(native.y[:self._ncon], dtype=float).copy()
                next_lam_x = np.asarray(native.y[self._ncon:], dtype=float).copy()
            qp_cost = native.cost
            iterations = native.iterations
            qp_update_ms = native.update_ms
            qp_iterations_ms = native.iteration_ms
            if self.qp_backend == "osqp":
                qp_primal_residual = native.primal_residual
                qp_dual_residual = native.dual_residual
                qp_rho_updates = native.rho_updates
                qp_rho_estimate = native.rho_estimate
            else:
                qp_primal_residual = max(
                    native.equality_residual, native.inequality_residual)
                qp_dual_residual = native.stationarity_residual
                qp_rho_updates = 0
                qp_rho_estimate = math.nan
                qp_used_uncondensed_fallback = bool(
                    native.used_uncondensed_fallback)
                qp_condensed_candidate_valid = (
                    native.condensed_candidate_valid)
        else:
            result = self._qp(
                h=self._sparse(self._h_sparsity, H), g=ca.DM(g),
                a=self._sparse(self._a_sparsity, A_values),
                lba=ca.DM(clo), uba=ca.DM(chi),
                lbx=ca.DM(lower), ubx=ca.DM(upper),
                x0=ca.DM(self._qp_x), lam_x0=ca.DM(self._qp_lam_x),
                lam_a0=ca.DM(self._qp_lam_a))
            stats = self._qp.stats()
            status = str(stats.get("return_status", "unknown"))
            if not (bool(stats.get("success", False))
                    or status.lower().startswith("solved")):
                raise RuntimeError(f"MPC RTI QP failed: {status}")
            qp_step = np.asarray(result["x"], dtype=float).reshape(-1)
            next_lam_x = np.asarray(
                result["lam_x"], dtype=float).reshape(-1).copy()
            next_lam_a = np.asarray(
                result["lam_a"], dtype=float).reshape(-1).copy()
            qp_cost = float(result["cost"])
            iterations = max(
                0, int(stats.get(
                    "iter_count", stats.get("iter", 0)) or 0))
            qp_update_ms = 0.0
            qp_iterations_ms = (time.perf_counter_ns()-assembled_at)/1e6
            qp_primal_residual = math.nan
            qp_dual_residual = math.nan
            qp_rho_updates = 0
            qp_rho_estimate = math.nan
        solved_at = time.perf_counter_ns()
        if not np.all(np.isfinite(qp_step)):
            raise RuntimeError("MPC RTI QP returned a nonfinite step")
        if (not np.all(np.isfinite(next_lam_x))
                or not np.all(np.isfinite(next_lam_a))):
            raise RuntimeError("MPC RTI QP returned nonfinite dual state")
        normalized_delta_u = qp_step[
            self._control_offset:self._slack_offset].reshape(
                self.horizon, self.nu)
        accepted_controls = self._accepted_controls
        np.multiply(
            normalized_delta_u, self._authority,
            out=accepted_controls)
        accepted_controls += self._U
        # The QP bounds the normalized step to the actuator limits; this
        # projection removes solver-tolerance overshoot. Anything beyond
        # roundoff is reported as `clipped_controls` rather than absorbed.
        unclipped = accepted_controls.copy()
        np.clip(
            accepted_controls, self.u_lo, self.u_hi,
            out=accepted_controls)
        clipped_controls = int(np.count_nonzero(
            np.abs(unclipped - accepted_controls) > 1e-12))
        accepted_states, accepted_body_left = self._rollout_accepted(
            x0, accepted_controls)
        if (not np.all(np.isfinite(accepted_controls))
                or not np.all(np.isfinite(accepted_states))
                or not np.all(np.isfinite(accepted_body_left))):
            raise RuntimeError("MPC accepted rollout contains nonfinite values")
        peak_bank, bank_violation = self._bank_metrics(accepted_body_left)
        attitude_constraint_violation = (
            self._attitude_constraint_violation(accepted_states, ref))
        if not all(math.isfinite(float(v)) for v in (
                qp_cost, peak_bank, bank_violation,
                attitude_constraint_violation)):
            raise RuntimeError("MPC solve produced nonfinite cost or diagnostics")
        command = accepted_controls[0].copy()
        if not np.all(np.isfinite(command)):
            raise RuntimeError("MPC command contains nonfinite values")
        # Commit the complete warm state only after the solve and accepted
        # rollout have passed every validation above.
        self._qp_x[:] = qp_step
        self._qp_lam_x[:] = next_lam_x
        self._qp_lam_a[:] = next_lam_a
        self._last_u = command.copy()
        self._U[:] = accepted_controls
        self._advance_blocks(
            self._U.reshape(-1), 0, self.horizon, self.nu, advance_stages)
        self._advance_qp_warm_start(advance_stages)
        finished_at = time.perf_counter_ns()
        slack = qp_step[self._slack_offset:].reshape(
            self.horizon, self.nc)
        controls = {name: float(command[index])
                    for index, name in enumerate(self.input_names)}
        saturated = tuple(sorted(
            name for index, name in enumerate(self.input_names)
            if math.isclose(command[index], self.u_lo[index], abs_tol=1e-6)
            or math.isclose(command[index], self.u_hi[index], abs_tol=1e-6)))
        timings = MPCTimings(
            rollout_linearize_ms=(linearized_at-total_start)/1e6,
            assemble_ms=(assembled_at-linearized_at)/1e6,
            solve_ms=(solved_at-assembled_at)/1e6,
            total_ms=(finished_at-total_start)/1e6,
            qp_update_ms=qp_update_ms,
            qp_iterations_ms=qp_iterations_ms,
        )
        final = MPCResult(
            controls=controls, control_vector=command,
            nominal_controls=accepted_controls,
            nominal_states=accepted_states,
            qp_status=status, qp_iterations=iterations,
            qp_cost=qp_cost,
            predicted_peak_bank=peak_bank,
            predicted_bank_violation=bank_violation,
            predicted_attitude_constraint_violation=(
                attitude_constraint_violation),
            bank_slack=float(np.max(slack * slack_scales)),
            saturated=saturated, timings=timings,
            qp_primal_residual=qp_primal_residual,
            qp_dual_residual=qp_dual_residual,
            qp_rho_updates=qp_rho_updates,
            qp_rho_estimate=qp_rho_estimate,
            hessian_regularization=HESSIAN_REGULARIZATION,
            clipped_controls=clipped_controls,
            qp_used_uncondensed_fallback=qp_used_uncondensed_fallback,
            qp_condensed_candidate_valid=qp_condensed_candidate_valid)
        self.last_result = final
        return final


__all__ = [
    "MPC",
    "CraftHorizonReference",
    "MPCReference",
    "MPCResult",
    "MPCTimings",
]
