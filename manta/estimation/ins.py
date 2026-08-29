"""Error-state strapdown INS with model-force disturbance observations.

Unlike :class:`EKF` and :class:`UKF`, ``INS`` does not propagate navigation
state through the vehicle dynamics. A selected physical :class:`~manta.parts.IMU`
supplies gyro and accelerometer samples to the strapdown recurrence; every
other selected output remains an ordinary Kalman measurement. In particular,
a colocated :class:`~manta.parts.ModelForce` predicts specific force from the
compiled wrench model and observes the accelerometer sample. That residual
directly observes accelerometer bias and any estimable disturbance state that
the force model reads.

There is deliberately no angular-velocity state and no torque-model residual.
Angular constraints such as NHC or coordinated-turn observations belong in
separate pseudo-parts.
"""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Any

import casadi as ca
import numpy as np

from ..fields import GravityField
from ..ir._linalg import spd_solve
from ..ir._names import resolve_suffix
from ..ir._rotation import (
    quat_conj,
    quat_from_rotmat_np,
    quat_mul,
    quat_to_rotmat,
    so3_exp,
)
from ..ir.frames import WorldFrame
from ..ir.module import PortField, entry_ident
from ..ir.state_spec import StateSpec, flatten_nested, resolve_slotset
from ..ir.types import Vec3
from ..linearization import LinearizedSystem
from ..linearization.engine import SensorModel, TickLinearizer
from ..linearization.partition import dependency_closure, partition_blocks
from ..linearization.system import freeze_complement, slot_of_tangent_index
from ..parts import IMU, ModelForce
from ..parts._mounting import rest_pose_from
from ..parts.articulation.joint import ArticulatedJoint
from ._assembly import (
    _FilterBase,
    _q_auto,
    emit_filter_module,
    initial_ambient,
    prepared_sensors,
    resolve_gates,
)
from ._kalman import joseph_update, symmetrize

_RIGID = ("position", "orientation", "velocity", "angular_velocity")

# A preintegrated packet spans `duration` seconds; the INS predict that
# consumes it must advance by exactly that interval, because the gravity and
# lever-arm terms are scaled by the predict `dt` while the packet deltas were
# integrated over `duration`. The kernel compares the two at this relative
# tolerance (float accumulation of sample intervals) and poisons the
# navigation state with NaN on a mismatch, so every backend fails instead of
# silently mis-scaling.
PREINTEGRATION_DURATION_RTOL = 1e-9

# The selected accelerometer sample drives both strapdown propagation and the
# `ModelForce.specific_force` pseudo-measurement, so their noises are
# correlated. INS drops that cross term, which is sound only while
# `rho = accel_noise_sigma / model_error_sigma` is small: the neglected
# contribution to the innovation covariance and gain scales like rho².
# Above the ceiling (25 % of R) the dropped term is no longer a second-order
# correction and construction is refused; above the warning level (1 % of R)
# the value is surfaced as a warning and in the artifact metadata. Neither
# is a silent drop. `model_error_sigma` here is the white per-axis floor of
# the `ModelForce` error model (its quietest axis); the Gauss–Markov
# component is filter state, not R. Both come from the part's fit evidence,
# which the INS requires (`ModelForce(evidence=...)`): a model-aided INS
# never runs on an implicit zero held-out bias or a white-only error model.
MODEL_FORCE_RHO_CEILING = 0.5
MODEL_FORCE_RHO_WARNING = 0.1
PREINTEGRATION_DURATION_DOC = (
    "Packet span in seconds. predict(u, dt) requires dt == duration "
    f"(relative tolerance {PREINTEGRATION_DURATION_RTOL:g}); a mismatch "
    "yields a NaN navigation state instead of mis-scaled gravity/lever terms."
)


def _qualified_parts(world, cls) -> dict[str, Any]:
    return {f"{craft.name}.{part.name}": part
            for craft in world.crafts for part in craft.parts
            if isinstance(part, cls)}


def _resolve_part(source_world, resolved_world, value, cls, *, who: str):
    resolved = _qualified_parts(resolved_world, cls)
    if isinstance(value, str):
        full = resolve_suffix(value, list(resolved), label=cls.__name__, who=who)
        return full, resolved[full]
    if not isinstance(value, cls):
        raise TypeError(f"{who}: expected a {cls.__name__} part or name")
    source = _qualified_parts(source_world, cls)
    matches = [name for name, part in source.items() if part is value]
    if len(matches) != 1:
        raise ValueError(
            f"{who}: supplied {cls.__name__} is not uniquely attached to world")
    return matches[0], resolved[matches[0]]


def _rigid_mount(craft, part, *, who: str) -> tuple[np.ndarray, np.ndarray]:
    cur = part.parent
    while cur is not None and cur is not craft.root:
        if isinstance(cur, ArticulatedJoint):
            raise TypeError(
                f"{who}: {part.name!r} is below articulated joint "
                f"{cur.name!r}; strapdown requires a rigid IMU mount")
        cur = cur.parent
    if cur is not craft.root:
        raise ValueError(f"{who}: {part.name!r} is not mounted on {craft.name!r}")
    return rest_pose_from(craft.root, part)


class _INSSystem:
    """Linearized-system-compatible IR over a strapdown recurrence."""

    def __init__(self, source, *, imu, track, sensors, inputs,
                 propagation: str) -> None:
        # Reuse the one authoritative model snapshot/compiler. The resulting
        # dynamics transition is only a source of ordinary measurement models
        # and non-navigation state recurrences; navigation is replaced below.
        base = LinearizedSystem(source, sensors=[], inputs=inputs)
        self.authoring_world = source
        self.world = base.world
        self.model = base.model
        self.crafts = base.crafts
        self.full_spec = base.full_spec
        self.tick = base.tick
        self.sample_rates = dict(base.sample_rates)
        self.noise_specs = list(base.noise_specs)
        self.Sigma = base.Sigma
        self._cf = base._cf
        self._sig = base._sig
        self.model_input_names = list(base.input_names)
        self.model_input_defaults = dict(base.input_defaults)

        source_world = (source.world_copy() if hasattr(source, "world_copy")
                        else source)
        self.imu_name, self.imu = _resolve_part(
            source_world, self.world, imu, IMU, who="INS")
        self.craft_name = self.imu_name.split(".", 1)[0]
        self.craft = next(c for c in self.crafts if c.name == self.craft_name)
        self.propagation = propagation
        self.lever_arm, self.R_craft_from_sensor = _rigid_mount(
            self.craft, self.imu, who="INS")

        self.accel_input = f"{self.imu_name}.accel"
        self.gyro_input = f"{self.imu_name}.gyro"
        self.preintegration_input_map: dict[str, str] = {}
        preintegration_fields: list[PortField] = []
        if propagation == "preintegrated":
            prefix = f"{self.imu_name}.preintegrated"
            packet_dims = {
                "delta_orientation": 4,
                "delta_velocity": 3,
                "delta_position": 3,
                "covariance": 81,
                "bias_jacobian": 54,
                "gyro_bias_reference": 3,
                "accel_bias_reference": 3,
                "start_gyro": 3,
                "duration": 1,
            }
            packet_defaults = {
                "delta_orientation": (1.0, 0.0, 0.0, 0.0),
            }
            packet_docs = {"duration": PREINTEGRATION_DURATION_DOC}
            for short, dim in packet_dims.items():
                full = f"{prefix}.{short}"
                self.preintegration_input_map[short] = full
                preintegration_fields.append(PortField(
                    full, dim, packet_defaults.get(short, np.zeros(dim)),
                    doc=packet_docs.get(short, "")))

        self.input_names = [*self.model_input_names,
                            self.accel_input, self.gyro_input,
                            *self.preintegration_input_map.values()]
        self.input_defaults = {
            **{name: base.input_defaults[name]
               for name in self.model_input_names},
            self.accel_input: np.zeros(3),
            self.gyro_input: np.zeros(3),
            **{field.name: np.atleast_1d(field.default).copy()
               for field in preintegration_fields},
        }
        self.input_fields = tuple([
            PortField(name, 1, float(base.input_defaults[name]),
                      rate=self.sample_rates.get(name))
            for name in self.model_input_names
        ] + [
            PortField(self.accel_input, 3, (0.0, 0.0, 0.0),
                      rate=self.sample_rates.get(self.accel_input)),
            PortField(self.gyro_input, 3, (0.0, 0.0, 0.0),
                      rate=self.sample_rates.get(self.gyro_input)),
        ] + preintegration_fields)
        self._input_slices: dict[str, slice] = {}
        off = 0
        for field in self.input_fields:
            self._input_slices[field.name] = slice(off, off + field.dim)
            off += field.dim
        self.u_defaults = np.concatenate([
            np.atleast_1d(np.asarray(field.default, dtype=float)).reshape(-1)
            for field in self.input_fields])

        all_outputs = list(self._sig.sensor_names)
        selected_imu_outputs = {self.accel_input, self.gyro_input}
        if sensors is None:
            chosen_sensors = [name for name in all_outputs
                              if name not in selected_imu_outputs]
        else:
            chosen = {resolve_suffix(name, all_outputs, label="sensor", who="INS")
                      for name in sensors}
            overlap = chosen & selected_imu_outputs
            if overlap:
                raise ValueError(
                    "INS: the selected IMU drives prediction and cannot also "
                    f"be an ordinary update sensor: {sorted(overlap)}")
            chosen_sensors = [name for name in all_outputs if name in chosen]
        self._chosen_sensors = chosen_sensors

        mandatory = {
            f"{self.craft_name}.position",
            f"{self.craft_name}.orientation",
            f"{self.craft_name}.velocity",
        }
        for bias in (f"{self.imu_name}.gyro_bias",
                     f"{self.imu_name}.accel_bias"):
            if bias in self.full_spec:
                mandatory.add(bias)
        if track is not None:
            unknown = set(track) - {self.craft_name}
            if unknown:
                raise KeyError(
                    f"INS: track may only name selected craft "
                    f"{self.craft_name!r}; got {sorted(unknown)}")
            requested = resolve_slotset(self.craft_name,
                                        track.get(self.craft_name, 0))
            omega = f"{self.craft_name}.angular_velocity"
            if omega in requested:
                raise ValueError(
                    "INS has no angular_velocity state; use algebraic angular "
                    "constraints or the dynamics-driven EKF/UKF")
            mandatory |= requested

        # Candidate contains the selected craft's non-omega state and every
        # disturbance state. A structural pass then removes states neither
        # observed nor required by the strapdown recurrence.
        candidate_names = {
            slot.name for slot in self.full_spec.slots
            if (slot.name.startswith(f"{self.craft_name}.")
                and slot.name != f"{self.craft_name}.angular_velocity")
            or not any(slot.name.startswith(f"{c.name}.") for c in self.crafts)
        }
        candidate = StateSpec.subset(self.full_spec, candidate_names)
        first = self._differentiate(candidate)
        seed = set(mandatory)
        sot = slot_of_tangent_index(candidate)
        for sm in first["sensors"].values():
            seed |= {sot[int(col)] for col in sm.observed_cols}
        kept = dependency_closure(first["F_pattern"], sot, seed)
        self.spec = StateSpec.subset(candidate, kept)
        final = first if kept == candidate_names else self._differentiate(self.spec)
        self._install(final)

        self.measurement_sources: dict[str, str] = {}
        self.rho_by_sensor: dict[str, float] = {}
        self.evidence_by_sensor: dict[str, Any] = {}
        for full in self.sensors:
            owner_name, output_name = full.rsplit(".", 1)
            part = _qualified_parts(self.world, ModelForce).get(owner_name)
            if part is None or output_name != "specific_force":
                continue
            r, R = _rigid_mount(self.craft, part, who="INS")
            if not (np.allclose(r, self.lever_arm, atol=1e-9)
                    and np.allclose(R, self.R_craft_from_sensor, atol=1e-9)):
                raise ValueError(
                    f"INS: ModelForce {owner_name!r} must be colocated and "
                    f"co-oriented with selected IMU {self.imu_name!r}")
            self.measurement_sources[full] = self.accel_input
            evidence = part.evidence
            if evidence is None:
                raise ValueError(
                    f"INS: ModelForce {owner_name!r} carries no fit evidence. "
                    "A model-aided INS consumes the held-out residual bias "
                    "and the time-correlated model-error covariance "
                    "explicitly; it refuses the implicit zero. Construct the "
                    "part with evidence=FitEvidence (FitResult.evidence(...) "
                    "/ NoiseFitResult.evidence(...) / held_out_evidence(...))"
                    f" on the {self.imu_name!r} accelerometer channel")
            if evidence.binding is None:
                raise ValueError(
                    f"INS: ModelForce {owner_name!r} carries unbound fit "
                    "evidence. Model-aided deployment requires evidence "
                    "issued by held_out_evidence(...) or a fit result so "
                    "the fitted/source artifact, profile, datasets, and "
                    "channel contract are exact")
            if not evidence.accepted:
                failed = ", ".join(
                    f"{c.criterion}[{c.axis}]={c.value:.4g} (limit {c.limit:.4g})"
                    for c in evidence.failed_checks)
                raise ValueError(
                    f"INS: ModelForce {owner_name!r} fit evidence on "
                    f"{evidence.channel!r} is not accepted; failed: {failed}. "
                    "Refit or revise the acceptance criteria — the decision "
                    "is recorded in the artifact, not overridden here")
            self.evidence_by_sensor[full] = evidence
            # The dropped cross term involves the white accelerometer noise
            # against the white part of R only; the correlated component is
            # filter state. Use the quietest axis (largest rho).
            sigma_model = min(part.white_sigmas)
            rho = (float(self.imu.accel_noise_sigma) / sigma_model
                   if sigma_model > 0.0 else float("inf"))
            if not rho <= MODEL_FORCE_RHO_CEILING:
                raise ValueError(
                    f"INS: ModelForce {owner_name!r} noise ratio rho = "
                    f"accel_noise_sigma/min(white model_error sigma) = "
                    f"{rho:.4g} exceeds "
                    f"MODEL_FORCE_RHO_CEILING={MODEL_FORCE_RHO_CEILING}; the "
                    "dropped process/measurement noise correlation is no "
                    "longer a second-order term. The white model-error floor "
                    "comes from the fit evidence — select a quieter "
                    "accelerometer or refit")
            if rho > MODEL_FORCE_RHO_WARNING:
                warnings.warn(
                    f"INS: ModelForce {owner_name!r} noise ratio rho={rho:.4g} "
                    f"exceeds MODEL_FORCE_RHO_WARNING={MODEL_FORCE_RHO_WARNING}"
                    " (the dropped noise correlation is ~rho² of R); the "
                    "value is recorded in the artifact metadata",
                    RuntimeWarning, stacklevel=4)
            self.rho_by_sensor[full] = rho

    def _gravity(self, position, t):
        field = next((f for f in self.world.fields
                      if isinstance(f, GravityField)), None)
        if field is None:
            return ca.MX.zeros(3, 1)
        return field.value_at_sym(
            Vec3[WorldFrame].from_mx(position), t).mx

    def _differentiate(self, spec: StateSpec) -> dict[str, Any]:
        n_tan = spec.tangent_dim
        n_u = sum(field.dim for field in self.input_fields)
        n_noise = sum(ns.dim for ns in self.noise_specs)
        x = ca.MX.sym("x", spec.ambient_dim, 1)
        u = ca.MX.sym("u", n_u, 1)
        dt = ca.MX.sym("dt", 1, 1)
        t = ca.MX.sym("t", 1, 1)
        noise = ca.MX.sym("noise", n_noise, 1) if n_noise else ca.MX.zeros(0, 1)
        zero_noise = ca.MX.zeros(n_noise, 1)
        model_u = u[:len(self.model_input_names)]
        accel_sample = u[self._input_slices[self.accel_input]]
        gyro_sample = u[self._input_slices[self.gyro_input]]

        noise_slices: dict[str, slice] = {}
        off = 0
        for ns in self.noise_specs:
            noise_slices[ns.full] = slice(off, off + ns.dim)
            off += ns.dim

        engine = TickLinearizer(self._cf, self.model_input_names,
                                self.noise_specs, "exact")
        ref = flatten_nested(self.world._initial_state_dict())
        frozen_base = freeze_complement(self.full_spec,
                                        {s.name for s in spec.slots}, ref)
        omega_name = f"{self.craft_name}.angular_velocity"
        p_name = f"{self.craft_name}.position"
        q_name = f"{self.craft_name}.orientation"
        v_name = f"{self.craft_name}.velocity"
        gyro_bias_name = f"{self.imu_name}.gyro_bias"
        accel_bias_name = f"{self.imu_name}.accel_bias"
        R_bs = ca.DM(self.R_craft_from_sensor)
        lever = ca.DM(self.lever_arm)
        q_bs = ca.DM(quat_from_rotmat_np(self.R_craft_from_sensor))
        q_sb = quat_conj(q_bs)

        def packet_chunk(name, dim):
            full = self.preintegration_input_map[name]
            return ca.reshape(u[self._input_slices[full]], dim, 1)

        packet_covariance = (ca.reshape(packet_chunk("covariance", 81), 9, 9)
                             if self.propagation == "preintegrated" else None)

        def state_chunk(xv, name, dim):
            if name not in spec:
                return ca.MX.zeros(dim, 1)
            slot = spec.slot(name)
            return xv[slot.ambient_offset:slot.ambient_offset + slot.ambient_dim]

        def noise_chunk(nv, name):
            sl = noise_slices.get(name)
            return ca.MX.zeros(3, 1) if sl is None else nv[sl]

        def evaluate(xv, nv, packet_error=None):
            gyro_bias = state_chunk(xv, gyro_bias_name, 3)
            accel_bias = state_chunk(xv, accel_bias_name, 3)
            gyro_noise = noise_chunk(nv, f"{self.imu_name}.gyro_noise")
            accel_noise = noise_chunk(nv, f"{self.imu_name}.accel_noise")
            if self.propagation == "raw":
                gyro_corrected = gyro_sample - gyro_bias - gyro_noise
                accel_corrected = accel_sample - accel_bias - accel_noise
            else:
                # The packet covariance already represents the raw IMU white
                # noise. Endpoint readings remain available to ordinary
                # measurement models but must not inject that noise twice.
                gyro_corrected = gyro_sample - gyro_bias
                accel_corrected = accel_sample - accel_bias
            omega_body = R_bs @ gyro_corrected

            frozen = dict(frozen_base)
            frozen[omega_name] = omega_body
            outs = engine._tick_outputs(
                spec, xv, frozen, model_u, dt, t, nv)

            p = state_chunk(xv, p_name, 3)
            q = state_chunk(xv, q_name, 4)
            v = state_chunk(xv, v_name, 3)
            R_wb = quat_to_rotmat(q)

            gravity_origin = self._gravity(p, t)
            if self.propagation == "raw":
                # The IMU is not generally at the craft origin. Centripetal
                # acceleration is driven directly by the measured gyro. The
                # tangential term uses the compiled model's instantaneous
                # alpha only as a kinematic lever correction; it is not an
                # angular residual and creates no omega state.
                omega_model_next = outs[omega_name]
                alpha_body = ca.substitute(ca.jacobian(omega_model_next, dt),
                                           dt, ca.MX.zeros(1, 1))
                lever_accel = (ca.cross(alpha_body, lever)
                               + ca.cross(omega_body,
                                          ca.cross(omega_body, lever)))
                sensor_position = p + R_wb @ lever
                gravity_sensor = self._gravity(sensor_position, t)
                gravity_delta_body = R_wb.T @ (
                    gravity_sensor - gravity_origin)
                force_origin_body = (R_bs @ accel_corrected
                                     + gravity_delta_body - lever_accel)
                accel_world = R_wb @ force_origin_body + gravity_origin
                q_next = quat_mul(q, so3_exp(omega_body * dt))
                q_next = q_next / ca.sqrt(
                    ca.dot(q_next, q_next) + 1e-30)
                replacements = {
                    p_name: p + v * dt + 0.5 * accel_world * dt * dt,
                    q_name: q_next,
                    v_name: v + accel_world * dt,
                }
            else:
                delta_q = packet_chunk("delta_orientation", 4)
                delta_v = packet_chunk("delta_velocity", 3)
                delta_p = packet_chunk("delta_position", 3)
                J_bias = ca.reshape(packet_chunk("bias_jacobian", 54), 9, 6)
                bias_delta = ca.vertcat(
                    gyro_bias - packet_chunk("gyro_bias_reference", 3),
                    accel_bias - packet_chunk("accel_bias_reference", 3),
                )
                correction = J_bias @ bias_delta
                delta_q = quat_mul(delta_q, so3_exp(correction[0:3]))
                delta_v = delta_v + correction[3:6]
                delta_p = delta_p + correction[6:9]
                if packet_error is not None:
                    delta_q = quat_mul(
                        delta_q, so3_exp(packet_error[0:3]))
                    delta_v = delta_v + packet_error[3:6]
                    delta_p = delta_p + packet_error[6:9]
                delta_q = delta_q / ca.sqrt(
                    ca.dot(delta_q, delta_q) + 1e-30)

                # Propagate the sensor origin, then transform both endpoints
                # back to the craft origin. This makes the lever arm exact at
                # packet boundaries without differentiating the gyro or using
                # the vehicle torque model.
                omega_start = R_bs @ (
                    packet_chunk("start_gyro", 3) - gyro_bias)
                q_ws = quat_mul(q, q_bs)
                R_ws = quat_to_rotmat(q_ws)
                sensor_p = p + R_wb @ lever
                sensor_v = v + R_wb @ ca.cross(omega_start, lever)
                sensor_p_next = (sensor_p + sensor_v * dt
                                 + 0.5 * gravity_origin * dt * dt
                                 + R_ws @ delta_p)
                sensor_v_next = (sensor_v + gravity_origin * dt
                                 + R_ws @ delta_v)
                q_next = quat_mul(quat_mul(q_ws, delta_q), q_sb)
                q_next = q_next / ca.sqrt(
                    ca.dot(q_next, q_next) + 1e-30)
                R_wb_next = quat_to_rotmat(q_next)
                # Residual consistency check: the packet was integrated over
                # `duration`, the gravity/lever terms above over `dt`. They
                # must agree; otherwise poison the navigation state so no
                # backend can continue on a mis-scaled prediction.
                duration = packet_chunk("duration", 1)
                consistent = (ca.fabs(dt - duration)
                              <= PREINTEGRATION_DURATION_RTOL
                              * ca.fmax(ca.fabs(dt), ca.fabs(duration)))
                poison = ca.if_else(consistent, ca.MX.zeros(1, 1),
                                    ca.MX.nan(1, 1))
                replacements = {
                    p_name: sensor_p_next - R_wb_next @ lever + poison,
                    q_name: q_next + poison,
                    v_name: (sensor_v_next
                             - R_wb_next @ ca.cross(omega_body, lever)
                             + poison),
                }
            chunks = []
            for slot in spec.slots:
                value = replacements.get(slot.name, outs[slot.name])
                chunks.append(ca.reshape(value, slot.ambient_dim, 1))
            return ca.vertcat(*chunks), outs

        x_new_noisy, outs_noisy = evaluate(x, noise)
        x_new = ca.substitute(x_new_noisy, noise, zero_noise)

        packet_Q = ca.MX.zeros(n_tan, n_tan)
        if self.propagation == "preintegrated":
            packet_error = ca.MX.sym("packet_error", 9, 1)
            x_packet, _ = evaluate(x, zero_noise, packet_error)
            packet_state_error = spec.boxminus_sym(x_packet, x_new)
            G_packet = ca.substitute(
                ca.jacobian(packet_state_error, packet_error), packet_error,
                ca.MX.zeros(9, 1))
            packet_Q = G_packet @ packet_covariance @ G_packet.T

        delta = ca.MX.sym("delta", n_tan, 1)
        x_pert = spec.boxplus_sym(x, delta)
        x_pert_new, outs_pert = evaluate(x_pert, zero_noise)
        delta_out = spec.boxminus_sym(x_pert_new, x_new)
        F = ca.substitute(ca.jacobian(delta_out, delta), delta,
                          ca.MX.zeros(n_tan, 1))
        F_pattern = np.array(ca.DM(F.sparsity()))

        L = L_pattern = Sigma = None
        if n_noise:
            noise_error = spec.boxminus_sym(x_new_noisy, x_new)
            L = ca.substitute(ca.jacobian(noise_error, noise),
                              noise, zero_noise)
            L_pattern = np.array(ca.DM(L.sparsity()))
            variances = []
            for ns in self.noise_specs:
                variances.extend([float(ns.sigma) ** 2] * ns.dim)
            Sigma = np.diag(variances)

        sensors: dict[str, SensorModel] = {}
        h_supports = []
        for full in self._chosen_sensors:
            dim = int(outs_noisy[full].numel())
            h_noisy = ca.reshape(outs_noisy[full], dim, 1)
            h = ca.substitute(h_noisy, noise, zero_noise)
            h_pert = ca.reshape(outs_pert[full], dim, 1)
            H = ca.substitute(ca.jacobian(h_pert, delta), delta,
                              ca.MX.zeros(n_tan, 1))
            cols = np.flatnonzero(np.array(ca.DM(H.sparsity())).any(axis=0))
            h_supports.append(cols)
            L_h = (ca.substitute(ca.jacobian(h_noisy, noise),
                                 noise, zero_noise) if n_noise else None)
            H_fn = ca.Function(
                f"H_{entry_ident(full)}", [x, u, dt, t], [H],
                ["x", "u", "dt", "t"], ["H"])
            sensors[full] = SensorModel(
                full, dim, h, h_noisy, H, L_h, cols, H_fn)

        predict_fn = ca.Function("predict", [x, u, dt, t], [x_new],
                                 ["x", "u", "dt", "t"], ["x_new"])
        F_fn = ca.Function("F", [x, u, dt, t], [F],
                           ["x", "u", "dt", "t"], ["F"])
        L_fn = (ca.Function("L", [x, u, dt, t], [L],
                            ["x", "u", "dt", "t"], ["L"])
                if L is not None else None)
        blocks = partition_blocks(n_tan, F_pattern, L_pattern, h_supports)
        return {
            "x": x, "u": u, "dt": dt, "t": t, "noise": noise,
            "x_new": x_new, "x_new_noisy": x_new_noisy,
            "F_sym": F, "F_pattern": F_pattern,
            "packet_Q_sym": packet_Q,
            "L_sym": L, "L_pattern": L_pattern, "Sigma": Sigma,
            "sensors": sensors, "predict_fn": predict_fn,
            "F_fn": F_fn, "L_fn": L_fn, "blocks": blocks,
        }

    def _install(self, result) -> None:
        self.x_sym = result["x"]
        self.u_sym = result["u"]
        self.dt_sym = result["dt"]
        self.t_sym = result["t"]
        self.n_sym = result["noise"]
        self.x_new = result["x_new"]
        self.x_new_noisy = result["x_new_noisy"]
        self.F_sym = result["F_sym"]
        self.packet_Q_sym = result["packet_Q_sym"]
        self.L_sym = result["L_sym"]
        self.Sigma = result["Sigma"]
        self.sensors = result["sensors"]
        self.predict_fn = result["predict_fn"]
        self.F_fn = result["F_fn"]
        self.L_fn = result["L_fn"]
        self.B_sym = self.B_fn = None
        self.blocks = result["blocks"]

    def resolve_u(self, values, *, who: str = "INS") -> np.ndarray:
        out = self.u_defaults.copy()
        names = list(self.input_names)
        for key, value in (values or {}).items():
            full = resolve_suffix(key, names, label="input", who=who)
            sl = self._input_slices[full]
            arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
            if arr.size != sl.stop - sl.start or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{who}: input {full!r} expects {sl.stop - sl.start} "
                    "finite value(s)")
            out[sl] = arr
        return out


class INS(_FilterBase):
    """Error-state strapdown inertial navigation transform.

    Args mirror ``EKF`` where their meanings agree. ``imu`` identifies the
    physical IMU whose accelerometer and gyro become required prediction
    inputs in ``u``. Its own outputs are removed from the ordinary update set.
    A selected, colocated ``ModelForce.specific_force`` is automatically
    sourced from that IMU's accelerometer by analysis tools. Such a part
    must carry accepted fit evidence (``ModelForce(evidence=...)``):
    construction refuses one without evidence, or whose evidence failed
    its acceptance criteria, naming what is missing.

    ``propagation="raw"`` consumes one accelerometer/gyro pair per predict.
    ``propagation="preintegrated"`` consumes packets emitted by
    :class:`~manta.estimation.imu_preintegrator.IMUPreintegrator`; the
    high-rate recurrence and the lower-rate INS can both be lowered to
    generated C/C++.
    """

    def __init__(self, world, *, imu,
                 track: dict | None = None,
                 sensors: list[str] | None = None,
                 inputs: list[str] | None = None,
                 discretization: str = "exact",
                 gates: float | dict[str, float] | None = None,
                 propagation: str = "raw") -> None:
        if discretization != "exact":
            raise ValueError(
                "INS derives its exact strapdown F by autodiff; "
                "discretization must be 'exact'")
        if propagation not in {"raw", "preintegrated"}:
            raise ValueError(
                "INS propagation must be 'raw' or 'preintegrated', got "
                f"{propagation!r}")
        sys = _INSSystem(world, imu=imu, track=track,
                         sensors=sensors, inputs=inputs,
                         propagation=propagation)
        self._bind_system(world, sys)
        self.imu = sys.imu_name
        self.propagation = propagation
        self.preintegration_input_map = MappingProxyType({
            **dict(sys.preintegration_input_map),
            **({"end_accel": sys.accel_input, "end_gyro": sys.gyro_input}
               if propagation == "preintegrated" else {}),
        })
        self.measurement_sources = MappingProxyType(dict(sys.measurement_sources))
        self.rho_by_sensor = MappingProxyType(dict(sys.rho_by_sensor))
        self.evidence_by_sensor = MappingProxyType(dict(sys.evidence_by_sensor))
        resolved_gates = resolve_gates(sys, gates, who="INS")

        spec, n_tan = sys.spec, sys.spec.tangent_dim
        x, u, dt, t = sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym
        P = ca.MX.sym("P", n_tan, n_tan)
        Q = ca.MX.sym("Q", n_tan, n_tan)
        F = sys.F_sym
        # Packet uncertainty is intrinsic to a preintegrated measurement and
        # is therefore retained even when the caller overrides the model's Q.
        Q_packet = sys.packet_Q_sym
        Q_auto = _q_auto(sys) + Q_packet
        predict_fn = ca.Function(
            "ins_predict", [x, P, u, dt, t],
            [sys.x_new, symmetrize(F @ P @ F.T + Q_auto)],
            ["x", "P", "u", "dt", "t"], ["x_new", "P_new"])
        predict_q_fn = ca.Function(
            "ins_predict_with_Q", [x, P, Q, u, dt, t],
            [sys.x_new, symmetrize(F @ P @ F.T + Q + Q_packet)],
            ["x", "P", "Q", "u", "dt", "t"], ["x_new", "P_new"])

        x0 = initial_ambient(sys.world, spec)
        updates = {}
        diagnostic_updates = {}
        override_updates = {}
        for ps in prepared_sensors(sys, spec, x0=x0, who="INS"):
            H = ca.substitute(sys.sensors[ps.full].H_sym, dt, ca.MX.zeros(1, 1))
            threshold = resolved_gates[ps.full]

            def expressions(R, ps=ps, H=H, threshold=threshold):
                candidate_x, candidate_P, nu, S = joseph_update(
                    x, P, ps.h, H, R, ps.z, spec)
                nis = ca.dot(nu, spd_solve(S, nu))
                accepted = (ca.MX.ones(1, 1) if threshold is None
                            else nis <= threshold)
                return (ca.if_else(accepted, candidate_x, x),
                        ca.if_else(accepted, candidate_P, P),
                        nu, S, nis, accepted)

            values = expressions(ps.R)
            ident = entry_ident(ps.full)
            updates[ps.full] = ca.Function(
                f"ins_update_{ident}", [x, P, ps.z, u, t], values[:2],
                ["x", "P", "z", "u", "t"], ["x_new", "P_new"])
            diagnostic_updates[ps.full] = ca.Function(
                f"ins_update_diagnostic_{ident}", [x, P, ps.z, u, t], values,
                ["x", "P", "z", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])
            R_override = ca.MX.sym(f"R_{ident}", ps.dim, ps.dim)
            overridden = expressions(R_override)
            override_updates[ps.full] = ca.Function(
                f"ins_update_with_R_{ident}",
                [x, P, ps.z, R_override, u, t], overridden,
                ["x", "P", "z", "R", "u", "t"],
                ["x_new", "P_new", "innovation", "innovation_covariance",
                 "nis", "accepted"])

        metadata = {
            "estimator": "ins",
            "propagation": propagation,
            "prediction_inputs": (
                (sys.accel_input, sys.gyro_input)
                if propagation == "raw"
                else (*sys.preintegration_input_map.values(),
                      sys.accel_input, sys.gyro_input)),
            "preintegration_input_map": self.preintegration_input_map,
            "preintegration_duration_rtol": (
                PREINTEGRATION_DURATION_RTOL
                if propagation == "preintegrated" else None),
            "measurement_sources": MappingProxyType(dict(sys.measurement_sources)),
            "rho_by_sensor": MappingProxyType(dict(sys.rho_by_sensor)),
            # The consumed fit evidence travels with the estimator artifact:
            # which held-out set, which bias, which tau/sigma, and the
            # acceptance checks that admitted it.
            "model_force_evidence": MappingProxyType(
                dict(sys.evidence_by_sensor)),
            "rho_ceiling": MODEL_FORCE_RHO_CEILING,
            "rho_warning": MODEL_FORCE_RHO_WARNING,
            "rho_warned_sensors": tuple(sorted(
                name for name, rho in sys.rho_by_sensor.items()
                if rho > MODEL_FORCE_RHO_WARNING)),
            "lever_arm_m": tuple(float(v) for v in sys.lever_arm),
            # The filter deliberately carries no angular-velocity state.
            # Runtime adapters can still publish the current body rate from
            # the selected gyro by applying this fixed rigid-mount rotation
            # and subtracting the estimated bias.
            "rotation_body_from_imu": tuple(
                float(v) for v in sys.R_craft_from_sensor.reshape(-1)
            ),
        }
        self._module = emit_filter_module(
            sys, spec, name=f"{sys.world.name}_ins", x0=x0,
            predict_fn=predict_fn, predict_q_fn=predict_q_fn,
            updates=updates, diagnostic_updates=diagnostic_updates,
            override_updates=override_updates, gates=resolved_gates,
            metadata_extra=metadata)
