"""Opt-in research augmentation for reused packet-boundary accelerometer noise.

Limited to a colocated IMU and uniform raw sampling. The first raw accelerometer
of a packet was the previous packet's fresh endpoint and may already have been
used for model-force correction. Retain its three standardized noise variables
until that sample has been integrated. No production INS default selects this.
"""

from copy import copy
from dataclasses import replace

import casadi as ca
import numpy as np

from ..ir._linalg import spd_solve
from ..ir.manifold import RnManifold
from ..ir.module import StateField, StateLayout, entry_ident
from ..ir.state_spec import StateSlot, StateSpec
from ._assembly import prepared_sensors
from ._kalman import joseph_update, symmetrize


def condition_start_noise(covariance, F, G, prior):
    """Replace a packet's independent unit start-noise marginal by its joint prior."""
    n = F.size1()
    C = prior[:n, n:]
    return symmetrize(covariance+G@(prior[n:, n:]-ca.MX.eye(G.size2()))@G.T
                      +F@C@G.T+G@C.T@F.T)


class AccelBoundarySpec(StateSpec):
    """Original finite chart times a standardized Euclidean noise coordinate."""

    def __init__(self, base):
        self.base = base
        super().__init__([*base.slots, StateSlot("ins_research.accel_boundary_noise",
                                                base.ambient_dim, RnManifold(3), base.tangent_dim)])
        self.error_model = getattr(base, "error_model", "product")+"_accel_boundary"

    def boxplus_sym(self, x, delta):
        a, n = self.base.ambient_dim, self.base.tangent_dim
        return ca.vertcat(self.base.boxplus_sym(x[:a], delta[:n]), x[a:]+delta[n:])

    def boxminus_sym(self, x, reference):
        a = self.base.ambient_dim
        return ca.vertcat(self.base.boxminus_sym(x[:a], reference[:a]), x[a:]-reference[a:])

    def exact_reset(self, x, delta):
        from ._kalman import _reset_jacobian
        return ca.diagcat(_reset_jacobian(self.base, delta[:self.base.tangent_dim],
                                         x[:self.base.ambient_dim]), ca.MX.eye(3))


def with_accelerometer_boundary(ir, *, sample_dt, accel_noise_sigma, common=True):
    """Retain common endpoint noise, or an exactly equivalent augmented control.

    With ``common=False`` the auxiliary coordinates are unobserved, and the
    original navigation mean/covariance must match the baseline. ``common=True``
    moves sigma_a² I from the force residual covariance into an explicit
    observation of the endpoint-noise state, preserving marginal R. The packet
    prediction then conditions its first-sample noise on the retained estimate
    and cross-covariance. This is an analytic first-order noise treatment.
    """
    module, system, base = ir.module(), ir.sys, ir.spec
    if (ir.propagation != "preintegrated" or "P_consider" in module.state
            or np.linalg.norm(module.metadata["lever_arm_m"]) > 1e-12):
        raise ValueError("accelerometer boundary research needs a colocated preintegrated INS without Schmidt states")
    if sample_dt <= 0 or accel_noise_sigma <= 0:
        raise ValueError("sample interval and accelerometer sigma must be positive")
    if "initialize_prior" not in module.functions:
        raise ValueError("accelerometer boundary research requires a physical-prior initializer")
    spec = AccelBoundarySpec(base)
    a, n = base.ambient_dim, base.tangent_dim
    x, P = ca.MX.sym("x", a+3), ca.MX.sym("P", n+3, n+3)
    u, dt, t = system.u_sym, system.dt_sym, system.t_sym
    Q = ca.MX.sym("Q", n+3, n+3)
    eta = x[a:]

    def adjusted_inputs(noise):
        adjusted = ca.MX(u)
        for field, scale in (("delta_velocity", sample_dt),
                             ("delta_position", sample_dt*(dt-.5*sample_dt))):
            selection = system._input_slices[ir.preintegration_input_map[field]]
            adjusted[selection] -= accel_noise_sigma*scale*noise
        return adjusted

    corrected = adjusted_inputs(eta)

    def substitute(expression, control):
        return ca.substitute([expression], [system.x_sym, system.u_sym], [x[:a], control])[0]

    F = substitute(system.F_sym, corrected)
    d = ca.MX.sym("accel_boundary_delta", 3)
    nominal = substitute(system.x_new, corrected)
    perturbed = substitute(system.x_new, adjusted_inputs(eta+d))
    G = ca.substitute(ca.jacobian(base.boxminus_sym(perturbed, nominal), d), d, ca.MX.zeros(3))
    functions = dict(module.functions)
    count = ca.floor(dt/sample_dt+.5)
    cadence_ok = ca.logic_and(count >= 1, ca.fabs(dt-count*sample_dt) <= 1e-10*dt)
    for name in ("predict", "predict_with_Q"):
        old = module.functions[name]
        values = {"x": x[:a], "P": P[:n, :n], "u": corrected, "dt": dt, "t": t, "Q": Q[:n, :n]}
        mean, covariance = old(*[values[key] for key in old.name_in()])
        # The original packet Q contains G I Gᵀ. Replace only that component
        # by its conditional covariance and cross terms, retaining all other
        # packet/process noise. Uniform colocation gives the exact first-sample
        # delta-v/delta-p sensitivity; G includes the companion Earth transform.
        covariance = condition_start_noise(covariance, F, G, P)
        next_P = ca.diagcat(symmetrize(covariance), ca.MX.eye(3))
        if name == "predict_with_Q":
            next_P += Q-ca.diagcat(Q[:n, :n], ca.MX.zeros(3, 3))
        next_x = ca.vertcat(mean, ca.MX.zeros(3))
        inputs = [x, P]+([Q] if name == "predict_with_Q" else [])+[u, dt, t]
        names = ["x", "P"]+(["Q"] if name == "predict_with_Q" else [])+["u", "dt", "t"]
        functions[name] = ca.Function("research_accel_"+name, inputs,
                                       [ca.if_else(cadence_ok, next_x, ca.MX.nan(a+3)),
                                        ca.if_else(cadence_ok, next_P, ca.MX.nan(n+3, n+3))],
                                       names, ["x_new", "P_new"])
    sources = module.metadata.get("measurement_sources", {})
    for sensor in prepared_sensors(system, base, x0=module.state.field("x").init, who="common-noise INS research"):
        ident = entry_ident(sensor.full)
        shared = common and sensor.full in sources
        h = ca.substitute(sensor.h, system.x_sym, x[:a])
        if shared:
            h += accel_noise_sigma*eta
        delta = ca.MX.sym("measurement_error", n+3)
        hd = ca.substitute(h, x, spec.boxplus_sym(x, delta))
        H = ca.substitute(ca.jacobian(hd, delta), delta, ca.MX.zeros(n+3))
        threshold = module.metadata["nis_gates"][sensor.full]

        def expressions(R, h=h, H=H, shared=shared, sensor=sensor, threshold=threshold):
            residual_R = R-(accel_noise_sigma**2*ca.MX.eye(sensor.dim) if shared else 0)
            mean, covariance, nu, S = joseph_update(x, P, h, H, residual_R, sensor.z, spec)
            nis = ca.dot(nu, spd_solve(S, nu))
            accepted = ca.MX.ones(1) if threshold is None else nis <= threshold
            return [ca.if_else(accepted, mean, x), ca.if_else(accepted, covariance, P), nu, S, nis, accepted]

        R = ca.substitute(sensor.R, system.x_sym, x[:a])
        # Evidence owns the *total* residual white variance. It must be large
        # enough to contain the known accelerometer noise being split out.
        check_R = ca.Function("check_residual_R", [x, u, t], [R])
        R0 = np.asarray(check_R(np.r_[module.state.field("x").init, np.zeros(3)], system.u_defaults, 0))
        if shared and np.linalg.eigvalsh(R0-accel_noise_sigma**2*np.eye(sensor.dim)).min() <= 0:
            raise ValueError("fit residual covariance does not contain positive independent model error")
        outputs = expressions(R)
        output_names = ["x_new", "P_new", "innovation", "innovation_covariance", "nis", "accepted"]
        for prefix, size in (("update_", 2), ("update_diagnostic_", 6)):
            name = prefix+ident
            functions[name] = ca.Function("research_accel_"+name, [x, P, sensor.z, u, t], outputs[:size],
                                           ["x", "P", "z", "u", "t"], output_names[:size])
        override = ca.MX.sym("R", sensor.dim, sensor.dim)
        model_R = ca.substitute(sensor.model_R, system.x_sym, x[:a])
        name = "update_with_R_"+ident
        functions[name] = ca.Function("research_accel_"+name, [x, P, sensor.z, override, u, t],
                                       expressions(override+model_R), ["x", "P", "z", "R", "u", "t"], output_names)
    initializer = module.functions["initialize_prior"]
    prior_inputs = [initializer.mx_in(i) for i in range(initializer.n_in())]
    prior_x, prior_P = initializer(*prior_inputs)
    functions["initialize_prior"] = ca.Function("research_accel_initialize_prior", prior_inputs,
                                                 [ca.vertcat(prior_x, ca.MX.zeros(3)), ca.diagcat(prior_P, ca.MX.eye(3))],
                                                 initializer.name_in(), initializer.name_out())
    fields = (StateField("x", "manifold", (a+3,), spec=spec,
                          init=np.r_[module.state.field("x").init, np.zeros(3)]),
              StateField("P", "matrix", (n+3, n+3),
                          init=np.block([[module.state.field("P").init, np.zeros((n, 3))],
                                         [np.zeros((3, n)), np.eye(3)]])))
    ports = tuple(replace(port, shape=(n+3, n+3)) if port.name == "Q" else port for port in module.ports)
    result = copy(ir)
    result.spec = spec
    result._module = replace(module, state=StateLayout(fields), ports=ports, functions=functions,
                             metadata={**module.metadata, "research_accel_boundary": bool(common),
                                       "research_accel_sample_dt": sample_dt,
                                       "research_accel_sigma": accel_noise_sigma,
                                       "research_scope": "uniform colocated raw samples; analytic first-order shared-noise augmentation"})
    return result
