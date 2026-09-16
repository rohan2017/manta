"""Research-only colored body-force discrepancy, independent of water current.

The extra state is acceleration-equivalent force error (m/s²) in body axes:
db = -b/tau dt + sqrt(2 sigma²/tau) dW. This is an explicit modeling prior,
not a claim that hydrodynamic errors are white or exactly Gauss-Markov.
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
from ._ins_protected_update import protected_update
from ._kalman import _reset_jacobian, joseph_update, symmetrize


class ForceDiscrepancySpec(StateSpec):
    def __init__(self, base):
        self.base = base
        super().__init__([*base.slots, StateSlot("ins_research.force_discrepancy",
            base.ambient_dim, RnManifold(3), base.tangent_dim)])
        self.error_model = getattr(base, "error_model", "product")+"_force_discrepancy"

    def boxplus_sym(self, x, delta):
        a, n = self.base.ambient_dim, self.base.tangent_dim
        return ca.vertcat(self.base.boxplus_sym(x[:a], delta[:n]), x[a:]+delta[n:])

    def boxminus_sym(self, x, reference):
        a = self.base.ambient_dim
        return ca.vertcat(self.base.boxminus_sym(x[:a], reference[:a]), x[a:]-reference[a:])

    def exact_reset(self, x, delta):
        return ca.diagcat(_reset_jacobian(self.base, delta[:self.base.tangent_dim],
                                         x[:self.base.ambient_dim]), ca.MX.eye(3))


def with_force_discrepancy(ir, *, sigma, tau_s, protect_navigation=False):
    module, system, base = ir.module(), ir.sys, ir.spec
    if (ir.covariance != "geometric" or "P_consider" in module.state
            or "local_current.velocity" not in base):
        raise ValueError("force-discrepancy research requires geometric current INS without Schmidt states")
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("finite nonnegative sigma and positive tau required")
    spec = ForceDiscrepancySpec(base)
    a, n = base.ambient_dim, base.tangent_dim
    x, P, Q = ca.MX.sym("x", a+3), ca.MX.sym("P", n+3, n+3), ca.MX.sym("Q", n+3, n+3)
    u, dt, t = system.u_sym, system.dt_sym, system.t_sym
    decay = ca.exp(-dt/tau_s)
    F = ca.substitute(system.F_sym, system.x_sym, x[:a])
    functions = dict(module.functions)
    for name in ("predict", "predict_with_Q"):
        old = module.functions[name]
        values = {"x": x[:a], "P": P[:n, :n], "Q": Q[:n, :n], "u": u, "dt": dt, "t": t}
        mean, covariance = old(*[values[key] for key in old.name_in()])
        cross = F @ P[:n, n:]*decay
        nuisance = decay**2*P[n:, n:]+sigma**2*(1-decay**2)*ca.MX.eye(3)
        next_P = ca.vertcat(ca.horzcat(covariance, cross), ca.horzcat(cross.T, nuisance))
        if name == "predict_with_Q":
            next_P += Q-ca.diagcat(Q[:n, :n], ca.MX.zeros(3, 3))
        inputs = [x, P]+([Q] if name == "predict_with_Q" else [])+[u, dt, t]
        names = ["x", "P"]+(["Q"] if name == "predict_with_Q" else [])+["u", "dt", "t"]
        functions[name] = ca.Function("research_force_"+name, inputs,
            [ca.vertcat(mean, decay*x[a:]), symmetrize(next_P)], names, ["x_new", "P_new"])
    sources = module.metadata.get("measurement_sources", {})
    for sensor in prepared_sensors(system, base, x0=module.state.field("x").init, who="force-discrepancy research"):
        ident = entry_ident(sensor.full)
        is_force = sensor.full in sources
        # ModelForce observes specific force in sensor axes. Keep the nuisance
        # in body axes so attitude excitation can distinguish it from current.
        discrepancy = ca.DM(system.R_craft_from_sensor).T @ x[a:]
        h = ca.substitute(sensor.h, system.x_sym, x[:a])+(discrepancy if is_force else 0)
        d = ca.MX.sym("measurement_error", n+3)
        hd = ca.substitute(h, x, spec.boxplus_sym(x, d))
        H = ca.substitute(ca.jacobian(hd, d), d, ca.MX.zeros(n+3))
        gate = module.metadata["nis_gates"][sensor.full]

        def update(R, h=h, H=H, sensor=sensor, is_force=is_force, gate=gate):
            if protect_navigation and is_force:
                c = base.slot("local_current.velocity").tangent_offset
                mean, cov, nu, S = protected_update(x, P, h, H, R, sensor.z, spec,
                                                    active=[c, c+1, c+2, n, n+1, n+2])
            else:
                mean, cov, nu, S = joseph_update(x, P, h, H, R, sensor.z, spec)
            nis = ca.dot(nu, spd_solve(S, nu))
            accepted = ca.MX.ones(1) if gate is None else nis <= gate
            return [ca.if_else(accepted, mean, x), ca.if_else(accepted, cov, P), nu, S, nis, accepted]

        R = ca.substitute(sensor.R, system.x_sym, x[:a])
        outputs = update(R)
        names = ["x_new", "P_new", "innovation", "innovation_covariance", "nis", "accepted"]
        for prefix, count in (("update_", 2), ("update_diagnostic_", 6)):
            name = prefix+ident
            functions[name] = ca.Function("research_force_"+name, [x, P, sensor.z, u, t],
                outputs[:count], ["x", "P", "z", "u", "t"], names[:count])
        R_override = ca.MX.sym("R", sensor.dim, sensor.dim)
        model_R = ca.substitute(sensor.model_R, system.x_sym, x[:a])
        name = "update_with_R_"+ident
        functions[name] = ca.Function("research_force_"+name, [x, P, sensor.z, R_override, u, t],
            update(R_override+model_R), ["x", "P", "z", "R", "u", "t"], names)
    prior = module.functions["initialize_prior"]
    inputs = [prior.mx_in(i) for i in range(prior.n_in())]
    px, pp = prior(*inputs)
    functions["initialize_prior"] = ca.Function("research_force_initialize_prior", inputs,
        [ca.vertcat(px, ca.MX.zeros(3)), ca.diagcat(pp, sigma**2*ca.MX.eye(3))], prior.name_in(), prior.name_out())
    fields = (StateField("x", "manifold", (a+3,), spec=spec, init=np.r_[module.state.field("x").init, np.zeros(3)]),
              StateField("P", "matrix", (n+3, n+3), init=np.block([
                  [module.state.field("P").init, np.zeros((n, 3))], [np.zeros((3, n)), sigma**2*np.eye(3)]])))
    ports = tuple(replace(port, shape=(n+3, n+3)) if port.name == "Q" else port for port in module.ports)
    result = copy(ir)
    result.spec = spec
    result._module = replace(module, state=StateLayout(fields), ports=ports, functions=functions,
        metadata={**module.metadata, "research_force_discrepancy_sigma_mps2": sigma,
                  "research_force_discrepancy_tau_s": tau_s, "research_force_protect_navigation": protect_navigation})
    return result
