"""Research-only conditional nonlinear packet prediction.

Retain attitude, gyro bias, the reused gyro endpoint and residual delta
rotation as nonlinear coordinates. Integrate the remaining conditional
Gaussian analytically, including every cross-covariance. This requires a
colocated IMU, a finite INS chart, constant local gravity, and noise that does
not drive attitude directly. It is not a general INS replacement.
"""

from copy import copy
from dataclasses import replace

import casadi as ca
import numpy as np

from ._ins_correction import conditional_statistics
from ._ins_moments import psd_root
from ._kalman import symmetrize


def augmented_map(ir, *, process_noise=True):
    """Return the physical transition and its 12-dimensional projection."""
    sys, spec = ir.sys, ir.spec
    if sys.propagation != "preintegrated" or not hasattr(sys, "boundary_state_name"):
        raise ValueError("conditional prediction requires the active gyro endpoint")
    if np.any(sys.lever_arm) or "P_consider" in ir.module().state:
        raise ValueError("conditional prediction requires colocated IMU and no Schmidt state")
    if not hasattr(spec, "gravity"):
        raise ValueError("conditional prediction requires the finite INS chart")
    active = []
    if process_noise and sys.L_sym is not None:
        active = [int(i) for i in np.flatnonzero(
            np.any(np.asarray(ca.DM(sys.L_sym.sparsity())), axis=0)) if sys.Sigma[i, i] > 0]
    n = spec.tangent_dim
    nq = len(active)
    delta = ca.MX.sym("conditional_augmented_error", n+nq+12)
    noise = ca.MX.zeros(sys.n_sym.numel())
    for j, i in enumerate(active):
        noise[i] = np.sqrt(sys.Sigma[i, i])*delta[n+j]
    root = psd_root(sys.boundary_joint_covariance_sym)[3:, 3:]
    output = sys.packet_residual_fn(spec.boxplus_sym(sys.x_sym, delta[:n]),
                                    sys.u_sym, sys.dt_sym, sys.t_sym, noise, root@delta[n+nq:])
    axes = []
    for name in (spec.orientation.name, sys.imu_name+".gyro_bias", sys.boundary_state_name):
        slot = spec.slot(name)
        axes.extend(range(slot.tangent_offset, slot.tangent_offset+3))
    projection = ca.vertcat(ca.DM.eye(n+nq+12)[axes, :],
                            ca.horzcat(ca.MX.zeros(3, n+nq), root[:3, :]))
    fn = ca.Function("conditional_packet_map", [sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym, delta],
                     [output, ca.jacobian(output, delta), projection],
                     ["x", "u", "dt", "t", "delta"], ["value", "jacobian", "projection"])
    return fn, nq


def predict_conditional(ir, P, *, process_noise=True, extra_Q=None):
    sys, spec = ir.sys, ir.spec
    x, u, dt, t = sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym
    fn, nq = augmented_map(ir, process_noise=process_noise)
    n = spec.tangent_dim
    zero = ca.MX.zeros(n+nq+12)
    projection = fn(x, u, dt, t, zero)[2]
    joint = ca.diagcat(P, ca.MX.eye(nq+12))
    _, _, _, details = conditional_statistics(zero, joint, projection,
        lambda d: fn(x, u, dt, t, d)[:2])
    _, values, wm, wc, remaining = details
    # At fixed nonlinear coordinates the output quaternion is fixed and
    # ambient Euclidean coordinates are affine in the conditional remainder.
    # Evaluate their conditional covariances once, outside mean iterations.
    conditional_covariances = [H@remaining@H.T for _, H in values]
    mean = values[0][0]
    for _ in range(3):
        offset = sum((w*spec.boxminus_sym(value, mean) for w, (value, _) in zip(wm, values)),
                     ca.MX.zeros(n))
        mean = spec.boxplus_sym(mean, offset)
    y = ca.MX.sym("conditional_output", spec.ambient_dim)
    reference = ca.MX.sym("conditional_reference", spec.ambient_dim)
    error = spec.boxminus_sym(y, reference)
    chart = ca.Function("conditional_output_chart", [y, reference], [error, ca.jacobian(error, y)])
    covariance = ca.MX.zeros(n, n)
    for (value, _), local_cov, a, b in zip(values, conditional_covariances, wm, wc):
        error, H = chart(value, mean)
        covariance += b*error@error.T+a*H@local_cov@H.T
    if extra_Q is not None:
        covariance += extra_Q
    return mean, symmetrize(covariance)


def with_conditional_prediction(ir):
    """Replace prediction only; updates and physical prior remain identical."""
    module, sys, spec = ir.module(), ir.sys, ir.spec
    P = ca.MX.sym("P", spec.tangent_dim, spec.tangent_dim)
    Q = ca.MX.sym("Q", spec.tangent_dim, spec.tangent_dim)
    functions = dict(module.functions)
    for override in (False, True):
        name = "predict_with_Q" if override else "predict"
        mean, covariance = predict_conditional(ir, P, process_noise=not override,
                                               extra_Q=Q if override else None)
        inputs = [sys.x_sym, P]+([Q] if override else [])+[sys.u_sym, sys.dt_sym, sys.t_sym]
        names = ["x", "P"]+(["Q"] if override else [])+["u", "dt", "t"]
        functions[name] = ca.Function("research_conditional_"+name, inputs, [mean, covariance],
                                      names, ["x_new", "P_new"])
    result = copy(ir)
    result._module = replace(module, functions=functions, metadata={**module.metadata,
        "research_prediction": "conditional-12", "research_prediction_points": 25,
        "research_prediction_contract": "constant local gravity; affine conditional complement; no displaced IMU"})
    for name, function in module.functions.items():
        if not name.startswith("predict"):
            assert result.module().functions[name] is function
    return result
