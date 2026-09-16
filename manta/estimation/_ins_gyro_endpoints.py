"""Research-only point-sampled gyro integration with shared endpoint noise.

Use ordered half-rotations at both ends of each uniform raw interval. Keep
the existing left-held specific-force quadrature and its Earth time moments.
This is second order for a smooth gyro signal, not a general second-order
translation scheme or an assumption appropriate to interval-averaged sensors.

The 9-delta/3-endpoint joint covariance is propagated before marginalizing
the old endpoint. Merely averaging rates and halving independent sample
variances would incorrectly discard the shared sample between intervals.
The existing INS packet consumer can condition on these boundary variables;
the production independent-endpoint framer/composer MUST NOT frame this output.
"""

import casadi as ca

from ..ir._rotation import quat_mul, quat_to_rotmat, so3_exp
from .imu_preintegrator import _local_error, _normalise


def gyro_endpoint_update(block):
    """Return a uniform-cadence recurrence; append right gyro to the 12 inputs.

    The base block supplies schema, initialization and unchanged bookkeeping.
    Inputs must contain contiguous point samples. Endpoint sample sigmas are
    density/sqrt(dt), including at packet boundaries. Nonuniform cadence and
    pre-averaged sensor output require a different acquisition noise contract.
    """
    original = block.update_fn
    x, dt, t = original.mx_in(0), original.mx_in(2), original.mx_in(3)
    u = ca.MX.sym("endpoint_inputs", 15)
    auxiliary = original(x, u[:12], dt, t)[0]
    layout, offset = {}, 0
    for field in block.outputs:
        layout[field.name] = slice(offset, offset+field.dim)
        offset += field.dim

    def get(name, value=x):
        return value[layout[name]]

    q, v, p = (get(name) for name in
               ("delta_orientation", "delta_velocity", "delta_position"))
    bg = get("gyro_bias_reference", auxiliary)
    ba = get("accel_bias_reference", auxiliary)
    a, left, right = u[:3]-ba, u[3:6]-bg, u[12:15]-bg

    def compose(q0, v0, p0, a0, w0, w1):
        force = quat_to_rotmat(q0) @ a0
        rotation = quat_mul(so3_exp(.5*dt*w0), so3_exp(.5*dt*w1))
        return (_normalise(quat_mul(q0, rotation)),
                v0+dt*force, p0+dt*v0+.5*dt*dt*force)

    nominal = compose(q, v, p, a, left, right)

    def derivative(expression, variable):
        return ca.substitute(ca.jacobian(expression, variable), variable,
                             ca.MX.zeros(variable.shape))

    error = ca.MX.sym("delta_error", 9)
    propagated = compose(quat_mul(q, so3_exp(error[:3])),
                         v+error[3:6], p+error[6:9], a, left, right)
    A = derivative(_local_error(*propagated, *nominal), error)
    db = ca.MX.sym("bias_error", 6)
    biased = compose(q, v, p, a-db[3:6], left-db[:3], right-db[:3])
    B = derivative(_local_error(*biased, *nominal), db)
    noise = ca.MX.sym("independent_sample_errors", 9)
    sg = block.gyro_noise_density/ca.sqrt(dt)
    sa = block.accel_noise_density/ca.sqrt(dt)
    noisy = compose(q, v, p, a+sa*noise[6:9],
                    left+sg*noise[:3], right+sg*noise[3:6])
    G = derivative(_local_error(*noisy, *nominal), noise)
    L, E, F = G[:, :3], G[:, 3:6], G[:, 6:9]
    C = ca.reshape(get("covariance"), 9, 9)
    C_left = ca.reshape(get("delta_end_gyro_cross_covariance"), 9, 3)
    C_start = ca.reshape(get("delta_start_gyro_cross_covariance"), 9, 3)
    first = get("sample_count") < .5
    left_start = ca.if_else(first, ca.MX.eye(3),
                           ca.reshape(get("start_end_gyro_correlation"), 3, 3).T)
    propagated_cross = A @ C_left @ L.T
    C_new = (A @ C @ A.T + L @ L.T + E @ E.T + F @ F.T
             + propagated_cross + propagated_cross.T)
    replacements = {
        "delta_orientation": nominal[0], "delta_velocity": nominal[1],
        "delta_position": nominal[2], "covariance": .5*(C_new+C_new.T),
        "delta_start_gyro_cross_covariance": A @ C_start + L @ left_start,
        "delta_end_gyro_cross_covariance": E,
        "start_end_gyro_correlation": ca.MX.zeros(3, 3),
        "bias_jacobian": A @ ca.reshape(get("bias_jacobian"), 9, 6) + B,
        "end_gyro": u[12:15],
    }
    next_state = ca.vertcat(*[
        ca.vec(replacements.get(field.name, get(field.name, auxiliary)))
        for field in block.outputs])
    return ca.Function("gyro_endpoint_preintegration", [x, u, dt, t],
                       [next_state, next_state])
