"""Nonlinear INS prediction with joint packet and Schmidt uncertainty.

Positive covariance weights, no variance floor. The quadrature augments the
navigation state with process noise and the actual correlated packet/boundary
variables; the same mechanization is evaluated at every point. Static Schmidt
parameters keep zero mean and unit covariance while their navigation cross
covariance is propagated in the new error chart.
"""

import casadi as ca
import numpy as np

from ._kalman import symmetrize, unscented_weights, ut_predict


def psd_root(matrix):
    """A scaled Cholesky square root allowing exact zero-variance directions.

    Only correlation-scale roundoff is truncated (64*n machine eps). A
    materially negative pivot or nonzero residual at a zero pivot produces
    NaN. No physical covariance or diagonal variance is added.
    """
    n = matrix.size1()
    tolerance = 64 * max(1, n) * np.finfo(float).eps
    scale = [ca.sqrt(matrix[i, i]) for i in range(n)]
    rows = [[ca.MX(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            denominator = scale[i] * scale[j]
            value = ca.if_else(
                denominator > 0,
                matrix[i, j] / denominator,
                ca.if_else(matrix[i, j] == 0, 0, float("nan")),
            )
            value -= sum((rows[i][k] * rows[j][k] for k in range(j)), ca.MX(0))
            if i == j:
                rows[i][j] = ca.sqrt(
                    ca.if_else(value >= -tolerance, ca.fmax(value, 0), float("nan"))
                )
            else:
                rows[i][j] = ca.if_else(
                    rows[j][j] > 0,
                    value / rows[j][j],
                    ca.if_else(ca.fabs(value) <= tolerance, 0, float("nan")),
                )
    return ca.vertcat(
        *(ca.horzcat(*(scale[i] * rows[i][j] for j in range(n))) for i in range(n))
    )


def predict_moments(sys, spec, P, C, *, process_noise=True, extra_Q=None):
    x, u, dt, t = sys.x_sym, sys.u_sym, sys.dt_sym, sys.t_sym
    n = spec.tangent_dim
    nc = 0 if C is None else C.size2()
    packet = sys.propagation == "preintegrated"
    boundary_state = hasattr(sys, "boundary_state_name")
    ns = nc - (3 if packet and not boundary_state else 0)
    joint = ca.vertcat(ca.horzcat(P, C), ca.horzcat(C.T, ca.MX.eye(nc))) if nc else P
    root = psd_root(joint)
    active = []
    if process_noise and sys.L_sym is not None:
        active = [
            int(i)
            for i in np.flatnonzero(
                np.any(np.asarray(ca.DM(sys.L_sym.sparsity())), axis=0)
            )
            if sys.Sigma[i, i] > 0
        ]
    nn = sys.n_sym.numel()
    total = n + nc + len(active) + (12 if packet else 0)
    wm, wc, spread = unscented_weights(total, 1.0, 2.0, 0.0)
    raw = ca.Function("ins_raw_noisy", [x, u, dt, t, sys.n_sym], [sys.x_new_noisy])
    zero_joint = ca.MX.zeros(n + nc)
    zero_noise = ca.MX.zeros(nn)
    zero_packet = ca.MX.zeros(12)
    points = [(zero_joint, zero_noise, zero_packet)]
    # Condition by factoring [start, delta, end] jointly, with start first.
    # Subtracting B B.T and then taking diagonal square roots loses the
    # unconditional scale: FMA leaves roundoff in an exactly-zero one-sample
    # conditional block, which residual normalization amplifies. The joint
    # factor removes roundoff in
    # correlation units and its trailing block is the conditional root.
    packet_root = (
        psd_root(sys.boundary_joint_covariance_sym)[3:, 3:] if packet else None
    )
    for sign in (1.0, -1.0):
        for i in range(n + nc):
            points.append((sign * spread * root[:, i], zero_noise, zero_packet))
        for i in active:
            noise = ca.MX(zero_noise)
            noise[i] = sign * spread * np.sqrt(sys.Sigma[i, i])
            points.append((zero_joint, noise, zero_packet))
        if packet:
            for i in range(12):
                points.append(
                    (zero_joint, zero_noise, sign * spread * packet_root[:, i])
                )
    propagated, considered = [], []
    for joint_delta, noise, residual in points:
        state = spec.boxplus_sym(x, joint_delta[:n])
        if packet and boundary_state:
            propagated.append(sys.packet_residual_fn(state, u, dt, t, noise, residual))
            considered.append(joint_delta[n:])
        elif packet:
            start = joint_delta[n + ns : n + nc]
            packet_error = sys.boundary_conditional_gain_sym @ start + residual
            propagated.append(
                sys.packet_noisy_fn(
                    state, u, dt, t, noise, packet_error[:9], start, packet_error[9:12]
                )
            )
            considered.append(ca.vertcat(joint_delta[n : n + ns], packet_error[9:12]))
        else:
            propagated.append(raw(state, u, dt, t, noise))
            considered.append(joint_delta[n:])
    mean, covariance = ut_predict(propagated, ca.MX.zeros(n, n), wm, wc, spec, 3)
    cross = None
    if nc:
        cross = sum(
            (
                weight * spec.boxminus_sym(point, mean) @ nuisance.T
                for weight, point, nuisance in zip(wc, propagated, considered)
            ),
            ca.MX.zeros(n, nc),
        )
    if extra_Q is not None:
        covariance = symmetrize(covariance + extra_Q)
    return mean, covariance, cross


def prior_moments_function(spec, consider_dimension=0):
    """Map a physical product-Gaussian prior into the INS finite error chart.

    Conditional Gauss-Hermite integration uses five nodes per attitude axis.
    For a fixed attitude the remaining INS Euclidean variables enter the chart
    affinely, so their conditional covariance is integrated analytically.
    Zero attitude variance is supported without adding a variance floor.
    """
    import itertools

    n = spec.tangent_dim
    x = ca.MX.sym("prior_x", spec.ambient_dim)
    P = ca.MX.sym("prior_P", n, n)
    start = spec.orientation.tangent_offset
    axes = list(range(start, start + 3))
    root = psd_root(P[axes, axes])
    conditional = ca.MX.zeros(n, 3)
    for j in range(3):
        residual = P[:, axes[j]] - sum(
            (conditional[:, k] * root[j, k] for k in range(j)), ca.MX.zeros(n)
        )
        conditional[:, j] = ca.if_else(root[j, j] > 0, residual / root[j, j], 0)
    remaining = symmetrize(P - conditional @ conditional.T)
    d = ca.MX.sym("prior_delta", n)
    reference = ca.MX.sym("reference", spec.ambient_dim)
    physical_point = spec.product_spec.boxplus_sym(x, d)
    error = spec.boxminus_sym(physical_point, reference)
    # A wrapped broad physical prior is not a Gaussian in this local chart.
    # Reject quadrature points outside the attitude logarithm branch instead
    # of aliasing them into an apparently confident initialization.
    error = ca.if_else(ca.dot(d[axes], d[axes]) < np.pi**2, error, ca.MX.nan(n, 1))
    errors_fn = ca.Function(
        "ins_prior_error", [x, d, reference], [error, ca.jacobian(error, d)]
    )
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    deltas, probabilities = [], []
    for indices in itertools.product(range(5), repeat=3):
        deltas.append(conditional @ ca.DM(nodes[list(indices)] * np.sqrt(2)))
        probabilities.append(float(np.prod(weights[list(indices)]) / np.pi**1.5))
    mean = x
    for _ in range(5):
        offset = sum(
            (w * errors_fn(x, d, mean)[0] for w, d in zip(probabilities, deltas)),
            ca.MX.zeros(n),
        )
        mean = spec.boxplus_sym(mean, offset)
    covariance = ca.MX.zeros(n, n)
    for w, d in zip(probabilities, deltas):
        error, jacobian = errors_fn(x, d, mean)
        covariance += w * (error @ error.T + jacobian @ remaining @ jacobian.T)
    outputs = [mean, symmetrize(covariance)]
    output_names = ["x_new", "P_new"]
    if consider_dimension:
        outputs.append(ca.MX.zeros(n, consider_dimension))
        output_names.append("P_consider_new")
    full_initializer = ca.Function(
        "ins_initialize_prior", [x, P], outputs, ["prior_x", "prior_P"], output_names
    )
    physical = getattr(spec.product_spec, "physical_spec", None)
    if physical is None:
        return full_initializer
    xp = ca.MX.sym("prior_x", physical.ambient_dim)
    pp = ca.MX.sym("prior_P", physical.tangent_dim, physical.tangent_dim)
    augmented_x = ca.vertcat(xp, ca.MX.zeros(3))
    augmented_P = ca.diagcat(pp, ca.MX.eye(3))
    return ca.Function(
        "ins_initialize_physical_prior",
        [xp, pp],
        list(full_initializer(augmented_x, augmented_P)),
        ["prior_x", "prior_P"],
        output_names,
    )


def with_prior_initialization(module, spec, consider_dimension=0):
    """Add the deployable physical-prior entry and map the Module defaults."""
    from dataclasses import replace

    from ..ir.module import EntryPoint, Port, PortRef, Role, StateField, StateLayout

    initializer = prior_moments_function(spec, consider_dimension)
    physical = getattr(spec.product_spec, "physical_spec", spec.product_spec)
    x0 = np.asarray(module.state.field("x").init)[: physical.ambient_dim]
    P0 = np.asarray(module.state.field("P").init)[
        : physical.tangent_dim, : physical.tangent_dim
    ]
    initialized = initializer(x0, P0)
    fields = list(module.state.fields)
    for index, name in enumerate(
        ["x", "P"] + (["P_consider"] if consider_dimension else [])
    ):
        field = module.state.field(name)
        fields[index] = StateField(
            field.name,
            field.kind,
            field.shape,
            init=np.asarray(initialized[index]).reshape(field.shape),
            spec=field.spec,
        )
    ports = (
        *module.ports,
        Port("prior_x", Role.STATE, (physical.ambient_dim,), spec=physical, init=x0),
        Port(
            "prior_P",
            Role.MATRIX,
            (physical.tangent_dim, physical.tangent_dim),
            init=P0,
        ),
    )
    writes = ("x", "P") + (("P_consider",) if consider_dimension else ())
    return replace(
        module,
        state=StateLayout(tuple(fields)),
        ports=ports,
        functions={**module.functions, "initialize_prior": initializer},
        entry_points=(
            *module.entry_points,
            EntryPoint(
                "initialize_prior",
                "initialize_prior",
                (PortRef("prior_x"), PortRef("prior_P")),
                writes=writes,
            ),
        ),
        metadata={
            **module.metadata,
            "prior_coordinates": "physical_product_gaussian",
            "prior_mapping": "conditional_hermite_5_per_attitude_axis",
        },
    )
