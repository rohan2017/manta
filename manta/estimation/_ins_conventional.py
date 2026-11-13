"""Modified-velocity INS coordinates, without gravity-dependent bias transport.

R_true = Exp(theta) R, v_true = Exp(theta) (v + dv), b_true = b + db.
The discrete Manta mechanization and sensor equations remain authoritative:
their derivatives are transformed into these coordinates. This is a research
baseline; a local Gaussian does not represent a broad curved bias posterior.
"""

from functools import cached_property

import casadi as ca
import numpy as np

from ..ir._rotation import quat_mul, quat_to_rotmat, so3_exp
from ..linearization.engine import SensorModel
from ._ins_error import INSStateSpec, _ambient, _tangent


class ConventionalINSStateSpec(INSStateSpec):
    error_model = "modified_velocity_additive_bias"

    def __init__(self, *args, retain_covariance=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_covariance = retain_covariance

    def _plus(self, x, d):
        out = self.product_spec.boxplus_sym(x, d)
        theta = d[_tangent(self.orientation)]
        dq = so3_exp(theta)
        D = quat_to_rotmat(dq)
        out[_ambient(self.orientation)] = quat_mul(dq, x[_ambient(self.orientation)])
        for slot in self.navigation_vectors:
            out[_ambient(slot)] = D @ (x[_ambient(slot)] + d[_tangent(slot)])
        return out

    def _minus(self, a, b):
        out = self.product_spec.boxminus_sym(a, b)
        D = (
            quat_to_rotmat(a[_ambient(self.orientation)])
            @ quat_to_rotmat(b[_ambient(self.orientation)]).T
        )
        for slot in self.navigation_vectors:
            out[_tangent(slot)] = D.T @ a[_ambient(slot)] - b[_ambient(slot)]
        return out

    @cached_property
    def exact_reset(self):
        if not self.retain_covariance:
            return super().exact_reset
        x = ca.MX.sym("x", self.ambient_dim)
        d = ca.MX.sym("d", self.tangent_dim)
        # Deliberate local-feedback approximation used by the comparator.
        return ca.Function(
            "conventional_local_reset", [x, d], [ca.MX.eye(self.tangent_dim)]
        )


def transform_derivatives(sys):
    """Convert physical F/H/Q to the selected modified-velocity tangent."""
    spec, x, n = sys.spec, sys.x_sym, sys.spec.tangent_dim
    d = ca.MX.sym("coordinate_delta", n)
    physical_error = spec.product_spec.boxminus_sym(spec.boxplus_sym(x, d), x)
    J = ca.substitute(ca.jacobian(physical_error, d), d, ca.MX.zeros(n))
    inverse_error = spec.boxminus_sym(spec.product_spec.boxplus_sym(x, d), x)
    T = ca.substitute(ca.jacobian(inverse_error, d), d, ca.MX.zeros(n))
    Tnext = ca.substitute(T, x, sys.x_new)
    sys.F_sym = Tnext @ sys.F_sym @ J
    if sys.L_sym is not None:
        sys.L_sym = Tnext @ sys.L_sym
    sys.packet_Q_sym = Tnext @ sys.packet_Q_sym @ Tnext.T
    if hasattr(sys, "packet_residual_G_sym"):
        sys.packet_residual_G_sym = Tnext @ sys.packet_residual_G_sym
        from ._ins_moments import psd_root

        root = psd_root(sys.boundary_joint_covariance_sym)[
            sys.boundary_dim :, sys.boundary_dim :
        ]
        mapped = sys.packet_residual_G_sym @ root
        sys.packet_Q_sym = mapped @ mapped.T
    for name, sm in list(sys.sensors.items()):
        H = sm.H_sym @ J
        cols = np.flatnonzero(np.asarray(ca.DM(H.sparsity())).any(axis=0))
        Hfn = ca.Function("conventional_H", [x, sys.u_sym, sys.dt_sym, sys.t_sym], [H])
        sys.sensors[name] = SensorModel(
            sm.full, sm.dim, sm.h_sym, sm.h_noisy_sym, H, sm.L_h_sym, cols, Hfn
        )
    sys.F_fn = ca.Function(
        "conventional_F", [x, sys.u_sym, sys.dt_sym, sys.t_sym], [sys.F_sym]
    )
    if sys.L_sym is not None:
        sys.L_fn = ca.Function(
            "conventional_L", [x, sys.u_sym, sys.dt_sym, sys.t_sym], [sys.L_sym]
        )
    from ..linearization.partition import partition_blocks

    sys.blocks = partition_blocks(
        n,
        np.asarray(ca.DM(sys.F_sym.sparsity())),
        np.asarray(ca.DM(sys.L_sym.sparsity())) if sys.L_sym is not None else None,
        [sm.observed_cols for sm in sys.sensors.values()],
    )
