"""Transient gyro-boundary error state for nonlinear packet INS.

A static Schmidt nuisance deliberately keeps its mean and covariance fixed.
That approximation is unsuitable for the nonlinear lever transform when aiding
learns the current endpoint gyro error and the next packet reuses that sample.
Here its mean, covariance and navigation cross covariance are updated together.
The next endpoint is replaced using the packet's stated conditional law.
No vehicle angular-rate dynamics or persistent IMU bias is introduced.
"""

import copy

import casadi as ca
import numpy as np

from ..ir.manifold import R3Manifold
from ..ir.module import entry_ident
from ..ir.state_spec import StateSpec, flatten_nested
from ..linearization.engine import SensorModel
from ..linearization.partition import partition_blocks


class BoundaryStateSpec(StateSpec):
    def __init__(self, physical, name):
        extended = StateSpec.from_layout(
            [(s.name, s.manifold) for s in physical.slots] + [(name, R3Manifold())]
        )
        super().__init__(list(extended.slots))
        self.physical_spec = physical
        self.boundary_state_name = name

    def pack_projected(self, source):
        values = flatten_nested(source)
        values.setdefault(self.boundary_state_name, np.zeros(3))
        return super().pack_projected(values)


def with_boundary_state(base):
    sys = copy.copy(base)
    name = f"{base.imu_name}.gyro_boundary_error"
    spec = BoundaryStateSpec(base.spec, name)
    sys.spec = spec
    sys.boundary_state_name = name
    old_na = base.spec.ambient_dim
    n = spec.tangent_dim
    x = ca.MX.sym("x", spec.ambient_dim)
    u, dt, t, noise = base.u_sym, base.dt_sym, base.t_sym, base.n_sym
    residual = ca.MX.sym("packet_residual", 12)
    start = x[old_na:]
    packet_error = base.boundary_conditional_gain_sym @ start + residual
    physical = base.packet_noisy_fn(
        x[:old_na], u, dt, t, noise, packet_error[:9], start, packet_error[9:]
    )
    full = ca.vertcat(physical, packet_error[9:])
    sys.packet_residual_fn = ca.Function(
        "ins_boundary_state_packet", [x, u, dt, t, noise, residual], [full]
    )
    x_noisy = ca.substitute(full, residual, ca.MX.zeros(12))
    x_new = ca.substitute(x_noisy, noise, ca.MX.zeros(noise.numel()))
    d = ca.MX.sym("delta", n)
    error = spec.boxminus_sym(ca.substitute(x_new, x, spec.boxplus_sym(x, d)), x_new)
    F = ca.substitute(ca.jacobian(error, d), d, ca.MX.zeros(n))
    L = (
        ca.substitute(
            ca.jacobian(spec.boxminus_sym(x_noisy, x_new), noise),
            noise,
            ca.MX.zeros(noise.numel()),
        )
        if noise.numel()
        else None
    )
    packet_error_output = spec.boxminus_sym(
        ca.substitute(full, noise, ca.MX.zeros(noise.numel())), x_new
    )
    G = ca.substitute(
        ca.jacobian(packet_error_output, residual), residual, ca.MX.zeros(12)
    )
    sys.x_sym = x
    sys.x_new = x_new
    sys.x_new_noisy = x_noisy
    sys.F_sym = F
    sys.L_sym = L
    sys.packet_Q_sym = G @ base.boundary_conditional_covariance_sym @ G.T
    sys.packet_residual_G_sym = G
    sys.predict_fn = ca.Function("predict", [x, u, dt, t], [x_new])
    sys.F_fn = ca.Function("F", [x, u, dt, t], [F])
    sys.L_fn = ca.Function("L", [x, u, dt, t], [L]) if L is not None else None
    sensors = {}
    supports = []
    for full_name, sm in base.sensors.items():
        h_noisy = base.boundary_sensor_functions[full_name](
            x[:old_na], u, dt, t, noise, start
        )
        h = ca.substitute(h_noisy, noise, ca.MX.zeros(noise.numel()))
        H = ca.substitute(
            ca.jacobian(ca.substitute(h, x, spec.boxplus_sym(x, d)), d),
            d,
            ca.MX.zeros(n),
        )
        Lh = (
            ca.substitute(
                ca.jacobian(h_noisy, noise), noise, ca.MX.zeros(noise.numel())
            )
            if noise.numel()
            else None
        )
        cols = np.flatnonzero(np.asarray(ca.DM(H.sparsity())).any(axis=0))
        supports.append(cols)
        Hfn = ca.Function(f"H_{entry_ident(full_name)}", [x, u, dt, t], [H])
        sensors[full_name] = SensorModel(
            full_name, sm.dim, h, h_noisy, H, Lh, cols, Hfn
        )
    sys.sensors = sensors
    sys.blocks = partition_blocks(
        n,
        np.asarray(ca.DM(F.sparsity())),
        np.asarray(ca.DM(L.sparsity())) if L is not None else None,
        supports,
    )
    return sys
