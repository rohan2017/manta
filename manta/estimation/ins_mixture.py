"""Gaussian-sum runtime over Manta's compiled INS kernels.

A single Gaussian loses the curved tilt/calibration posterior in precise,
weakly excited navigation. This runtime splits that *posterior*, including its
cross-covariances, and retains the branches through subsequent predictions and
measurement likelihoods. Public moments use physical product coordinates.

The initial interval is processed once by the component filter. Splitting adds
no observations, assumes no stationarity, and changes no sensor/bias noise.
This is a NumPy runtime, not a new mechanization or a C++ Module transform.
"""

from dataclasses import dataclass
from itertools import product
import math

import casadi as ca
import numpy as np

from ._ins_moments import psd_root
from ..ir.module import entry_ident


def _moments(spec, target, order=5):
    """Conditional quadrature: attitude numerical, remaining coordinates affine."""
    n, a = spec.tangent_dim, spec.ambient_dim
    x, d, ref = ca.MX.sym("x", a), ca.MX.sym("d", n), ca.MX.sym("ref", a)
    P, mu = ca.MX.sym("P", n, n), ca.MX.sym("mu", n)
    error = target.boxminus_sym(spec.boxplus_sym(x, d), ref)
    value = ca.Function("mixture_error", [x, d, ref], [error])
    derivative = ca.Function(
        "mixture_error_jacobian", [x, d, ref], [ca.jacobian(error, d)]
    )
    oi = spec.orientation.tangent_offset
    axes = list(range(oi, oi + 3))
    perm = axes + [i for i in range(n) if i not in axes]
    root = psd_root(P[perm, perm])[np.argsort(perm).tolist(), :]
    remaining = root[:, 3:]
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    mean, second = ca.MX.zeros(n), ca.MX.zeros(n, n)
    for index in product(range(order), repeat=3):
        delta = mu + root[:, :3] @ ca.DM(np.sqrt(2) * nodes[list(index)])
        weight = float(np.prod(weights[list(index)]) / np.pi**1.5)
        e = value(x, delta, ref)
        L = derivative(x, delta, ref) @ remaining
        mean += weight * e
        second += weight * (e @ e.T + L @ L.T)
    args = [x, P, ref, mu]
    return (
        ca.Function("mixture_mean_error", args, [mean]),
        ca.Function("mixture_second_moment", args, [second]),
    )


def mixture_kernels(spec, *, order=5, within_fraction=0.01):
    """Pure splitting and physical-moment kernels, shared by runtime and audits."""
    if type(order) is not int or order < 3 or order % 2 != 1:
        raise ValueError("mixture quadrature order must be odd and at least 3")
    if not 0 < within_fraction < 1:
        raise ValueError("within_fraction must lie strictly between zero and one")
    n, a = spec.tangent_dim, spec.ambient_dim
    x, P = ca.MX.sym("x", a), ca.MX.sym("P", n, n)
    mean, second = _moments(spec, spec)
    # Express the angular block in a basis whose last axis is gravity up.
    up = np.asarray(spec.up).ravel()
    axis = np.eye(3)[np.argmin(np.abs(up))]
    horizontal = axis - up * (up @ axis)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.column_stack((horizontal, np.cross(up, horizontal), up))
    oi = spec.orientation.tangent_offset
    T = np.eye(n)
    T[oi : oi + 3, oi : oi + 3] = basis.T
    axes = [oi, oi + 1]
    perm = axes + [i for i in range(n) if i not in axes]
    rotated = ca.DM(T) @ P @ ca.DM(T).T
    root = ca.DM(T).T @ psd_root(rotated[perm, perm])[np.argsort(perm).tolist(), :]
    residual = ca.horzcat(root[:, :2] * np.sqrt(within_fraction), root[:, 2:])
    within = residual @ residual.T
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    means, covariances, probabilities = [], [], []
    for index in product(range(order), repeat=2):
        delta = root[:, :2] @ ca.DM(
            np.sqrt(2 * (1 - within_fraction)) * nodes[list(index)]
        )
        center = spec.boxplus_sym(x, delta)
        for _ in range(4):
            center = spec.boxplus_sym(center, mean(x, within, center, delta))
        means.append(center)
        covariances.append(second(x, within, center, delta))
        probabilities.append(float(np.prod(weights[list(index)]) / np.pi))
    split = ca.Function(
        "ins_split_posterior", [x, P], [ca.horzcat(*means), ca.horzcat(*covariances)]
    )
    physical_mean, physical_second = _moments(spec, spec.product_spec)
    return split, physical_mean, physical_second, np.array(probabilities)


@dataclass(frozen=True)
class GaussianSumCheckpoint:
    x: np.ndarray
    P: np.ndarray
    component_x: np.ndarray
    component_P: np.ndarray
    log_weights: np.ndarray
    time: float
    contract: tuple

    def __post_init__(self):
        for name in ("x", "P", "component_x", "component_P", "log_weights"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)


class GaussianSumINS:
    """Held INS mixture using the usual reset/predict/update runtime interface.

    ``component`` is a geometric INS transform with gates disabled. Optional
    gates apply to the whole mixture's predictive moments, before any branch
    is changed. Each observation updates branch weights using its full
    Gaussian predictive likelihood, including the innovation determinant.

    Splitting follows the first accepted correction at/after ``split_after_s``.
    ``order`` and ``within_fraction`` control numerical mixture resolution;
    they do not alter the physical prior. Check quadrature convergence for new
    operating envelopes. Static Schmidt calibration parameters are unsupported.
    """

    def __init__(
        self,
        component,
        *,
        order=5,
        within_fraction=0.01,
        split_after_s=10.0,
        gates=None,
        compile=True,
    ):
        from ..codegen.numpy._filter import NumpyFilter

        if component.covariance != "geometric":
            raise ValueError("GaussianSumINS requires geometric component coordinates")
        if not math.isfinite(split_after_s) or split_after_s < 0:
            raise ValueError("split_after_s must be finite and nonnegative")
        self.base = NumpyFilter(component.module())
        if self.base.P_consider is not None:
            raise ValueError(
                "GaussianSumINS does not support static Schmidt parameters"
            )
        if any(
            v is not None
            for v in self.base.module.metadata.get("nis_gates", {}).values()
        ):
            raise ValueError("disable component gates; supply mixture gates separately")
        self.module = self.base.module
        self.component_spec = component.spec
        self.spec = component.spec.product_spec
        self.order, self.within_fraction = order, within_fraction
        self.split_after_s = float(split_after_s)
        self.gates = {} if gates is None else dict(gates)
        for name, threshold in self.gates.items():
            self.base._resolve_sensor(name)
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("mixture gates must be finite and positive")
        self.gates = {self.base._resolve_sensor(k): v for k, v in self.gates.items()}
        self._native = bool(compile)
        self._kernels, self._mapped = {}, {}
        split, mean, second, self._split_weights = mixture_kernels(
            self.component_spec, order=order, within_fraction=within_fraction
        )
        self._auxiliary = dict(split=split, mean=mean, second=second)
        self._contract = (
            self.module.artifact_id,
            order,
            within_fraction,
            split_after_s,
            tuple(sorted(self.gates.items())),
        )
        self._revision = 0
        self.Q = None
        self.reset()

    def _kernel(self, name, count=1):
        from ..codegen.numpy._compile import compile_functions

        if name not in self._kernels:
            function = self._auxiliary.get(name, self.module.functions.get(name))
            if function is None:
                raise KeyError(name)
            self._kernels[name] = (
                compile_functions(
                    {name: function}, optimization="O1", max_instructions=150000
                )[name]
                if self._native
                else function
            )
        key = (name, count)
        if key not in self._mapped:
            self._mapped[key] = self._kernels[name].map(count)
        return self._mapped[key]

    def _changed(self):
        if not np.isfinite(self._x).all() or not np.isfinite(self._P).all():
            raise ArithmeticError("nonfinite INS mixture component")
        self._mean = self._covariance = None
        self._revision += 1

    @property
    def state_revision(self):
        return self._revision

    @property
    def time(self):
        return self._t

    @property
    def component_count(self):
        return self._x.shape[1]

    @property
    def P_consider(self):
        return None

    @property
    def rho_by_sensor(self):
        return self.base.rho_by_sensor

    def _summarize_mean(self):
        if self._mean is not None:
            return self._mean
        m, n = self.component_count, self.spec.tangent_dim
        weights = np.exp(self._log_weights)
        center = self._x[:, int(np.argmax(weights))].copy()
        zero = np.zeros((n, m))
        for _ in range(4):
            delta = (
                np.asarray(
                    self._kernel("mean", m)(
                        self._x, self._P, np.tile(center[:, None], (1, m)), zero
                    )
                )
                @ weights
            )
            center = self.spec.boxplus_num(center, delta)
        self._mean = center
        return center

    def _summarize_covariance(self):
        if self._covariance is not None:
            return self._covariance
        center = self._summarize_mean()
        m, n = self.component_count, self.spec.tangent_dim
        weights = np.exp(self._log_weights)
        zero = np.zeros((n, m))
        second = np.asarray(
            self._kernel("second", m)(
                self._x, self._P, np.tile(center[:, None], (1, m)), zero
            )
        )
        P = sum(
            (
                weight * second[:, i * n : (i + 1) * n]
                for i, weight in enumerate(weights)
            ),
            np.zeros((n, n)),
        )
        self._covariance = 0.5 * (P + P.T)
        return self._covariance

    @property
    def x(self):
        return self._summarize_mean().copy()

    @property
    def P(self):
        return self._summarize_covariance().copy()

    def state_dict(self):
        return self.spec.to_nested(self.x)

    def reset(self, state=None, *, P=None):
        self.base.reset(state, P=P)
        self._x = self.base.x[:, None]
        self._P = self.base.P
        self._log_weights = np.zeros(1)
        self._t = 0.0
        self._changed()

    def predict(self, dt, *, t=None, u=None, Q=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("predict requires finite positive dt")
        t0 = self._t if t is None else float(t)
        if not math.isfinite(t0 + dt):
            raise ValueError("predict requires finite time")
        inputs = self.base._kernel_u(u)
        self.base._check_packet_duration(dt, inputs)
        m = self.component_count
        process = self.Q if Q is None else Q
        args = [self._x, self._P]
        name = "predict"
        if process is not None:
            process = self.base._validate_covariance(
                process, who="predict Q", positive_definite=False
            )
            args.append(np.tile(process, (1, m)))
            name = "predict_with_Q"
        values = self._kernel(name, m)(*args, np.tile(inputs[:, None], (1, m)), dt, t0)
        self._x, self._P = np.asarray(values[0]), np.asarray(values[1])
        self._t = t0 + dt
        self._changed()

    def preintegrated_inputs(self, packet, *, u=None):
        return self.base.preintegrated_inputs(packet, u=u)

    def predict_preintegrated(self, packet, *, t=None, u=None, Q=None):
        self.predict(
            float(np.asarray(packet["duration"]).item()),
            t=t,
            u=self.preintegrated_inputs(packet, u=u),
            Q=Q,
        )

    def update(self, target, z=None, R=None, *, t=None, u=None):
        from ..codegen.numpy._filter import UpdateResult

        full = self.base._resolve_sensor(target)
        dim, m = self.module.port(full).size, self.component_count
        z = np.asarray(z, dtype=float).reshape(-1)
        now = self._t if t is None else float(t)
        if z.shape != (dim,) or not np.isfinite(z).all() or not math.isfinite(now):
            raise ValueError("invalid measurement or time")
        ident = entry_ident(full)
        args = [self._x, self._P, np.tile(z[:, None], (1, m))]
        name = "update_diagnostic_" + ident
        if R is not None:
            R = self.base._validate_measurement_covariance(R, dim, full)
            args.append(np.tile(R, (1, m)))
            name = "update_with_R_" + ident
        values = self._kernel(name, m)(
            *args, np.tile(self.base._kernel_u(u)[:, None], (1, m)), now
        )
        innovation = np.asarray(values[2]).T
        covariances = np.asarray(values[3]).reshape(dim, m, dim).transpose(1, 0, 2)
        weights = np.exp(self._log_weights)
        mean = weights @ innovation
        residuals = innovation - mean
        S = (
            np.einsum("m,mij->ij", weights, covariances)
            + (residuals.T * weights) @ residuals
        )
        nis = float(mean @ np.linalg.solve(S, mean))
        gate = self.gates.get(full)
        accepted = gate is None or nis <= gate
        if accepted:
            sign, logdet = np.linalg.slogdet(covariances)
            if np.any(sign <= 0) or not np.isfinite(logdet).all():
                raise ArithmeticError("invalid mixture innovation covariance")
            log_weights = self._log_weights - 0.5 * (
                np.asarray(values[4]).ravel() + logdet
            )
            normalizer = np.max(log_weights)
            log_weights -= normalizer + np.log(np.exp(log_weights - normalizer).sum())
            self._x, self._P = np.asarray(values[0]), np.asarray(values[1])
            self._log_weights = log_weights
            if m == 1 and self._t + 1e-12 >= self.split_after_s:
                self._x, self._P = map(
                    np.asarray, self._kernel("split")(self._x, self._P)
                )
                self._log_weights = np.log(self._split_weights)
            self._changed()
        return UpdateResult(full, mean, S, nis, accepted, gate, R is not None)

    def checkpoint(self):
        return GaussianSumCheckpoint(
            self.x, self.P, self._x, self._P, self._log_weights, self._t, self._contract
        )

    def restore(self, checkpoint):
        if (
            not isinstance(checkpoint, GaussianSumCheckpoint)
            or checkpoint.contract != self._contract
        ):
            raise ValueError("checkpoint belongs to a different INS mixture")
        x, P, weights = (
            checkpoint.component_x,
            checkpoint.component_P,
            checkpoint.log_weights,
        )
        m, n, a = len(weights), self.spec.tangent_dim, self.spec.ambient_dim
        if (
            m not in (1, self.order**2)
            or x.shape != (a, m)
            or P.shape != (n, n * m)
            or not all(np.isfinite(v).all() for v in (x, P, weights))
            or not math.isfinite(checkpoint.time)
            or not np.isclose(np.exp(weights).sum(), 1.0, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("invalid INS mixture checkpoint")
        for i in range(m):
            self.base._validate_covariance(
                P[:, i * n : (i + 1) * n],
                who="restore mixture P",
                positive_definite=False,
            )
        self._x, self._P, self._log_weights = x.copy(), P.copy(), weights.copy()
        self._t = checkpoint.time
        self._changed()

    def set_state_keep_covariance(self, state):
        raise NotImplementedError(
            "moving a mixture requires transporting every component; use reset for a new prior"
        )

    def compile_functions(self, names, **kwargs):
        from ..codegen.numpy._compile import compile_functions

        names = tuple(dict.fromkeys(names))
        if not names:
            raise ValueError("compile_functions requires at least one function name")
        result = compile_functions(
            {name: self.module.functions[name] for name in names}, **kwargs
        )
        self._kernels.update(result)
        self._mapped.clear()
        return self

    def compile_sensor_updates(self, sensors, *, covariance="model", **kwargs):
        if covariance not in ("model", "per_sample"):
            raise ValueError("unknown measurement covariance mode")
        prefix = (
            "update_with_R_" if covariance == "per_sample" else "update_diagnostic_"
        )
        requested = (sensors,) if isinstance(sensors, str) else sensors
        self.compile_functions(
            [prefix + entry_ident(self.base._resolve_sensor(s)) for s in requested],
            **kwargs,
        )
        return self
