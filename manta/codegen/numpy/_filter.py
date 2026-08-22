"""NumpyFilter — the HELD predict/update filter view."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from ..._validation import require_finite, require_positive
from ...estimation._kalman import joseph_update_np
from ...ir._names import resolve_suffix
from ...ir.module import entry_ident
from ._runtime import NumpyRuntime

_LOG = logging.getLogger(__name__)


def _psd_roundoff_tolerance(matrix: np.ndarray) -> float:
    """Legacy absolute floor plus a scale-aware eigensolver error bound."""
    dimension = matrix.shape[0]
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    return max(1e-12, 64.0 * np.finfo(float).eps * dimension * scale)


@dataclass(frozen=True)
class FilterCheckpoint:
    """Complete restart point for a filter runtime.

    Arrays are owned snapshots rather than views into the live runtime.
    ``time`` is the filter's logical model time, not a wall clock.
    """

    x: np.ndarray
    P: np.ndarray
    time: float
    artifact_id: str

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        P = np.asarray(self.P)
        if x.dtype.kind not in "iuf" or P.dtype.kind not in "iuf":
            raise TypeError("FilterCheckpoint: x and P must be real numeric arrays")
        x = np.asarray(x, dtype=float).copy()
        P = np.asarray(P, dtype=float).copy()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
            raise ValueError("FilterCheckpoint: x and P must be finite")
        if not np.isfinite(self.time):
            raise ValueError("FilterCheckpoint: time must be finite")
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("FilterCheckpoint: artifact_id is required")
        x.flags.writeable = False
        P.flags.writeable = False
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "time", float(self.time))


@dataclass(frozen=True)
class UpdateResult:
    """Diagnostics and disposition of one measurement fold."""

    sensor: str
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    accepted: bool
    gate: float | None
    covariance_overridden: bool


class NumpyFilter(NumpyRuntime):
    """A predict/update filter over a held `x`/`P`, with baked per-sensor
    update kernels — the same surface every backend emits.

    You own the loop, identically in numpy and C++: fold each fresh
    measurement at the pre-predict state, then predict.

        for nm in sensors:
            if gate[nm].due(t):
                ekf.update(nm, sim.reading(nm), u=u)   # update-then-...
        ekf.predict(dt, u=u)                           # ...-predict

    The update-then-predict order is yours to keep: a reading sampled at the
    interval start belongs against the current (pre-predict) state.

    Clock: same convention as `NumpySim` — the runtime tracks `t`,
    `predict(dt)` advances it, and an explicit `t=` overrides it for that
    call. (The kernels stay pure; this is caller-side bookkeeping the two
    runtimes must agree on: a filter that silently pinned `t=0` while the
    sim advanced left every time-dependent world linearized at t=0.)
    """

    def __init__(self, module) -> None:
        super().__init__(module)
        self._Q: np.ndarray | None = None        # default process noise
        self._custom_h_cache: dict = {}          # h_sym -> (h_fn, H_fn)
        rho_warning = module.metadata.get("rho_warning")
        for sensor, rho in module.metadata.get("rho_by_sensor", {}).items():
            if rho_warning is not None and rho > rho_warning:
                _LOG.warning(
                    "%s disturbance-observer noise ratio rho[%s]=%.6g exceeds "
                    "the warning level %.3g (ceiling %.3g)",
                    module.name, sensor, rho, rho_warning,
                    module.metadata["rho_ceiling"])
                continue
            # rho is a useful dimensionless diagnostic, not a universal
            # estimator-selection threshold. Its acceptable range depends on
            # the identified model error, spectra, operating envelope, and
            # application policy, none of which this generic runtime owns.
            _LOG.info("%s disturbance-observer noise ratio rho[%s]=%.6g",
                      module.name, sensor, rho)

    # ---- estimate access -------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._state["x"].copy()

    @property
    def P(self) -> np.ndarray:
        return self._state["P"].copy()

    @property
    def estimate_vector(self) -> np.ndarray:
        return self._state["x"]

    @property
    def time(self) -> float:
        """Current logical filter time (advanced only by ``predict``)."""
        return self._t

    @property
    def rho_by_sensor(self) -> dict[str, float]:
        """INS IMU/model noise ratio diagnostics; empty for EKF/UKF."""
        return dict(self.module.metadata.get("rho_by_sensor", {}))

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Current estimate nested by owner."""
        return self._spec.to_nested(self._state["x"])

    def reset(self, state: dict | None = None, *,
              P: np.ndarray | None = None) -> None:
        """Reset state, covariance, and clock from the Module defaults.

        ``state`` is merged over the declared initial state and ``P`` may
        replace the declared initial covariance. To move only the nominal
        state while deliberately preserving covariance, use
        :meth:`set_state_keep_covariance`.
        """
        x_field = self.module.state.field("x")
        next_x = self._spec.pack_any(
            state, base=x_field.init) if state is not None else np.asarray(
                x_field.init, dtype=float).reshape(-1).copy()
        pf = self.module.state.field("P")
        next_P = np.asarray(
            pf.init, dtype=float).reshape(pf.shape).copy()
        if P is not None:
            next_P = self._validate_covariance(P, who="reset P",
                                               positive_definite=False)
        self._validate_staged_state({"x": next_x, "P": next_P})
        self._state["x"], self._state["P"], self._t = next_x, next_P, 0.0

    def reset_from_model_record(self, record: dict, *,
                                P: np.ndarray | None = None) -> None:
        """Reset from a broader authoring record containing inputs/noise.

        This explicit projection is for ``Craft.initial_state()``-style
        records.  Ordinary :meth:`reset` remains strict so typoed state keys
        cannot disappear among unrelated model fields.
        """
        projected = self._spec.pack_projected(record)
        self.reset(self._spec.to_nested(projected), P=P)

    def checkpoint(self) -> FilterCheckpoint:
        """Capture nominal state, covariance, and logical time atomically."""
        return FilterCheckpoint(self._state["x"], self._state["P"],
                                float(self._t), self.module.artifact_id)

    def restore(self, checkpoint: FilterCheckpoint) -> None:
        """Restore a checkpoint after strict shape/finite validation.

        Restore never partially mutates the live filter: all values are
        validated and copied before any runtime field changes.
        """
        if not isinstance(checkpoint, FilterCheckpoint):
            raise TypeError("restore: expected FilterCheckpoint")
        if checkpoint.artifact_id != self.module.artifact_id:
            raise ValueError(
                "restore: checkpoint belongs to a different Module artifact")
        x = np.asarray(checkpoint.x, dtype=float)
        P = np.asarray(checkpoint.P, dtype=float)
        expected_x = (self._spec.ambient_dim,)
        expected_P = (self._spec.tangent_dim, self._spec.tangent_dim)
        if x.shape != expected_x:
            raise ValueError(
                f"restore: x shape {x.shape} doesn't match {expected_x}")
        if P.shape != expected_P:
            raise ValueError(
                f"restore: P shape {P.shape} doesn't match {expected_P}")
        t = float(checkpoint.time)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)) \
                or not np.isfinite(t):
            raise ValueError("restore: checkpoint contains non-finite values")
        if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
            raise ValueError("restore: P must be symmetric")
        if np.linalg.eigvalsh(P).min() < -_psd_roundoff_tolerance(P):
            raise ValueError("restore: P must be positive semidefinite")
        new_x, new_P = x.copy(), P.copy()
        self._state["x"], self._state["P"], self._t = new_x, new_P, t

    def set_state_keep_covariance(self, state: dict) -> None:
        """Replace the nominal state while preserving covariance and clock.

        This is intentionally separate from :meth:`reset`: retaining a
        covariance after moving its linearization point is an advanced,
        explicit operation.
        """
        self._state["x"] = self._spec.pack_any(
            state, base=self.module.state.field("x").init)

    @property
    def Q(self):
        """Default process noise for `predict` (overridden per-call by
        `predict(dt, Q=...)`; `None` uses the model's baked `L Σ Lᵀ`)."""
        return self._Q

    @Q.setter
    def Q(self, value) -> None:
        self._Q = (None if value is None else self._validate_covariance(
            value, who="Q", positive_definite=False))

    # ---- predict / update (the uniform kernel surface) -------------------

    def preintegrated_inputs(
            self, packet: Mapping[str, Any], *,
            u: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge an ``IMUPreintegrator`` readout into INS inputs.

        This is a naming/validation convenience for the NumPy runtime. The
        generated C++ filter exposes the same fields directly on ``Inputs``.
        ``packet`` is the dict returned by the recurrence's ``step`` or
        ``readouts`` method.
        """
        mapping = dict(self.module.metadata.get(
            "preintegration_input_map", {}))
        if not mapping:
            raise TypeError(
                "preintegrated_inputs requires an INS constructed with "
                "propagation='preintegrated'")
        if not isinstance(packet, Mapping):
            raise TypeError("preintegrated_inputs packet must be a mapping")
        missing = sorted(set(mapping) - set(packet))
        if missing:
            raise KeyError(
                f"preintegrated_inputs packet is missing {missing}")
        merged = dict(u or {})
        input_names = self._input_names()
        occupied = {
            resolve_suffix(key, input_names, label="input",
                           who="preintegrated_inputs")
            for key in merged
        }
        collisions = sorted(occupied & set(mapping.values()))
        if collisions:
            raise ValueError(
                "preintegrated_inputs: u also supplies packet-owned input(s) "
                f"{collisions}")
        merged.update({full: packet[short]
                       for short, full in mapping.items()})
        return merged

    def predict_preintegrated(
            self, packet: Mapping[str, Any], *, t: float | None = None,
            u: dict[str, Any] | None = None,
            Q: np.ndarray | None = None) -> None:
        """Advance a preintegrated INS by the packet's accumulated duration."""
        if "duration" not in packet:
            raise KeyError("predict_preintegrated packet is missing 'duration'")
        dt = require_positive(
            packet["duration"], name="predict_preintegrated packet duration")
        self.predict(dt, t=t, u=self.preintegrated_inputs(packet, u=u), Q=Q)

    def predict(self, dt: float, *, t: float | None = None,
                u: dict[str, Any] | None = None,
                Q: np.ndarray | None = None) -> None:
        """Advance the estimate by `dt`. Process noise: an explicit `Q`, else
        `self.Q`, else the model's baked `L Σ Lᵀ`. `u` is `{input: value}`
        (unset inputs fall to the Module's declared defaults); pass the same
        held `u` truth ran on. `t=None` uses (and advances) the runtime's
        clock, matching `NumpySim.step`; an explicit `t` overrides and
        resynchronizes it."""
        dt = require_positive(dt, name=f"{type(self).__name__}.predict dt")
        t0 = self._t if t is None else float(require_finite(
            t, name=f"{type(self).__name__}.predict t"))
        next_t = float(require_finite(
            t0 + dt, name=f"{type(self).__name__}.predict resulting time"))
        process_Q = Q if Q is not None else self._Q
        if process_Q is not None:
            process_Q = self._validate_covariance(
                process_Q, who="predict Q", positive_definite=False)
        u_vec = self.build_u(u)
        self._check_packet_duration(dt, u_vec)
        self._predict_kernel(dt, t0, u_vec, process_Q)
        self._t = next_t

    def _check_packet_duration(self, dt: float, u_vec: np.ndarray) -> None:
        """A preintegrated INS must advance by exactly the packet's span.

        The kernel itself poisons the state on a mismatch (so generated C++
        fails the same way); this names the cause before that happens.
        """
        rtol = self.module.metadata.get("preintegration_duration_rtol")
        if rtol is None:
            return
        duration_input = self.module.metadata["preintegration_input_map"][
            "duration"]
        offset = 0
        for field in self._u_fields():
            if field.name == duration_input:
                break
            offset += field.dim
        duration = float(u_vec[offset])
        if abs(dt - duration) > rtol * max(abs(dt), abs(duration)):
            raise ValueError(
                f"{type(self).__name__}.predict dt={dt!r} differs from the "
                f"preintegrated packet duration={duration!r} "
                f"({duration_input}); a packet must be consumed over exactly "
                "its own span (use predict_preintegrated)")

    def update(self, target, z=None, R=None, *, t: float | None = None,
               u: dict[str, Any] | None = None) -> UpdateResult:
        """Fold one measurement at the current state.

        * `update("gps.position", z)` — by sensor name (full or suffix),
          through the baked covariance and gate.
        * `update("gps.position", z, R=sample_R)` — typed per-sample
          covariance override through the deployable Module entry point.
        * `update(h_sym, z, R=R)` — a caller-supplied `h(x)` callable +
          measurement covariance (custom measurements; numpy-only).

        `t=None` reads the runtime's clock (which `predict` advances); a
        measurement is dt-independent, so `update` never advances it.
        """
        if callable(target):
            if z is None or R is None:
                raise TypeError("update(h_sym, z, R=...): z and R required")
            return self._update_custom(target, z, R)
        return self._fold_sensor(
            self._resolve_sensor(target), z, self.build_u(u), R=R,
            t=self._t if t is None else float(require_finite(
                t, name=f"{type(self).__name__}.update t")))

    # ---- kernels ---------------------------------------------------------

    def _predict_kernel(self, dt: float, t: float, u_vec: np.ndarray,
                        Q: np.ndarray | None) -> None:
        if Q is None:
            self._run(self.module.entry("predict"),
                      {"u": u_vec, "dt": dt, "t": t})
        else:
            self._run(self.module.entry("predict_with_Q"),
                      {"Q": np.asarray(Q, dtype=float), "u": u_vec,
                       "dt": dt, "t": t})

    def _resolve_sensor(self, name: str) -> str:
        return resolve_suffix(name, [p.name for p in self._meas_ports_ir],
                              label="sensor", who=type(self).__name__)

    def _fold_sensor(self, full: str, z, u_vec: np.ndarray, *,
                     R=None, t: float = 0.0) -> UpdateResult:
        """Fold a sensor and return the generated kernel's diagnostics."""
        port = self.module.port(full)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float)).reshape(-1)
        if z_arr.size != port.size:
            raise ValueError(
                f"update {full}: expected z of size {port.size}, got "
                f"{z_arr.size}.")
        if not np.all(np.isfinite(z_arr)):
            raise ValueError(f"update {full}: z contains non-finite values")
        ident = entry_ident(full)
        values = {full: z_arr, "u": u_vec, "t": t}
        if R is None:
            method = f"update_diagnostic_{ident}"
        else:
            R = self._validate_measurement_covariance(R, port.size, full)
            method = f"update_with_R_{ident}"
            values[f"R_{ident}"] = R
        result = self._run(self.module.entry(method), values)
        gate = self.module.metadata.get("nis_gates", {}).get(full)
        return UpdateResult(
            sensor=full,
            innovation=np.asarray(result[f"innovation_{ident}"], dtype=float)
            .reshape(-1).copy(),
            innovation_covariance=np.asarray(
                result[f"innovation_covariance_{ident}"], dtype=float).copy(),
            nis=float(np.asarray(result[f"nis_{ident}"]).reshape(-1)[0]),
            accepted=bool(
                np.asarray(result[f"accepted_{ident}"]).reshape(-1)[0]),
            gate=gate,
            covariance_overridden=R is not None,
        )

    @staticmethod
    def _validate_measurement_covariance(R, dim: int,
                                         full: str) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if R.shape != (dim, dim):
            raise ValueError(
                f"update {full}: R shape {R.shape} doesn't match "
                f"{(dim, dim)}")
        if not np.all(np.isfinite(R)):
            raise ValueError(f"update {full}: R contains non-finite values")
        if not np.allclose(R, R.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"update {full}: R must be symmetric")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"update {full}: R must be positive definite") from exc
        return R.copy()

    def _validate_covariance(self, value, *, who: str,
                             positive_definite: bool) -> np.ndarray:
        dim = self._spec.tangent_dim
        matrix = np.asarray(value)
        if matrix.dtype.kind not in "iuf":
            raise TypeError(f"{who}: covariance must be real numeric data")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (dim, dim):
            raise ValueError(
                f"{who}: shape {matrix.shape} doesn't match {(dim, dim)}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{who}: contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{who}: must be symmetric")
        eigen_min = float(np.linalg.eigvalsh(matrix).min())
        psd_tolerance = _psd_roundoff_tolerance(matrix)
        invalid = (eigen_min <= 0.0 if positive_definite
                   else eigen_min < -psd_tolerance)
        if invalid:
            relation = "positive definite" if positive_definite \
                else "positive semidefinite"
            raise ValueError(
                f"{who}: must be {relation} "
                f"(minimum eigenvalue {eigen_min:.17g}, "
                f"roundoff tolerance {psd_tolerance:.17g})"
            )
        return matrix.copy()

    def _validate_staged_state(self, state: dict[str, np.ndarray]) -> None:
        if "x" in state and not np.all(np.isfinite(state["x"])):
            raise ValueError("filter state x contains non-finite values")
        if "P" in state:
            self._validate_covariance(
                state["P"], who="filter state P", positive_definite=False)

    def _custom_h_fns(self, h_sym: Callable):
        """The `(h, H)` `ca.Function` pair for a caller-supplied `h(x)`,
        built once per callable and memoized — the symbolic construction
        (trace + jacobian + two Functions) is far too expensive to redo
        on every in-loop update."""
        try:
            cached = self._custom_h_cache.get(h_sym)
        except TypeError:                        # unhashable callable
            cached = None
        if cached is not None:
            return cached
        spec = self._spec
        x_sym = ca.MX.sym("x", spec.ambient_dim, 1)
        h_mx = h_sym(x_sym)
        h_mx = ca.reshape(h_mx, h_mx.numel(), 1)
        delta = ca.MX.sym("delta", spec.tangent_dim, 1)
        h_pert = h_sym(spec.boxplus_sym(x_sym, delta))
        h_pert = ca.reshape(h_pert, h_pert.numel(), 1)
        H_sym = ca.substitute(ca.jacobian(h_pert, delta), delta,
                              ca.MX.zeros(spec.tangent_dim, 1))
        fns = (ca.Function("h", [x_sym], [h_mx]),
               ca.Function("H", [x_sym], [H_sym]))
        try:
            self._custom_h_cache[h_sym] = fns
        except TypeError:
            pass
        return fns

    def _update_custom(self, h_sym: Callable, z, R) -> UpdateResult:
        """Joseph update for a caller-supplied `h(x)` — built on the spec's
        manifold ops; the one genuinely runtime-defined measurement."""
        spec = self._spec
        z = np.asarray(z, dtype=float).reshape(-1)
        if not np.all(np.isfinite(z)):
            raise ValueError("update custom: z contains non-finite values")
        h_fn, H_fn = self._custom_h_fns(h_sym)
        x_now = self._state["x"]
        h_x = np.asarray(h_fn(x_now)).reshape(-1)
        H = np.asarray(H_fn(x_now))
        if not np.all(np.isfinite(h_x)) or not np.all(np.isfinite(H)):
            raise ValueError("update custom: measurement model is non-finite")
        if h_x.size != z.size:
            raise ValueError(
                f"update: h(x) size {h_x.size} doesn't match z size {z.size}")
        R = self._validate_measurement_covariance(R, z.size, "custom")
        x_new, P_new, innovation, S = joseph_update_np(
            self._state["P"], H, R,
            x=x_now, z=z, h=h_x, boxplus=spec.boxplus_num)
        nis = float(innovation @ np.linalg.solve(S, innovation))
        if not np.isfinite(nis) or not np.all(np.isfinite(innovation)) \
                or not np.all(np.isfinite(S)):
            raise ValueError("update custom: update produced non-finite diagnostics")
        self._validate_staged_state({"x": x_new, "P": P_new})
        self._state["x"] = x_new
        self._state["P"] = P_new
        return UpdateResult("custom", innovation.copy(), S.copy(), nis,
                            True, None, True)
