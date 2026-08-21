"""NumpyRuntime — the generic kernel engine every view subclasses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np

from ...ir.module import Hosting, Module, Role, StateRef
from ...ir._names import resolve_suffix
from ..target import resolve_args
from ._compile import _compiled_functions, compile_functions


def _split(full: str) -> tuple[str, str]:
    owner, rest = full.split(".", 1)
    return owner, rest


def finite_array(value, *, who: str, size: int | None = None,
                 shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Own and validate real finite runtime data without coercing text/bools."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(
            f"{who}: expected real numeric data, got dtype {raw.dtype}")
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{who}: contains non-finite values")
    if size is not None and arr.size != size:
        raise ValueError(f"{who}: expected {size} value(s), got {arr.size}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{who}: expected shape {shape}, got {arr.shape}")
    return arr.copy()


def pack_fields(fields, source, *, default=0.0, required: bool = False,
                who: str = "pack") -> np.ndarray:
    """Flat vector over `fields` (in order), each filled from
    `source[field.name]` when present, validated against the field's dim.
    A missing field falls back to `default` — a constant or a callable
    `default(field)` (scalar defaults broadcast across a vector field) —
    unless `required`, which raises. The one place the numpy views pack a
    named-field dict (controls, params, noise, recurrence inputs) into the
    flat kernel vector."""
    names = {f.name for f in fields}
    unknown = sorted(set(source) - names)
    if unknown:
        raise TypeError(f"{who}: unknown field(s) {unknown}")
    chunks = []
    for f in fields:
        if f.name in source:
            v = np.atleast_1d(finite_array(
                source[f.name], who=f"{who}: {f.name!r}", size=f.dim)).ravel()
        elif required:
            raise KeyError(
                f"{who}: missing {f.name!r}; required: "
                f"{[g.name for g in fields]}")
        else:
            d = default(f) if callable(default) else default
            v = np.atleast_1d(finite_array(
                d, who=f"{who}: default for {f.name!r}")).ravel()
            if v.size == 1 and f.dim > 1:
                v = np.full(f.dim, v[0])
            elif v.size != f.dim:
                raise ValueError(
                    f"{who}: default for {f.name!r} expects dim {f.dim}, "
                    f"got {v.size}.")
        chunks.append(v)
    return np.concatenate(chunks) if chunks else np.zeros(0)


def unpack_fields(fields, vec) -> dict[str, Any]:
    """Inverse of `pack_fields`: split a flat vector over `fields` (in
    order) into `{field name: value}` — scalar fields unwrapped to float,
    vector fields as owned copies."""
    expected = sum(f.dim for f in fields)
    v = finite_array(vec, who="unpack_fields", size=expected).reshape(-1)
    out: dict[str, Any] = {}
    off = 0
    for f in fields:
        seg = v[off:off + f.dim]
        out[f.name] = float(seg[0]) if f.dim == 1 else seg.copy()
        off += f.dim
    return out


@dataclass
class _DenseEvaluationBuffer:
    """Reusable native buffers for one dense CasADi function.

    CasADi's ordinary Python call path materializes one ``DM`` object per
    output and then converts every object back to NumPy. A simulation oracle
    can expose hundreds of sensor ports from one plant tick, making that
    language-boundary bookkeeping more expensive than the compiled dynamics.
    ``FunctionBuffer`` writes those same dense outputs directly into one owned
    NumPy allocation. The runtime remains the owner of typed validation and
    scattering; only the transport across the native boundary changes.
    """

    memory: Any
    evaluate: Any
    result: np.ndarray
    offsets: np.ndarray


def _dense_evaluation_buffer(function: Any) -> _DenseEvaluationBuffer | None:
    """Build a direct buffer when every CasADi input/output is dense.

    Sparse matrices need their sparsity pattern expanded before they match a
    Module port's dense shape, so the generic ``DM`` path remains authoritative
    for them. Truth-plant ticks are dense and take this measured hot path.
    """
    if any(
        function.nnz_in(index) != function.numel_in(index)
        for index in range(function.n_in())
    ) or any(
        function.nnz_out(index) != function.numel_out(index)
        for index in range(function.n_out())
    ):
        return None
    memory, evaluate = function.buffer()
    sizes = tuple(
        int(function.numel_out(index)) for index in range(function.n_out())
    )
    offsets = np.cumsum((0, *sizes), dtype=np.int64)
    result = np.empty(int(offsets[-1]), dtype=float)
    for index, (start, end) in enumerate(pairwise(offsets)):
        memory.set_res(index, memoryview(result[start:end]))
    return _DenseEvaluationBuffer(memory, evaluate, result, offsets)


class NumpyRuntime:
    """The generic engine over a typed `Module`: state storage + the
    typed-arg gather → kernel call → scatter. Views subclass it."""

    def __init__(self, module: Module) -> None:
        self.module = module
        self._functions = module.functions      # swapped by _enable_compile()
        self._spec = module.spec
        self._state: dict[str, np.ndarray] = {}
        for f in module.state.fields:
            a = finite_array(f.init, who=f"{module.name} state {f.name!r}",
                             size=int(np.prod(f.shape)))
            # Always copy: the Module is shared, immutable IR — a view
            # here would alias every runtime built from the same
            # transform onto one array (and let a runtime mutate the
            # Module's init in place).
            self._state[f.name] = (a.reshape(f.shape).copy()
                                   if f.kind == "matrix"
                                   else a.reshape(-1).copy())

        self._u_port = module.sole_port(Role.CONTROL)
        self._noise_port = module.sole_port(Role.NOISE)
        self._meas_ports_ir = module.ports_by_role(Role.MEASUREMENT)
        self._y_port = module.sole_port(Role.OUTPUT)
        self._x_port = module.sole_port(Role.STATE)
        self._param_port = module.sole_port(Role.PARAMETER)
        self._param_overrides: dict[str, np.ndarray] = {}
        # Keyed by the actual ca.Function identity because selected functions
        # may be replaced by compiled externals after construction.
        self._evaluation_buffers: dict[int, _DenseEvaluationBuffer | None] = {}

        self._t = 0.0

    def _enable_compile(self) -> "NumpyRuntime":
        """Require `cc`-compiled externals (bit-identical, ~8x per call)."""
        self._functions = _compiled_functions(dict(self.module.functions))
        return self

    def compile_functions(
        self,
        function_names: Iterable[str],
        *,
        optimization: Literal["startup", "balanced", "runtime"] = "balanced",
    ) -> "NumpyRuntime":
        """Compile a selected hot subset of this runtime's kernels.

        Large transforms need not make native execution all-or-nothing. A
        rate loop can compile its dominant kernel while retaining interpreted
        entry points that execute rarely. Selection is by stable Module
        function identity, and a failure leaves the runtime unchanged.
        """
        names = tuple(dict.fromkeys(function_names))
        if not names:
            raise ValueError("compile_functions requires at least one function name")
        unknown = sorted(set(names) - set(self.module.functions))
        if unknown:
            raise KeyError(f"unknown Module function(s) {unknown}")
        selected = {name: self.module.functions[name] for name in names}
        compiled = compile_functions(selected, optimization=optimization)
        self._functions = {**self._functions, **compiled}
        return self

    # ---- kernel engine (typed-arg gather → call → scatter) -----------

    def call(self, method: str, values: dict[str, Any] | None = None,
             **kw) -> dict[str, np.ndarray]:
        """Run one entry point. `values`/kwargs are keyed by port or state
        name (use the dict for dotted names); TIME ports default to 0.

        The Hosting contract: a THREADED module's state is the caller's —
        supply state fields by name in `values` (unsupplied ones fall back
        to the engine's last-written copy) and read the fresh writes from
        the returned dict alongside the entry's returns. A HELD module's
        state lives in the runtime and is read/written in place; only the
        entry's returns come back."""
        vals = dict(values or {})
        vals.update(kw)
        return self._run(self.module.entry(method), vals)

    def _run(self, ep, values: dict[str, Any]) -> dict[str, np.ndarray]:
        m = self.module
        args = resolve_args(m, ep, values,
                            state_lookup=lambda n: self._state[n],
                            param_default=self.param_vector)
        checked_args = []
        for ref, arg in zip(ep.args, args):
            if isinstance(ref, StateRef):
                expected_size = int(np.prod(m.state.field(ref.name).shape))
            else:
                expected_size = m.port(ref.name).size
            checked_args.append(finite_array(
                arg, who=f"{m.name}.{ep.method} argument {ref.name!r}",
                size=expected_size))
        fn = self._functions[ep.fn]
        buffer_key = id(fn)
        if buffer_key not in self._evaluation_buffers:
            self._evaluation_buffers[buffer_key] = _dense_evaluation_buffer(fn)
        evaluation = self._evaluation_buffers[buffer_key]
        buffered = evaluation is not None
        if evaluation is None:
            res = fn(*checked_args)
            outs = [res] if fn.n_out() == 1 else list(res)
        else:
            # FunctionBuffer consumes column-major flat storage. Keeping these
            # owned arrays alive through evaluate() also keeps every registered
            # native pointer valid for the duration of the call.
            input_buffers = [
                np.asarray(arg, dtype=float).reshape(-1, order="F").copy()
                for arg in checked_args
            ]
            for index, arg in enumerate(input_buffers):
                evaluation.memory.set_arg(index, memoryview(arg))
            evaluation.evaluate()
            if evaluation.memory.ret() != 0:
                raise RuntimeError(
                    f"{m.name}.{ep.method}: kernel {ep.fn!r} returned "
                    f"{evaluation.memory.ret()}"
                )
            if not np.all(np.isfinite(evaluation.result)):
                raise ValueError(
                    f"{m.name}.{ep.method}: kernel {ep.fn!r} produced "
                    "non-finite values"
                )
            outs = [
                evaluation.result[start:end]
                for start, end in pairwise(evaluation.offsets)
            ]
        expected = len(ep.writes) + len(ep.returns)
        if len(outs) != expected:
            raise RuntimeError(
                f"{m.name}.{ep.method}: kernel {ep.fn!r} produced "
                f"{len(outs)} outputs but the entry point declares "
                f"{len(ep.writes)} writes + {len(ep.returns)} returns "
                f"= {expected}.")
        threaded = m.hosting is Hosting.THREADED
        staged_state: dict[str, np.ndarray] = {}
        ret: dict[str, np.ndarray] = {}
        for i, w in enumerate(ep.writes):
            fld = m.state.field(w)
            arr = (
                np.asarray(outs[i], dtype=float).copy()
                if buffered
                else finite_array(
                    outs[i], who=f"{m.name}.{ep.method} state result {w!r}",
                    size=int(np.prod(fld.shape)))
            )
            val = (arr.reshape(fld.shape, order="F") if fld.kind == "matrix"
                   else arr.reshape(-1))
            staged_state[w] = val
            if threaded:                  # THREADED: writes go to the caller
                ret[w] = val
        for name, o in zip(ep.returns, outs[len(ep.writes):]):
            port = m.port(name)
            a = (
                np.asarray(o, dtype=float).copy()
                if buffered
                else finite_array(
                    o, who=f"{m.name}.{ep.method} return {name!r}",
                    size=port.size)
            )
            ret[name] = (a.reshape(port.shape, order="F") if len(port.shape) == 2
                         else a.reshape(-1))
        self._validate_staged_state(staged_state)
        self._state.update(staged_state)
        return ret

    def _validate_staged_state(self, state: dict[str, np.ndarray]) -> None:
        """View-specific invariants checked before any live state changes."""

    # ---- shared metadata helpers (all from the Module) ----------------

    def _u_fields(self):
        return self._u_port.fields if self._u_port is not None else ()

    def _input_names(self) -> list[str]:
        return [f.name for f in self._u_fields()]

    def build_u(self, u: dict[str, Any] | None) -> np.ndarray:
        """Resolve a `{name: value}` dict (full or suffix names) to the
        flat control vector over the Module's declared defaults."""
        names = self._input_names()
        source = {resolve_suffix(k, names, label="input",
                                 who=type(self).__name__): v
                  for k, v in (u or {}).items()}
        return pack_fields(self._u_fields(), source,
                           default=lambda f: f.default, who="build_u")

    # ---- promoted parameters (PARAMETER port) --------------------------

    def set_parameters(self, values: dict[str, Any]) -> None:
        """Override promoted-parameter values (full or suffix names);
        every subsequent kernel call uses them. Values not overridden
        stay at the Module's declared defaults."""
        if self._param_port is None:
            raise ValueError(
                f"{self.module.name}: module declares no parameter port — "
                f"build the transform with parameters=[...] to promote "
                f"tunable Parameters.")
        names = [f.name for f in self._param_port.fields]
        dims = {f.name: f.dim for f in self._param_port.fields}
        staged = dict(self._param_overrides)
        for k, v in values.items():
            full = resolve_suffix(k, names, label="parameter",
                                  who=type(self).__name__)
            arr = finite_array(v, who=f"set_parameters: {full!r}",
                               size=dims[full]).ravel()
            staged[full] = arr
        self._param_overrides = staged

    def param_vector(self) -> np.ndarray:
        """The flat promoted-parameter vector: declared defaults merged
        with `set_parameters` overrides, in port-field order."""
        port = self._param_port
        if port is None:
            return np.zeros(0)
        return pack_fields(port.fields, self._param_overrides,
                           default=lambda f: f.default, who="param_vector")

    @property
    def spec(self):
        return self._spec

    @property
    def input_names(self) -> list[str]:
        """The Module's declared control-input names, in order."""
        return self._input_names()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} over {self.module!r}>"
