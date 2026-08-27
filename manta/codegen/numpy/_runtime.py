"""NumpyRuntime — the generic kernel engine every view subclasses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ...ir._names import resolve_suffix
from ...ir.module import Hosting, Module, ModuleKind, Role, StateRef
from ..target import resolve_args
from ._compile import (
    DEFAULT_COMPILATION_TIMEOUT_S,
    DEFAULT_MAX_INSTRUCTIONS,
    Optimization,
    _compiled_functions,
    compile_functions,
    validate_max_instructions,
)


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


def _real_float_vector(value, *, who: str) -> np.ndarray:
    """`finite_array` minus the finiteness pass: the same dtype rejection,
    returning a flat float view/copy for callers that check finiteness once
    over a whole packed vector."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(
            f"{who}: expected real numeric data, got dtype {raw.dtype}")
    arr = raw if raw.dtype == _FLOAT else raw.astype(float)
    return arr.reshape(-1)


_FLOAT = np.dtype(float)


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
            v = _real_float_vector(source[f.name], who=f"{who}: {f.name!r}")
            if v.size != f.dim:
                raise ValueError(
                    f"{who}: {f.name!r}: expected {f.dim} value(s), got {v.size}")
        elif required:
            raise KeyError(
                f"{who}: missing {f.name!r}; required: "
                f"{[g.name for g in fields]}")
        else:
            d = default(f) if callable(default) else default
            v = _real_float_vector(d, who=f"{who}: default for {f.name!r}")
            if v.size == 1 and f.dim > 1:
                v = np.full(f.dim, v[0])
            elif v.size != f.dim:
                raise ValueError(
                    f"{who}: default for {f.name!r} expects dim {f.dim}, "
                    f"got {v.size}.")
        chunks.append(v)
    packed = np.concatenate(chunks) if chunks else np.zeros(0)
    if not np.isfinite(packed).all():
        # Re-validate per field so the error names the offending field
        # exactly as before; the fast path only skipped this on success.
        for f in fields:
            if f.name in source:
                finite_array(source[f.name], who=f"{who}: {f.name!r}", size=f.dim)
            else:
                d = default(f) if callable(default) else default
                finite_array(d, who=f"{who}: default for {f.name!r}")
        raise ValueError(f"{who}: contains non-finite values")
    return packed


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


@dataclass(frozen=True)
class _EntryPlan:
    """Argument sizes and output layouts of one entry point, resolved once."""

    args: tuple[tuple[str, int], ...]
    writes: tuple[tuple[str, int, tuple[int, ...] | None], ...]
    returns: tuple[tuple[str, int, tuple[int, ...] | None], ...]


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
        self._param_vector_cache: np.ndarray | None = None
        # Keyed by the actual ca.Function identity because selected functions
        # may be replaced by compiled externals after construction.
        self._evaluation_buffers: dict[int, _DenseEvaluationBuffer | None] = {}
        # Per entry point: argument sizes and output layouts resolved once.
        # The Module is immutable, so this never goes stale; resolving it
        # per call was most of a simulator tick for wide truth plants.
        self._entry_plans: dict[str, _EntryPlan] = {}
        self._input_name_list = [f.name for f in self._u_fields()]

        self._t = 0.0

    def _enable_compile(
        self,
        *,
        optimization: Optimization | None = None,
        timeout_s: float | None = DEFAULT_COMPILATION_TIMEOUT_S,
        max_instructions: int | None = DEFAULT_MAX_INSTRUCTIONS,
    ) -> NumpyRuntime:
        """Require optimized ``cc`` externals for every kernel.

        Full simulator translation units can contain several vehicles and
        hundreds of output channels. Use O1 by default for those truth graphs,
        while allowing their caller to choose O0/O1/O2; smaller runtime models
        receive the target-native runtime profile. ``max_instructions`` is
        the owner-declared size gate (``None`` disables it).
        """
        selected: Optimization = optimization or (
            "O1" if self.module.kind is ModuleKind.SIMULATOR else "runtime"
        )
        self._functions = _compiled_functions(
            dict(self.module.functions),
            max_instr=validate_max_instructions(max_instructions),
            optimization=selected,
            timeout_s=timeout_s,
        )
        return self

    def compile_functions(
        self,
        function_names: Iterable[str],
        *,
        optimization: Optimization = "balanced",
        timeout_s: float = DEFAULT_COMPILATION_TIMEOUT_S,
        max_instructions: int | None = DEFAULT_MAX_INSTRUCTIONS,
    ) -> NumpyRuntime:
        """Compile a selected hot subset of this runtime's kernels.

        Large transforms need not make native execution all-or-nothing. A
        rate loop can compile its dominant kernel while retaining interpreted
        entry points that execute rarely. Selection is by stable Module
        function identity, and a failure leaves the runtime unchanged.
        ``max_instructions`` raises or disables (``None``) the size gate.
        """
        names = tuple(dict.fromkeys(function_names))
        if not names:
            raise ValueError("compile_functions requires at least one function name")
        unknown = sorted(set(names) - set(self.module.functions))
        if unknown:
            raise KeyError(f"unknown Module function(s) {unknown}")
        selected = {name: self.module.functions[name] for name in names}
        compiled = compile_functions(
            selected, max_instructions=max_instructions,
            optimization=optimization, timeout_s=timeout_s,
        )
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

    def _entry_plan(self, ep) -> _EntryPlan:
        plan = self._entry_plans.get(ep.method)
        if plan is None:
            m = self.module
            args = []
            for ref in ep.args:
                if isinstance(ref, StateRef):
                    size = int(np.prod(m.state.field(ref.name).shape))
                else:
                    size = m.port(ref.name).size
                args.append((ref.name, size))
            writes = []
            for w in ep.writes:
                fld = m.state.field(w)
                writes.append((w, int(np.prod(fld.shape)),
                               fld.shape if fld.kind == "matrix" else None))
            returns = []
            for name in ep.returns:
                port = m.port(name)
                returns.append((name, port.size,
                                port.shape if len(port.shape) == 2 else None))
            plan = _EntryPlan(tuple(args), tuple(writes), tuple(returns))
            self._entry_plans[ep.method] = plan
        return plan

    def _run(self, ep, values: dict[str, Any]) -> dict[str, np.ndarray]:
        m = self.module
        plan = self._entry_plan(ep)
        args = resolve_args(m, ep, values,
                            state_lookup=lambda n: self._state[n],
                            param_default=self.param_vector)
        fn = self._functions[ep.fn]
        buffer_key = id(fn)
        if buffer_key not in self._evaluation_buffers:
            self._evaluation_buffers[buffer_key] = _dense_evaluation_buffer(fn)
        evaluation = self._evaluation_buffers[buffer_key]
        buffered = evaluation is not None
        who = f"{m.name}.{ep.method}"
        if evaluation is None:
            checked_args = [
                finite_array(arg, who=f"{who} argument {name!r}", size=size)
                for (name, size), arg in zip(plan.args, args)
            ]
            res = fn(*checked_args)
            outs = [res] if fn.n_out() == 1 else list(res)
        else:
            # FunctionBuffer consumes column-major flat storage. One owned
            # float view per argument: validated, flattened column-major,
            # and kept alive through evaluate() so every registered native
            # pointer stays valid for the duration of the call.
            input_buffers = []
            for (name, size), arg in zip(plan.args, args):
                raw = np.asarray(arg)
                if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
                    raise TypeError(
                        f"{who} argument {name!r}: expected real numeric "
                        f"data, got dtype {raw.dtype}")
                col = np.asarray(raw, dtype=float).reshape(-1, order="F")
                if col.size != size:
                    raise ValueError(
                        f"{who} argument {name!r}: expected {size} value(s), "
                        f"got {col.size}")
                if not np.isfinite(col).all():
                    raise ValueError(
                        f"{who} argument {name!r}: contains non-finite values")
                if not col.flags.c_contiguous:
                    col = np.ascontiguousarray(col)
                input_buffers.append(col)
            for index, arg in enumerate(input_buffers):
                evaluation.memory.set_arg(index, memoryview(arg))
            evaluation.evaluate()
            if evaluation.memory.ret() != 0:
                raise RuntimeError(
                    f"{who}: kernel {ep.fn!r} returned "
                    f"{evaluation.memory.ret()}"
                )
            if not np.isfinite(evaluation.result).all():
                raise ValueError(
                    f"{who}: kernel {ep.fn!r} produced non-finite values"
                )
            # One owned allocation for this call's outputs; the buffer is
            # reused by the next evaluate(), so results must not alias it.
            owned = evaluation.result.copy()
            outs = [
                owned[start:end]
                for start, end in pairwise(evaluation.offsets)
            ]
        expected = len(ep.writes) + len(ep.returns)
        if len(outs) != expected:
            raise RuntimeError(
                f"{who}: kernel {ep.fn!r} produced "
                f"{len(outs)} outputs but the entry point declares "
                f"{len(ep.writes)} writes + {len(ep.returns)} returns "
                f"= {expected}.")
        threaded = m.hosting is Hosting.THREADED
        staged_state: dict[str, np.ndarray] = {}
        ret: dict[str, np.ndarray] = {}
        for i, (w, size, shape) in enumerate(plan.writes):
            arr = (
                outs[i]
                if buffered
                else finite_array(
                    outs[i], who=f"{who} state result {w!r}", size=size)
            )
            val = (arr.reshape(shape, order="F") if shape is not None
                   else arr.reshape(-1))
            staged_state[w] = val
            if threaded:                  # THREADED: writes go to the caller
                ret[w] = val
        for (name, size, shape), o in zip(plan.returns, outs[len(ep.writes):]):
            a = (
                o
                if buffered
                else finite_array(o, who=f"{who} return {name!r}", size=size)
            )
            ret[name] = (a.reshape(shape, order="F") if shape is not None
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
        return list(self._input_name_list)

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
        self._param_vector_cache = None

    def param_vector(self) -> np.ndarray:
        """The flat promoted-parameter vector: declared defaults merged
        with `set_parameters` overrides, in port-field order."""
        port = self._param_port
        if port is None:
            return np.zeros(0)
        cached = self._param_vector_cache
        if cached is None:
            cached = pack_fields(port.fields, self._param_overrides,
                                 default=lambda f: f.default,
                                 who="param_vector")
            self._param_vector_cache = cached
        # Callers may mutate the vector they receive; the cache stays pristine.
        return cached.copy()

    @property
    def spec(self):
        return self._spec

    @property
    def input_names(self) -> list[str]:
        """The Module's declared control-input names, in order."""
        return self._input_names()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} over {self.module!r}>"
