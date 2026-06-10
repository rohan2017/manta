"""NumpyRuntime — the generic kernel engine every view subclasses."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...bus import PortSet
from ...ir.module import Module, PortRef, Role, StateRef
from ...linearization import resolve_suffix
from ._compile import _compiled_functions


def _split(full: str) -> tuple[str, str]:
    owner, rest = full.split(".", 1)
    return owner, rest


class NumpyRuntime:
    """The generic engine over a typed `Module`: state storage + the
    typed-arg gather → kernel call → scatter. Views subclass it."""

    def __init__(self, module: Module) -> None:
        self.module = module
        self._functions = module.functions      # swapped by _enable_compile()
        self._spec = module.spec
        self._state: dict[str, np.ndarray] = {}
        for f in module.state.fields:
            a = np.asarray(f.init, dtype=float)
            self._state[f.name] = (a.reshape(f.shape) if f.kind == "matrix"
                                   else a.reshape(-1).copy())

        ctrl = module.ports_by_role(Role.CONTROL)
        self._u_port = ctrl[0] if ctrl else None
        noi = module.ports_by_role(Role.NOISE)
        self._noise_port = noi[0] if noi else None
        self._meas_ports_ir = module.ports_by_role(Role.MEASUREMENT)
        out = module.ports_by_role(Role.OUTPUT)
        self._y_port = out[0] if out else None
        st = module.ports_by_role(Role.STATE)
        self._x_port = st[0] if st else None
        par = module.ports_by_role(Role.PARAMETER)
        self._param_port = par[0] if par else None
        self._param_overrides: dict[str, np.ndarray] = {}
        self._methods = {e.method for e in module.entry_points}

        self._t = 0.0
        self._ports = PortSet()

    def _enable_compile(self) -> "NumpyRuntime":
        """Replace the interpreted CasADi functions with `cc`-compiled
        externals (bit-identical, ~8x per call). Idempotent."""
        self._functions = _compiled_functions(dict(self.module.functions))
        return self

    # ---- kernel engine (typed-arg gather → call → scatter) -----------

    def call(self, method: str, values: dict[str, Any] | None = None,
             **kw) -> dict[str, np.ndarray]:
        """Run one entry point. `values`/kwargs are keyed by port name
        (use the dict for dotted names); TIME ports default to 0."""
        vals = dict(values or {})
        vals.update(kw)
        return self._run(self.module.entry(method), vals)

    def _run(self, ep, values: dict[str, Any]) -> dict[str, np.ndarray]:
        m = self.module
        args = []
        for a in ep.args:
            if isinstance(a, StateRef):
                args.append(self._state[a.name])
            elif isinstance(a, PortRef):
                if a.name in values:
                    args.append(values[a.name])
                elif m.port(a.name).role is Role.TIME:
                    args.append(0.0)
                elif m.port(a.name).role is Role.PARAMETER:
                    # Declared values, with any set_parameters overrides.
                    args.append(self.param_vector())
                else:
                    raise KeyError(
                        f"{m.name}.{ep.method}: missing value for port "
                        f"{a.name!r}.")
            else:                                  # pragma: no cover
                raise TypeError(f"unknown kernel arg {a!r}")
        fn = self._functions[ep.fn]
        res = fn(*args)
        outs = [res] if fn.n_out() == 1 else list(res)
        expected = len(ep.writes) + len(ep.returns)
        if len(outs) != expected:
            raise RuntimeError(
                f"{m.name}.{ep.method}: kernel {ep.fn!r} produced "
                f"{len(outs)} outputs but the entry point declares "
                f"{len(ep.writes)} writes + {len(ep.returns)} returns "
                f"= {expected}.")
        for i, w in enumerate(ep.writes):
            fld = m.state.field(w)
            arr = np.asarray(outs[i], dtype=float)
            self._state[w] = (arr.reshape(fld.shape) if fld.kind == "matrix"
                              else arr.reshape(-1))
        ret: dict[str, np.ndarray] = {}
        for name, o in zip(ep.returns, outs[len(ep.writes):]):
            port = m.port(name)
            a = np.asarray(o, dtype=float)
            ret[name] = (a.reshape(port.shape) if len(port.shape) == 2
                         else a.reshape(-1))
        return ret

    # ---- shared metadata helpers (all from the Module) ----------------

    def _u_fields(self):
        return self._u_port.fields if self._u_port is not None else ()

    def _input_names(self) -> list[str]:
        return [f.name for f in self._u_fields()]

    def build_u(self, u: dict[str, Any] | None) -> np.ndarray:
        """Resolve a `{name: value}` dict (full or suffix names) to the
        flat control vector over the Module's declared defaults."""
        fields = self._u_fields()
        out = np.array([float(np.asarray(f.default).ravel()[0])
                        for f in fields])
        if u:
            names = self._input_names()
            idx = {n: i for i, n in enumerate(names)}
            for k, v in u.items():
                full = resolve_suffix(k, names, label="input",
                                      who=type(self).__name__)
                out[idx[full]] = float(np.asarray(v).ravel()[0])
        return out

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
        for k, v in values.items():
            full = resolve_suffix(k, names, label="parameter",
                                  who=type(self).__name__)
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size != dims[full]:
                raise ValueError(
                    f"set_parameters: {full!r} expects {dims[full]} "
                    f"value(s), got {arr.size}.")
            self._param_overrides[full] = arr

    def param_vector(self) -> np.ndarray:
        """The flat promoted-parameter vector: declared defaults merged
        with `set_parameters` overrides, in port-field order."""
        port = self._param_port
        if port is None:
            return np.zeros(0)
        chunks = [self._param_overrides.get(
                      f.name, np.asarray(f.default, dtype=float).ravel())
                  for f in port.fields]
        return np.concatenate(chunks) if chunks else np.zeros(0)

    @property
    def spec(self):
        return self._spec

    @property
    def input_names(self) -> list[str]:
        """The Module's declared control-input names, in order."""
        return self._input_names()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} over {self.module!r}>"
