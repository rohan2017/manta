"""NumpyModule — the ONE generic native-Python lowering of a `Module`.

`NumpyModule(module)` allocates storage for the Module's State and exposes
one method per `EntryPoint`. Each method gathers the kernel's positional
arguments (from State fields + the supplied Ports), evaluates the baked
`ca.Function`, then scatters its outputs back into State (`writes_state`)
and returns the rest (`returns`). It knows nothing about "Sim", "EKF",
"Kalman", or "PID" — it just runs the Module. This is the numpy half of the
generic backend contract (the other half, "run a `ca.Function`", is trivial
in numpy: a `ca.Function` is directly callable).

The ergonomic façades (nested-dict translation, the signal bus, the noise
driver, name resolution) are deliberately *not* here — they layer over any
Module and are kept above this runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _init_value(field) -> np.ndarray:
    """Allocate a State field's initial value from its `init`."""
    if field.init is not None:
        a = np.asarray(field.init, dtype=float)
        return a.reshape(field.shape) if field.kind == "matrix" else a.reshape(-1)
    if field.kind == "matrix":
        return np.zeros(field.shape, dtype=float)
    return np.zeros(field.size, dtype=float)


class NumpyModule:
    """Generic runtime over a `Module`: State dict + an entry-point method
    each. Produced by `TargetNumpy`'s `lower_module`."""

    def __init__(self, module) -> None:
        self.module = module
        self.state: dict[str, np.ndarray] = {
            f.name: _init_value(f) for f in module.state.fields}
        self._fields = set(self.state)
        for ep in module.entry_points:
            setattr(self, ep.method, self._bind(ep))

    # ---- entry-point dispatch -----------------------------------------

    def _bind(self, ep):
        def method(**ports) -> dict[str, np.ndarray]:
            return self._run(ep, ports)
        method.__name__ = ep.method
        method.__qualname__ = f"NumpyModule.{ep.method}"
        return method

    def _run(self, ep, ports: dict[str, Any]) -> dict[str, np.ndarray]:
        args = []
        for name in ep.kernel_args:
            if name in self._fields:
                args.append(self.state[name])
            elif name in ports:
                args.append(ports[name])
            elif name in ("dt", "t"):
                args.append(0.0)        # measurement kernels: dt/t default 0
            else:
                raise KeyError(
                    f"NumpyModule.{ep.method}: missing value for kernel arg "
                    f"{name!r}; supply it as a keyword port.")
        fn = self.module.functions[ep.fn]
        res = fn(*args)
        outs = [res] if fn.n_out() == 1 else list(res)
        for i, fname in enumerate(ep.writes_state):
            self.state[fname] = self._shape_field(outs[i], fname)
        rest = outs[len(ep.writes_state):]
        return {name: np.asarray(r, dtype=float).reshape(-1)
                for name, r in zip(ep.returns, rest)}

    def _shape_field(self, val, fname: str) -> np.ndarray:
        a = np.asarray(val, dtype=float)
        field = self.module.state.field(fname)
        return a.reshape(field.shape) if field.kind == "matrix" else a.reshape(-1)

    def reset(self) -> None:
        """Reset every State field to its Module-declared default."""
        for f in self.module.state.fields:
            self.state[f.name] = _init_value(f)

    def __repr__(self) -> str:
        return f"<NumpyModule over {self.module!r}>"
