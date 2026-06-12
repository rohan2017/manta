"""Graph — the user-facing entry point for building an IR.

A `Graph` is a context-manager-scoped collection of named symbolic inputs and
named symbolic outputs. Inside the `with Graph() as g:` block, factory
methods like `Vec3[F].input("name")` register placeholders that participate
in subsequent IR expressions. `g.output(value, name="...")` records what to
return.

The graph itself is a thin layer over CasADi — when you call `g.compile()`,
the collected inputs and outputs become arguments to `casadi.Function`. The
compiled function is callable from Python directly (the "Python backend" —
no separate runtime to install).
"""

from __future__ import annotations

import threading
from typing import Any

import casadi as ca

# Forward refs to avoid circular imports — types.py imports from graph.py.
# (Resolved at runtime.)


_local = threading.local()


def _current_graph() -> "Graph":
    g = getattr(_local, "graph", None)
    if g is None:
        raise RuntimeError(
            "No active Graph. Wrap IR-building code in 'with manta.ir.Graph() "
            "as g:' before calling .input() or constructing constants.")
    return g


class Graph:
    """The container for an IR being assembled.

    Usage::

        with Graph() as g:
            x = Vec3[F].input("x")
            ...
            g.output(y, "y")

        fn = g.compile()     # returns a CompiledGraph; call it as fn(x=...)

    Holds insertion-ordered dicts of named inputs and named outputs. The
    underlying symbolic graph lives inside the IR values themselves (each one
    wraps a `casadi.MX` node); the Graph is a registry, not the math itself.
    """

    def __init__(self, name: str = "graph") -> None:
        self._name = name
        self._inputs: dict[str, Any]  = {}   # name → IR value
        self._outputs: dict[str, Any] = {}   # name → IR value
        self._saved_active: "Graph | None" = None

    # ----- Context manager -------------------------------------------------

    def __enter__(self) -> "Graph":
        self._saved_active = getattr(_local, "graph", None)
        _local.graph = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _local.graph = self._saved_active
        self._saved_active = None

    # ----- Internal registration (called by types.input/etc.) --------------

    def _register_input(self, name: str, value) -> None:
        if name in self._inputs:
            raise ValueError(
                f"Graph '{self._name}': duplicate input name '{name}'.")
        self._inputs[name] = value

    def output(self, value, name: str) -> None:
        """Record `value` as a named output of this graph.

        It is intentionally allowed for a name to appear as BOTH an input
        and an output — that's the natural pattern for state-update
        functions where the input is the current state and the output is
        the new state at the same slot.
        """
        from .types import _IRValue   # local import to break the cycle
        if not isinstance(value, _IRValue):
            raise TypeError(
                f"Graph.output: expected an IR value, got {type(value).__name__}")
        if name in self._outputs:
            raise ValueError(
                f"Graph '{self._name}': duplicate output name '{name}'.")
        self._outputs[name] = value

    # ----- Introspection ---------------------------------------------------

    @property
    def inputs(self) -> dict[str, Any]:
        return dict(self._inputs)

    @property
    def outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def summary(self) -> str:
        from .debug import render_summary
        return render_summary(self)

    # ----- Compilation -----------------------------------------------------

    def compile(self,
                defaults: dict[str, Any] | None = None) -> "CompiledGraph":
        """Build a `casadi.Function` from this graph's inputs and outputs.

        `defaults` — optional `{input_name: default_value}` map used by
        the compiled wrapper when the caller omits an input. Lets
        ergonomic inputs (like `t` for time) stay optional in the
        majority of tests.
        """
        if not self._outputs:
            raise ValueError(
                f"Graph '{self._name}': no outputs registered; "
                f"call g.output(value, 'name') before compiling.")
        in_names  = list(self._inputs.keys())
        out_names = list(self._outputs.keys())
        in_mxs    = [v._mx for v in self._inputs.values()]
        out_mxs   = [v._mx for v in self._outputs.values()]
        # Allow input/output names to overlap — that's the natural pattern
        # for state-update tick functions (current state vs. new state at
        # the same slot name).
        opts = {"allow_duplicate_io_names": True}
        fn = ca.Function(self._name, in_mxs, out_mxs,
                         in_names, out_names, opts)
        return CompiledGraph(fn, dict(self._inputs), dict(self._outputs),
                             defaults=defaults)

    def jacobian(self, of: str, wrt: str) -> "CompiledGraph":
        """Return a CompiledGraph computing d(output `of`) / d(input `wrt`).

        The Jacobian is symbolic — CasADi differentiates the expression tree
        directly. Sparsity is tracked automatically.
        """
        if of not in self._outputs:
            raise KeyError(
                f"Graph.jacobian: '{of}' is not an output of this graph. "
                f"Available outputs: {list(self._outputs)}")
        if wrt not in self._inputs:
            raise KeyError(
                f"Graph.jacobian: '{wrt}' is not an input of this graph. "
                f"Available inputs: {list(self._inputs)}")
        out_val = self._outputs[of]
        in_val  = self._inputs[wrt]
        J_mx = ca.jacobian(out_val._mx, in_val._mx)

        # Wrap the Jacobian as a callable Function that takes ALL of this
        # graph's inputs (since J may depend on any of them) and returns the
        # Jacobian matrix.
        in_names = list(self._inputs.keys())
        in_mxs   = [v._mx for v in self._inputs.values()]
        fn = ca.Function(
            f"{self._name}_jac_{of}_wrt_{wrt}",
            in_mxs, [J_mx], in_names, ["jacobian"])
        # The Jacobian doesn't fit the (name → IR-value) shape exactly — its
        # output is a 2-D matrix without frame tags. Return a thin caller
        # that just passes through to the casadi.Function.
        return CompiledGraph(fn, dict(self._inputs),
                             {"jacobian": _RawMatrix(J_mx)})


class _RawMatrix:
    """Internal: a placeholder for output entries that aren't IR-typed
    values (e.g., a Jacobian matrix). Carries just the MX so CompiledGraph
    can render it in summary output without needing a frame tag."""
    def __init__(self, mx):
        self._mx = mx

    @property
    def shape(self):
        return self._mx.shape


class CompiledGraph:
    """A callable wrapper around `casadi.Function`.

    Call with keyword args by input name; receive a dict of outputs by name.
    Underlying CasADi Function is exposed as `.casadi_function` for users who
    want to drop down (e.g., for codegen, batching, or to compose with other
    CasADi functions).
    """

    def __init__(self, fn: ca.Function, input_specs: dict[str, Any],
                 output_specs: dict[str, Any],
                 defaults: dict[str, Any] | None = None) -> None:
        self.casadi_function = fn
        self._input_specs  = input_specs
        self._output_specs = output_specs
        # Output-only names (i.e. graph outputs that aren't also inputs).
        # These appear in tick(...) results but the next-tick state is
        # expected to carry them via the `{**state, **out}` merge pattern;
        # passing them back into __call__ is benign and silently dropped.
        # Other unknown keys still raise — typo protection.
        self._output_only = set(output_specs) - set(input_specs)
        # Optional per-input default values. Filled in by __call__ when the
        # caller omits the key. Used for ergonomic inputs like `t` that
        # most callers don't bother to thread but a few (planet rotation)
        # need.
        self._defaults: dict[str, Any] = dict(defaults or {})

    # ----- Call -----------------------------------------------------------

    def __call__(self, **kwargs):
        # Drop output-only keys carried via the {**state, **out} merge.
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in self._output_only}
        # Fill in defaults for any expected inputs the caller didn't supply.
        for name, default in self._defaults.items():
            kwargs.setdefault(name, default)
        # Validate input set.
        provided = set(kwargs)
        expected = set(self._input_specs)
        missing  = expected - provided
        extra    = provided - expected
        if missing:
            raise ValueError(
                f"CompiledGraph.__call__: missing input(s): {sorted(missing)}")
        if extra:
            raise ValueError(
                f"CompiledGraph.__call__: unknown input(s): {sorted(extra)}")

        # CasADi accepts numpy arrays, lists, or scalars; positional or
        # keyword. We always go keyword for stability.
        result = self.casadi_function(**{k: kwargs[k] for k in self._input_specs})

        # CasADi returns a dict when the function has named outputs; coerce
        # to plain numpy arrays for ergonomic downstream use.
        out = {}
        for name in self._output_specs:
            val = result[name]
            out[name] = _to_numpy(val)
        return out

    def __repr__(self) -> str:
        inputs  = ", ".join(self._input_specs)
        outputs = ", ".join(self._output_specs)
        return (f"<CompiledGraph {self.casadi_function.name()}: "
                f"({inputs}) → ({outputs})>")


def _to_numpy(val):
    """Numpy-ergonomic shape coercion for a CasADi DM result.

    Conversions:
      (1, 1)  → Python float            (scalar)
      (n, 1)  → 1-D ndarray of length n (column vector)
      other   → 2-D ndarray             (matrices etc.)
    """
    import numpy as np
    arr = np.array(val)
    if arr.shape == (1, 1):
        return float(arr[0, 0])
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(arr.shape[0])
    return arr
