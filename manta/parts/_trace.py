"""Compile-trace machinery — how declared attributes read symbols.

This is compiler infrastructure, not part-author surface: the world-tick
compiler opens a `TraceBindings` context, binds each part/disturbance's
declared State / Input / Noise attributes (and promoted Parameters) to
the current tick's symbolic nodes, and `DeclarationHost.__getattribute__`
(`manta.parts.base`) consults the active trace on every attribute read —
the instances themselves are never mutated.

Consumers: `tick/world_tick.py` (opens the trace), `fields/base.py`
(disturbance reads mid-trace via `active_trace`), and the compile-time
numeric/symbolic branch points throughout `tick/` (`is_promoted`,
`declared_attr`).
"""

from __future__ import annotations

import threading
from typing import Any


_trace_local = threading.local()


class TraceBindings:
    """Per-compile attribute bindings (a thread-local context manager).

    While active, a `DeclarationHost` attribute read resolves through these
    bindings instead of the instance: `self.throttle` inside `update()`
    returns the current tick's symbolic node, and the part/disturbance
    instance is NEVER mutated. Exception-safe by construction — the
    bindings live here, not on the objects, so there is nothing to restore
    when a trace unwinds (normally or via an exception).

    Craft rigid-body state symbols live here too (`bind_craft_state` /
    `craft_sym_state`), so a disturbance anchored to a craft (e.g.
    CraftWindBubble) can read that craft's symbolic position mid-trace
    without the compiler ever stashing anything on the Craft instance.
    """

    __slots__ = ("_by_owner", "_craft_state")

    def __init__(self) -> None:
        self._by_owner: dict[int, dict[str, Any]] = {}
        self._craft_state: dict[int, dict[str, Any]] = {}

    def bind(self, owner, name: str, symbol) -> None:
        """Bind `owner.<name>` to `symbol` for the duration of the trace."""
        self._by_owner.setdefault(id(owner), {})[name] = symbol

    def bind_craft_state(self, craft, syms: dict) -> None:
        """Bind a craft's rigid-body state symbols (compile Pass 0a)."""
        self._craft_state[id(craft)] = syms

    def craft_sym_state(self, craft) -> dict:
        """The bound rigid-body state symbols of `craft` for this trace."""
        try:
            return self._craft_state[id(craft)]
        except KeyError:
            raise RuntimeError(
                f"TraceBindings: craft '{craft.name}' has no symbolic state "
                f"bound — it is not part of the world being compiled.")

    def __enter__(self) -> "TraceBindings":
        if getattr(_trace_local, "active", None) is not None:
            raise RuntimeError(
                "TraceBindings: a trace is already active on this thread.")
        _trace_local.active = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _trace_local.active = None


def active_trace() -> "TraceBindings | None":
    """The trace active on this thread, if any. Compile-time hook for code
    that runs inside a trace without a `trace` handle in scope (e.g. a
    disturbance's `contribute_at_sym`)."""
    return getattr(_trace_local, "active", None)


def declared_attr(owner, name: str, default=None):
    """`getattr(owner, name, default)` that bypasses any active trace
    bindings — always the configured (numeric) instance value. For
    compile-time numeric snapshots that must not pick up a promoted
    parameter's symbol (rest-pose joint inertia, mass guards).

    The defaulting wrapper around `DeclarationHost.declared_value` — the
    one home of the `object.__getattribute__` trace bypass — so the two
    read idioms cannot drift apart."""
    try:
        return owner.declared_value(name)
    except AttributeError:
        return default


def is_promoted(value) -> bool:
    """True iff `value` reads as a trace-bound IR value (carries an
    `_mx` CasADi node) rather than a plain Python number/tuple — i.e.
    the attribute was promoted to a live graph input (tunable Parameter,
    bound State/Input/Noise) on the active trace. The single sniff every
    compile-time read site keys its symbolic-vs-numeric branch on."""
    return hasattr(value, "_mx")


def scalar_mx(value):
    """A possibly-promoted scalar attribute (or `ctx.dt`) as a raw MX
    node: the bound symbol when promoted, else a constant. The single
    home of the promoted-scalar read idiom — every part's compile-time
    scalar read goes through this instead of hand-rolling the branch."""
    import casadi as ca
    return value._mx if is_promoted(value) else ca.MX(float(value))
