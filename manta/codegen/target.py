"""The backend contract.

A backend implements exactly two things: a way to run/translate a
`ca.Function`, and one generic lowering of a typed `Module`
(`manta.ir.module`). Nothing in a backend mentions a transform — `Sim`,
`EKF`, `LQR`, and the recurrence blocks all reach it as Modules.

`as_module` is the one entry-point helper: backends accept either a
`Module` or any transform exposing `.module()`.
"""

from __future__ import annotations

from typing import Any, Callable

from ..ir.module import Module, Role, StateRef


def for_role(role: Role, handlers: dict, *, who: str):
    """Dispatch on an EntryPoint-argument `Role`, raising the one canonical
    `NotImplementedError` (naming the role + the `ARG_ROLES` contract) for
    any uncovered Role.

    `handlers` maps `Role → zero-arg callable` producing that role's value;
    the matched callable is invoked and its result returned. Centralizes the
    arg-dispatch ladder every interpreted/emitting backend would otherwise
    re-write — growing `Role` surfaces here once, not in N copies."""
    fn = handlers.get(role)
    if fn is None:
        raise NotImplementedError(
            f"{who}: role {role.name} unhandled — update {who} (and the "
            f"ARG_ROLES contract in manta.ir.module) for the new Role.")
    return fn()


def as_module(x, who: str) -> Module:
    """Return `x` as a `Module` (calling `x.module()` for a transform)."""
    if isinstance(x, Module):
        return x
    m = getattr(x, "module", None)
    if callable(m):
        mod = m()
        if isinstance(mod, Module):
            return mod
    raise TypeError(
        f"{who}: expected a Module or a transform with .module() "
        f"(Sim, EKF, LQR, a recurrence block), got {type(x).__name__}")


def resolve_args(module: Module, ep, values: dict,
                 *, state_lookup: Callable[[str], Any],
                 param_default: Callable[[], Any],
                 time_default: Any = 0.0) -> list:
    """Positional kernel args for entry `ep`, from a `{name: value}` dict,
    applying the defaults every interpreted backend shares: any arg named in
    `values` comes from there (a caller threading its own state supplies the
    `StateRef` names too); an unsupplied `StateRef` falls back to
    `state_lookup(name)`, an unsupplied TIME port to `time_default`, an
    unsupplied PARAMETER port to `param_default()`, and any other
    unsupplied port that declares an `init` to that declared default (an
    LQR's gain and feed-forward, which default to its built solve). Any
    other unsupplied PortRef is a KeyError, and a `values` key naming no
    argument of `ep` is a TypeError — a typo'd key must never be silently
    ignored. (The C++ backend emits typed source instead and does not use
    this; the JAX rollout walks roles symbolically for `lax.scan`.)"""
    args = []
    used = set()
    for a in ep.args:
        if a.name in values:
            args.append(values[a.name])
            used.add(a.name)
            continue
        if isinstance(a, StateRef):
            args.append(state_lookup(a.name))
            continue
        port = module.port(a.name)
        if port.role is Role.TIME:
            args.append(time_default)
        elif port.role is Role.PARAMETER:
            args.append(param_default())
        elif port.init is not None:
            args.append(port.init)
        else:
            raise KeyError(
                f"{module.name}.{ep.method}: missing value for port "
                f"{a.name!r}.")
    unknown = set(values) - used
    if unknown:
        raise TypeError(
            f"{module.name}.{ep.method}: unknown value key(s) "
            f"{sorted(unknown)}; this entry accepts "
            f"{[a.name for a in ep.args]}.")
    return args
